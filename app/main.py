import json
import os

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer
from pydantic import BaseModel

from app.auth.security import decode_jwt_token
from app.auth.users import verify_credentials, get_user_role, create_access_token
from app.guardrails.guardrail import check_input_guardrail
from app.pipeline.rag_chain import ask_question, stream_rag_question, get_retrieved_sources
from app.utils.audit_logger import log_audit_event
from app.utils.audit_reader import read_audit_events, summarize_events

APP_ENV = os.getenv("APP_ENV", "development").strip().lower()
_configured_origins = os.getenv("CORS_ALLOWED_ORIGINS", "").strip()
if _configured_origins:
    ALLOWED_ORIGINS = [origin.strip() for origin in _configured_origins.split(",") if origin.strip()]
elif APP_ENV == "production":
    raise RuntimeError("CORS_ALLOWED_ORIGINS must be configured in production.")
else:
    ALLOWED_ORIGINS = ["http://localhost:8501", "http://127.0.0.1:8501"]

if APP_ENV == "production" and ("*" in ALLOWED_ORIGINS or not ALLOWED_ORIGINS):
    raise RuntimeError("Production CORS origins must be explicit; wildcard origins are not allowed.")

app = FastAPI(
    title="Enterprise RBAC RAG API",
    description="Internal company RAG API with JWT authentication, hybrid retrieval and guarded responses.",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


class LoginRequest(BaseModel):
    username: str
    password: str


class ChatRequest(BaseModel):
    question: str


bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user(credentials=Depends(bearer_scheme)):
    """Authenticate identity from JWT, but always resolve current permissions from the DB."""
    if credentials is None or not getattr(credentials, "credentials", None):
        raise HTTPException(status_code=401, detail="Missing bearer token")

    payload = decode_jwt_token(credentials.credentials)
    username = payload.get("sub") if payload else None
    if not username:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    current_role = get_user_role(username)
    if not current_role:
        raise HTTPException(status_code=401, detail="User account is no longer available")

    # Ignore the role claim in the token: role changes take effect immediately.
    return {"username": username, "role": current_role}


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "vector_store": os.path.exists("./chroma_db"),
        "reranker": "ms-marco-MiniLM-L-6-v2",
        "cache": "enabled",
        "langsmith_tracing": os.getenv("LANGCHAIN_TRACING_V2", "").strip().lower() == "true",
        "audit_log": os.path.exists("security_audit.log"),
    }


@app.post("/api/v1/auth/login")
def login(request: LoginRequest):
    if not verify_credentials(request.username, request.password):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    role = get_user_role(request.username)
    token = create_access_token(request.username, role)
    return {
        "access_token": token,
        "token_type": "bearer",
        "username": request.username,
        "role": role,
    }


@app.post("/api/v1/chat")
def chat(request: ChatRequest, user=Depends(get_current_user)):
    username = user["username"]
    role = user["role"]

    violation = check_input_guardrail(request.question)
    if violation:
        log_audit_event(username, role, "CHAT", request.question, f"BLOCKED: {violation}", [])
        return {"response": violation, "sources": [], "guardrail_triggered": True}

    response = ask_question(username, request.question, input_guardrail_checked=True)
    sources = get_retrieved_sources(role, request.question)
    log_audit_event(username, role, "CHAT", request.question, "PASSED", sources)
    return {"response": response, "sources": sources, "role": role}


@app.get("/api/v1/chat/stream")
def chat_stream(question: str = Query(...), user=Depends(get_current_user)):
    username = user["username"]
    role = user["role"]

    def event_generator():
        yield f"data: {json.dumps({'event': 'metadata', 'role': role})}\n\n"
        violation = check_input_guardrail(question)
        if violation:
            log_audit_event(username, role, "STREAM", question, f"BLOCKED: {violation}", [])
            yield f"data: {json.dumps({'event': 'error', 'message': violation})}\n\n"
            return

        for safe_chunk in stream_rag_question(
            role, question, username=username, input_guardrail_checked=True
        ):
            yield f"data: {json.dumps({'event': 'token', 'token': safe_chunk})}\n\n"

        sources = get_retrieved_sources(role, question)
        log_audit_event(username, role, "STREAM", question, "PASSED", sources)
        yield f"data: {json.dumps({'event': 'done', 'sources': sources})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/v1/audit/summary")
def audit_summary(user=Depends(get_current_user)):
    """Admin-only aggregate view over the security audit log."""
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin role required")
    return summarize_events(read_audit_events())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
