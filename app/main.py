import json
import os

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer
from pydantic import BaseModel
from sqlalchemy import text
from langchain_chroma import Chroma

from app.auth.database import engine
from app.auth.security import decode_jwt_token
from app.auth.users import verify_credentials, get_user_role, create_access_token
from app.embeddings import get_embeddings
from app.guardrails.guardrail import check_input_guardrail
from app.pipeline.rag_chain import generate_response_with_sources
from app.utils.audit_logger import log_audit_event
from app.utils.audit_reader import read_audit_events, summarize_events
from app.utils.index_state import get_active_collection_name, get_index_generation

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
if APP_ENV == "production" and not os.getenv("GROQ_API_KEY", "").strip():
    raise RuntimeError("GROQ_API_KEY must be configured in production.")

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
    return {"username": username, "role": current_role}


@app.get("/health/live")
def health_live():
    """Liveness only: no database, vector-store or model loading."""
    return {"status": "alive"}


@app.get("/health/ready")
def health_ready():
    """Readiness checks local critical dependencies without calling the external LLM."""
    checks = {}

    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception:
        checks["database"] = "error"

    try:
        persist_directory = "./chroma_db"
        if not os.path.isdir(persist_directory):
            raise RuntimeError("Vector store directory is missing")
        vectorstore = Chroma(
            collection_name=get_active_collection_name(persist_directory),
            persist_directory=persist_directory,
            embedding_function=get_embeddings(),
        )
        chunk_count = vectorstore._collection.count()
        if chunk_count < 1:
            raise RuntimeError("Active vector collection is empty")
        checks["vector_store"] = {"status": "ok", "chunks": chunk_count}
        generation = get_index_generation(persist_directory)
    except Exception:
        checks["vector_store"] = {"status": "error"}
        generation = None

    ready = checks.get("database") == "ok" and checks.get("vector_store", {}).get("status") == "ok"
    payload = {
        "status": "ready" if ready else "not_ready",
        "checks": checks,
        "index_generation": generation,
    }
    return JSONResponse(status_code=200 if ready else 503, content=payload)


@app.get("/health")
def health_check():
    """Compatibility health route used by the Docker healthcheck."""
    return health_ready()


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

    response, sources = generate_response_with_sources(
        role,
        request.question,
        username=username,
        input_guardrail_checked=True,
    )
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

        response, sources = generate_response_with_sources(
            role,
            question,
            username=username,
            input_guardrail_checked=True,
        )
        yield f"data: {json.dumps({'event': 'token', 'token': response})}\n\n"
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
