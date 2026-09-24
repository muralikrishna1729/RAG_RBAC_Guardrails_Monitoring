import os
import json
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.security import HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from app.auth.security import decode_jwt_token
from app.auth.users import verify_credentials, get_user_role, create_access_token
from app.guardrails.guardrail import check_input_guardrail
from app.pipeline.rag_chain import ask_question, stream_rag_question, get_retrieved_sources
from app.utils.audit_logger import log_audit_event
from app.utils.audit_reader import read_audit_events, summarize_events

app = FastAPI(title="Enterprise RBAC RAG API",
    description="Production-grade RAG API with JWT Auth, Hybrid Search, Cross-Encoder Re-Ranking, and Token Streaming.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins =["*"],
    allow_credentials = True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LoginRequest(BaseModel):
    username: str
    password: str

class ChatRequest(BaseModel):
    question: str

bearer_scheme = HTTPBearer(auto_error=False)


# JWT enforcement: identity comes ONLY from the signed token, never from the request body
def get_current_user(credentials=Depends(bearer_scheme)):
    if credentials is None or not getattr(credentials, "credentials", None):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    payload = decode_jwt_token(credentials.credentials)
    if not payload or "sub" not in payload or "role" not in payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return {"username": payload["sub"], "role": payload["role"]}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "vector_store": os.path.exists("./chroma_db"),
        "reranker": "ms-marco-MiniLM-L-6-v2",
        "cache": "enabled",
        "langsmith_tracing": os.getenv("LANGCHAIN_TRACING_V2", "").strip().lower() == "true",
        "audit_log": os.path.exists("security_audit.log")
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
        "role": role
    }

@app.post("/api/v1/chat")
def chat(request: ChatRequest, user = Depends(get_current_user)):
    username = user["username"]
    role = get_user_role(username)
    if not role:
        raise HTTPException(status_code=403, detail="User not recognized")
    
    violation = check_input_guardrail(request.question)
    if violation:
        log_audit_event(username, role, "CHAT", request.question, f"BLOCKED: {violation}", [])
        return {"response": f"{violation}", "sources": [], "guardrail_triggered": True}

    response = ask_question(username, request.question)
    sources = get_retrieved_sources(role, request.question)
    log_audit_event(username, role, "CHAT", request.question, "PASSED", sources)
    return {"response": response, "sources": sources, "role": role}

@app.get("/api/v1/chat/stream")
def chat_stream(question: str = Query(...), user = Depends(get_current_user)):
    username = user["username"]
    role = get_user_role(username)
    if not role:
        raise HTTPException(status_code=403, detail="User not recognized")

    def event_generator():
        yield f"data: {json.dumps({'event': 'metadata', 'role': role})}\n\n"
        violation = check_input_guardrail(question)
        if violation:
            log_audit_event(username, role, "STREAM", question, f"BLOCKED: {violation}", [])
            yield f"data: {json.dumps({'event': 'error', 'message': f'⚠️ {violation}'})}\n\n"
            return
            
        for token in stream_rag_question(role, question):
            yield f"data: {json.dumps({'event': 'token', 'token': token})}\n\n"
            
        sources = get_retrieved_sources(role, question)
        log_audit_event(username, role, "STREAM", question, "PASSED", sources)
        yield f"data: {json.dumps({'event': 'done', 'sources': sources})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/v1/audit/summary")
def audit_summary(user = Depends(get_current_user)):
    """Admin-only aggregate view over the security audit log."""
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin role required")
    events = read_audit_events()
    return summarize_events(events)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)