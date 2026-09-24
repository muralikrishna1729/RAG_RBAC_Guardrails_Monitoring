import os
import json
from typing import Dict
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Query, status, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from app.auth.users import verify_credentials, get_user_role, create_access_token, get_all_demo_users
from app.guardrails.guardrail import check_input_guardrail
from app.pipeline.rag_chain import ask_question, stream_rag_question, get_retrieved_sources
from app.utils.audit_logger import log_audit_event

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
    username: str
    question: str
security = HTTPBasic()

# Dummy user database
users_db: Dict[str, Dict[str, str]] = {
    "Tony": {"password": "password123", "role": "engineering"},
    "Bruce": {"password": "securepass", "role": "marketing"},
    "Sam": {"password": "financepass", "role": "finance"},
    "Peter": {"password": "pete123", "role": "engineering"},
    "Sid": {"password": "sidpass123", "role": "marketing"},
    "Natasha": {"passwoed": "hrpass123", "role": "hr"}
}


# Authentication dependency
def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    username = credentials.username
    password = credentials.password
    user = users_db.get(username)
    if not user or user["password"] != password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"username": username, "role": user["role"]}


# Login endpoint
@app.get("/login")
def login(user=Depends(authenticate)):
    return {"message": f"Welcome {user['username']}!", "role": user["role"]}


# Protected test endpoint
@app.get("/test")
def test(user=Depends(authenticate)):
    return {"message": f"Hello {user['username']}! You can now chat.", "role": user["role"]}


# Protected chat endpoint
@app.post("/chat")
def query(user=Depends(authenticate), message: str = "Hello"):
    return "Implement this endpoint."


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "vector_store": os.path.exists("./chroma_db"),
        "reranker": "ms-marco-MiniLM-L-6-v2",
        "cache": "enabled"
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
def chat(request: ChatRequest):
    role = get_user_role(request.username)
    if not role:
        raise HTTPException(status_code=403, detail="User not recognized")
    
    violation = check_input_guardrail(request.question)
    if violation:
        log_audit_event(request.username, role, "CHAT", request.question, f"BLOCKED: {violation}", [])
        return {"response": f"{violation}", "sources": [], "guardrail_triggered": True}

    response = ask_question(request.username, request.question)
    sources = get_retrieved_sources(role, request.question)
    log_audit_event(request.username, role, "CHAT", request.question, "PASSED", sources)
    return {"response": response, "sources": sources, "role": role}

@app.get("/api/v1/chat/stream")
def chat_stream(username: str = Query(...), question: str = Query(...)):
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
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)