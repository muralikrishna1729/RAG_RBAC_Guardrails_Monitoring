import json 
import logging 
import time 
from datetime import datetime

audit_logger = logging.getLogger("security_audit")
audit_logger.setLevel(logging.INFO)
handler = logging.FileHandler("security_audit.log")
handler.setFormatter(logging.Formatter('%(message)s'))
audit_logger.addHandler(handler)


def log_audit_event(username:str, role:str, action:str, question:str, guardrail_status:str, sources:list):
    """Logs a structured JSON security audit event"""
    event = {
        "timestamp": time.time().isoformat() + "Z",
        "username": username,
        "role_claim": role,
        "action": action,
        "query_length": len(question),
        "guardrail_status": guardrail_status,
        "retrieved_sources_count": len(sources),
        "accessed_sources": sources[:3]
    }
    audit_logger.info(json.dumps(event))
    print(f"🔒 AUDIT LOG: [{event['timestamp']}] User={username} Role={role} Guardrail={guardrail_status}")
