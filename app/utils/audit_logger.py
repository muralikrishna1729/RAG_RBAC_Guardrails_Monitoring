import json
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4

_AUDIT_PATH = Path(os.getenv("AUDIT_LOG_PATH", "security_audit.log"))
_AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)

audit_logger = logging.getLogger("security_audit")
audit_logger.setLevel(logging.INFO)
audit_logger.propagate = False

if not any(getattr(handler, "_is_security_audit_handler", False) for handler in audit_logger.handlers):
    handler = RotatingFileHandler(
        _AUDIT_PATH,
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    handler._is_security_audit_handler = True
    handler.setFormatter(logging.Formatter("%(message)s"))
    audit_logger.addHandler(handler)


def _source_label(source: str) -> str:
    """Keep source identity in audit records without copying retrieved text snippets."""
    return str(source).split(":", 1)[0][:200]


def log_audit_event(
    username: str,
    role: str,
    action: str,
    question: str,
    guardrail_status: str,
    sources: list,
):
    """Write one structured audit event without logging prompts or document contents."""
    event = {
        "event_id": uuid4().hex,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "username": username,
        "role": role,
        "action": action,
        "query_length": len(question or ""),
        "guardrail_status": guardrail_status,
        "retrieved_sources_count": len(sources or []),
        "accessed_sources": [_source_label(source) for source in (sources or [])[:3]],
    }
    audit_logger.info(json.dumps(event, ensure_ascii=False))
