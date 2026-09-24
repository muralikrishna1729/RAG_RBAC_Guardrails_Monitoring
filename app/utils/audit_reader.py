"""Reads and aggregates structured security-audit events from the audit log.

Shared by the Streamlit security dashboard and the /api/v1/audit/summary API.
The log path can be overridden with the AUDIT_LOG_PATH environment variable.
"""

import json
import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _audit_log_path() -> Path:
    return Path(os.getenv("AUDIT_LOG_PATH", str(_PROJECT_ROOT / "security_audit.log")))


def read_audit_events() -> list:
    """Returns all parseable audit events (one JSON object per log line)."""
    log_path = _audit_log_path()
    events = []
    if not log_path.exists():
        return events
    with log_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # skip malformed / legacy lines
    return events


def summarize_events(events: list) -> dict:
    """Aggregates audit events into dashboard/metrics-friendly counts."""
    from collections import Counter

    def is_blocked(event: dict) -> bool:
        return str(event.get("guardrail_status", "")).startswith("BLOCKED")

    by_status = Counter("blocked" if is_blocked(e) else "passed" for e in events)
    blocked_reasons = Counter(
        e["guardrail_status"] for e in events if is_blocked(e)
    )

    return {
        "total_events": len(events),
        "passed": by_status.get("passed", 0),
        "blocked": by_status.get("blocked", 0),
        "by_role": dict(Counter(e.get("role_claim", "unknown") for e in events)),
        "by_user": dict(Counter(e.get("username", "unknown") for e in events)),
        "by_action": dict(Counter(e.get("action", "unknown") for e in events)),
        "blocked_reasons": dict(blocked_reasons),
        "last_event_at": events[-1].get("timestamp") if events else None,
    }
