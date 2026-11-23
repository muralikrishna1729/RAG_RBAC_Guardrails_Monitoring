"""Read and aggregate structured security-audit events from the active and rotated logs."""
import json
import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _audit_log_path() -> Path:
    return Path(os.getenv("AUDIT_LOG_PATH", str(_PROJECT_ROOT / "security_audit.log")))


def read_audit_events() -> list:
    """Return parseable events from rotated files in chronological file order."""
    log_path = _audit_log_path()
    events = []
    paths = [
        Path(f"{log_path}.{index}")
        for index in range(5, 0, -1)
    ] + [log_path]

    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as audit_file:
            for line in audit_file:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return events


def summarize_events(events: list) -> dict:
    """Aggregate current and legacy audit event formats."""
    from collections import Counter

    def is_blocked(event: dict) -> bool:
        return str(event.get("guardrail_status", "")).startswith("BLOCKED")

    def event_role(event: dict) -> str:
        return event.get("role", event.get("role_claim", "unknown"))

    by_status = Counter("blocked" if is_blocked(event) else "passed" for event in events)
    blocked_reasons = Counter(
        event["guardrail_status"] for event in events if is_blocked(event)
    )

    return {
        "total_events": len(events),
        "passed": by_status.get("passed", 0),
        "blocked": by_status.get("blocked", 0),
        "by_role": dict(Counter(event_role(event) for event in events)),
        "by_user": dict(Counter(event.get("username", "unknown") for event in events)),
        "by_action": dict(Counter(event.get("action", "unknown") for event in events)),
        "blocked_reasons": dict(blocked_reasons),
        "last_event_at": events[-1].get("timestamp") if events else None,
    }
