import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.utils.audit_reader import read_audit_events, summarize_events


class AuditReaderTests(unittest.TestCase):
    def test_reads_rotated_and_active_logs_and_supports_old_role_key(self):
        with tempfile.TemporaryDirectory() as directory:
            streamlit_log = Path(directory) / "streamlit.log"
            api_log = Path(directory) / "api.log"

            streamlit_log.with_name("streamlit.log.1").write_text(
                json.dumps({
                    "timestamp": "2026-01-01T00:00:00+00:00",
                    "username": "alice",
                    "role_claim": "finance",
                    "action": "CHAT",
                    "guardrail_status": "PASSED",
                }) + "\n",
                encoding="utf-8",
            )
            streamlit_log.write_text(
                json.dumps({
                    "timestamp": "2026-01-02T00:00:00+00:00",
                    "username": "bob",
                    "role": "hr",
                    "action": "CHAT",
                    "guardrail_status": "BLOCKED: pii",
                }) + "\n",
                encoding="utf-8",
            )
            api_log.write_text(
                json.dumps({
                    "timestamp": "2026-01-03T00:00:00+00:00",
                    "username": "david",
                    "role": "engineering",
                    "action": "STREAM",
                    "guardrail_status": "PASSED",
                }) + "\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {
                "AUDIT_LOG_PATHS": f"{streamlit_log},{api_log}",
                "AUDIT_LOG_PATH": "",
            }):
                events = read_audit_events()

            summary = summarize_events(events)
            self.assertEqual(summary["total_events"], 3)
            self.assertEqual(summary["passed"], 2)
            self.assertEqual(summary["blocked"], 1)
            self.assertEqual(summary["by_role"], {
                "finance": 1,
                "hr": 1,
                "engineering": 1,
            })
            self.assertEqual(summary["last_event_at"], "2026-01-03T00:00:00+00:00")


if __name__ == "__main__":
    unittest.main()
