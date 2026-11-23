# Production hardening notes

This deployment is for one company. Authorization is department-based (HR, Finance, Engineering, Marketing, General and Admin); there is no tenant model or tenant isolation layer.

## Production environment

Set these in the deployment environment or .env:

- APP_ENV=production
- JWT_SECRET_KEY to a unique random secret of at least 32 characters
- CORS_ALLOWED_ORIGINS to the exact HTTPS frontend origin(s), comma-separated
- GROQ_API_KEY to the production API key
- ENABLE_DEMO_USERS=false
- DOCKER_DATABASE_URL to the intended persistent container database URL (optional; defaults to persistent SQLite)

The Compose file defaults to a single persistent SQLite database at /app/data/app_auth.db, shared by the Streamlit and FastAPI services. For a small single-company deployment this keeps operations simple. Use PostgreSQL only if expected concurrency or operational requirements justify it; configure it through DOCKER_DATABASE_URL.

Production startup fails if the JWT secret is missing/too short or CORS origins are not explicitly configured. Do not use the development fallback secret in a deployed environment.

## Create the first administrator

Demo accounts are seeded only in development/test. Create the first production administrator interactively:

```bash
docker compose exec fastapi python -m app.auth.bootstrap_admin
```

The command prompts for a username and password; it does not accept a default password or print the password. Run it once against the persistent database.

## Document ingestion and activation

Run ingestion from the repository root:

```bash
python -m app.ingestion.ingest
```

The ingestion process now:

1. Calculates stable document IDs and SHA-256 source-content hashes.
2. Calculates document-version and deterministic chunk IDs.
3. Acquires a single-writer lock in the Chroma persistence directory.
4. Writes into a new versioned Chroma collection without deleting the active collection.
5. Checks the stored chunk count.
6. Atomically switches the active-collection marker only after validation succeeds.

If staging fails, the previous active collection remains selected. Previous collections are intentionally retained for rollback; a retention/cleanup command is still required before long-term use. If the process crashes while holding the writer lock, verify no ingestion process is running before manually removing chroma_db/.ingestion.lock.

## Security and response behavior

- API authorization resolves the current role from the database on every request; JWT role claims are informational only.
- Streamed answers are buffered and PII-redacted before being released. SSE remains available, but token-by-token display is intentionally disabled until a safe rolling redactor is implemented.
- Answer generation and source citations now use the same retrieved document objects.
- Semantic and BM25 caches are scoped to the active index generation.
- Embedding batches retry with smaller batches and validate vector count, dimension and finite values.
- Audit logs are JSON lines, rotate at 5 MB with five backups, and omit query/document contents. Compose keeps API and Streamlit audit files separate to avoid multi-process rotation races.

## Current limitations to address before a wider rollout

- Per-document version lifecycle/job records are not yet stored in SQL; the current version pointer is index/collection-level, with per-document hashes in chunk metadata.
- Old Chroma collections are retained; automated retention cleanup and rollback tooling are still pending.
- A stale ingestion lock after a hard process crash requires operator verification/removal.
- The local persistent Chroma deployment still needs an operational decision about concurrent access from multiple containers. Keep ingestion as a single writer and test this deployment topology under realistic load.
- The readiness endpoint still needs real database/vector-store checks.
- Bounded LLM retries/timeouts and the final regression/load test suite are still pending.

## Verification

```bash
python -m compileall -q app
python -m unittest discover -s tests -v
python tools/ui_smoke_test.py
```

Then run a full ingestion against a copy of the current data, verify the active collection changes only after successful validation, and test each department role against cross-department documents before deployment.
