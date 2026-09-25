# RBAC RAG Chatbot

![Python](https://img.shields.io/badge/python-3.11-blue)
![LangChain](https://img.shields.io/badge/LangChain-RAG-1C3C3C)
![ChromaDB](https://img.shields.io/badge/ChromaDB-vector--store-orange)
![Groq](https://img.shields.io/badge/LLM-Groq_gpt--oss--120b-green)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)

An internal company chatbot that answers questions over department documents (finance, HR, marketing, engineering) while enforcing **role-based access control at the retrieval layer** — a user only ever gets chunks their role is permitted to see. Adds input guardrails against PII leakage, prompt injection and out-of-scope queries plus output PII redaction, full observability via LangSmith tracing, and offline quality evaluation of the dense-only baseline against the shipped hybrid pipeline via Ragas.

---

## Architecture

**Query flow:**
1. User logs in → role resolved (`finance`, `hr`, `marketing`, `engineering`, `admin`)
2. Query passes input guardrails (PII, prompt-injection and scope checks)
3. Retriever queries ChromaDB with a **role-based metadata filter** — non-permitted chunks never leave the vector store
4. Retrieved chunks + query sent to Groq (`gpt-oss-120b`) via LangChain
5. Response passes output guardrails (PII redaction) before being shown
6. Every step traced automatically in LangSmith

**Ingestion flow:**
`.md` / `.csv` department documents → `RecursiveCharacterTextSplitter` → each chunk tagged with `role` metadata → embedded with `all-MiniLM-L6-v2` → stored in ChromaDB

---

## Tech Stack

| Technology | Purpose | Why chosen |
|---|---|---|
| LangChain | RAG orchestration | Chains retrieval, prompt formatting, and LLM calls with minimal boilerplate |
| ChromaDB | Vector store | Native metadata filtering — enables RBAC at the DB query level, not post-filtering |
| all-MiniLM-L6-v2 (HuggingFace) | Embeddings | Runs locally, no API cost, good quality for English business documents |
| Groq (`gpt-oss-120b`) | LLM inference | Fast inference for a responsive chat experience |
| Ragas | Offline evaluation | Measures faithfulness, relevancy, recall, precision on a synthetic testset |
| LangSmith | Observability | Automatic tracing of every chain step with zero extra code |
| Streamlit | UI | Fast to build a login + chat interface for an internal tool |

---

## Key Technical Decisions

**RBAC via ChromaDB metadata filtering, not middleware or separate collections** — every chunk is tagged with a `role` during ingestion. At query time, the retriever applies a filter (`{"$or": [{"role": role}, {"role": "general"}]}`) so a finance user's search never returns HR chunks — they're excluded inside the vector search itself, not hidden after the fact. Admin bypasses the filter entirely. One collection is simpler to maintain than five.

**Two-layer guardrails — input and output** — Input: regex-based PII detection (email, phone, Aadhaar patterns) blocks the query before it reaches the LLM, plus an LLM-as-judge scope check ("is this a company-business question?") to reject off-topic queries. Output: the same PII regex is re-applied to the LLM's response and matches are redacted, catching anything that leaked through generation. This is not a full enterprise guardrail suite — injection detection is pattern-based rather than a trained classifier, so novel phrasings can slip through — but it covers the core production risks for an internal tool: accidental PII exposure and trivial prompt overrides.

**RecursiveCharacterTextSplitter over naive character splitting** — 1000-character chunks with 200-character overlap, using separators that try paragraph breaks first, then lines, then sentences. This avoids cutting text mid-sentence, and the overlap means an answer spanning two chunks still has enough context in each.

**LangSmith for observability instead of custom logging** — Three environment variables enable full automatic tracing: every retrieval call, the exact prompt sent to the LLM, token counts, latency per step, and which chunks were fetched. This made debugging wrong answers (bad retrieval vs. bad generation) fast without writing any logging code.

**Ragas for offline quality measurement** — Rather than guessing whether retrieval is "good enough," a synthetic testset is generated and scored on faithfulness, answer relevancy, context recall, and context precision. This surfaced a concrete, measurable weak point (context precision) rather than a vague sense that answers "seemed fine."

---

## Project Structure

```
app/
  ingestion/
    ingest.py            # Loads .md/.csv docs, chunks, tags with role, stores in ChromaDB
  pipeline/
    rag_chain.py          # Hybrid retrieval + RBAC filter + Groq chain + semantic cache + streaming
  retrieval/
    hybrid_rerank.py      # BM25 search, Reciprocal Rank Fusion, cross-encoder re-ranking
  embeddings/
    local_embeddings.py   # Lazy CPU-only all-MiniLM-L6-v2 embeddings, local model folder first
  auth/
    users.py              # Bcrypt password hashing + role store (finance, hr, marketing, engineering, admin)
    security.py           # JWT access tokens carrying sub/role claims
    database.py, models.py # SQLite-backed user store (app_auth.db)
  guardrails/
    guardrail.py           # PII regex + prompt-injection regex + LLM-as-judge scope check
  evaluation/
    evaluate.py           # Synthetic testset + Ragas: dense-only baseline vs hybrid + re-rank
  ui/
    theme.py               # Stylesheet injection, avatars, role pills
    components.py          # Sidebar, chat header/history, empty state, sources, pipeline pill
    assets/custom.css      # Dark theme + component styles
  utils/
    audit_logger.py        # Structured security audit log
    audit_reader.py        # Reads/aggregates security_audit.log for the admin dashboard
  main.py                  # Optional FastAPI layer (JWT auth + SSE streaming)
app.py                     # Streamlit entrypoint — thin orchestration over app/ui
.streamlit/
  config.toml              # Declarative dark Streamlit theme
tools/
  ui_smoke_test.py         # Headless UI regression check
resources/
  data/                    # Raw department documents (finance, hr, marketing, engineering, general)
  test_data/               # Synthetic Ragas testset + baseline/upgraded result CSVs
  models/                  # Cached all-MiniLM-L6-v2 weights (git-ignored)
chroma_db/                 # Auto-generated by ingestion — not committed to GitHub
```

---

## RBAC Design

Every document chunk is tagged at ingestion time:

```python
doc.metadata["role"] = department  # "finance", "hr", "marketing", "engineering", "general"
```

At query time, the retriever filters based on the logged-in user's role:

```python
search_kwargs = {"k": 6}
if role != "admin":
    search_kwargs["filter"] = {"$or": [{"role": role}, {"role": "general"}]}
```

- `general` documents (company-wide policies) are visible to every role
- `admin` bypasses the filter and can retrieve across all departments
- The filter runs **inside ChromaDB at vector search time** — restricted chunks are never fetched, not just hidden from the response
- The **same predicate feeds the BM25 half** of hybrid retrieval: `bm25_search()` loads the role's permitted chunks from Chroma with the identical `$or` filter, so adding lexical search cannot widen access

---

## Guardrails

**Input (checked before the query reaches the LLM):**
- PII detection via regex — email, 10-digit phone, Aadhaar-style patterns — blocks the query immediately if matched
- Prompt-injection / jailbreak detection via a curated regex set (`ignore previous instructions`, `reveal your system prompt`, `system prompt override`, DAN-mode overrides, …) — rejects adversarial instructions before any model call
- Scope check via an LLM-as-judge prompt ("Is this question related to company business — HR, finance, marketing, engineering? Answer only YES or NO.") — rejects off-topic queries

**Output (checked before the response is shown):**
- Same PII regex patterns applied to the LLM's response; any match is replaced with `[REDACTED]`

**Known gaps (by design):**
- Injection/jailbreak detection is regex-based over a curated pattern list — clever paraphrases of an attack can bypass it
- Patterns are English-only, and the scope judge adds one LLM call of latency on the first (non-cached) query of a session
- Hallucination is not caught at the guardrail level — it's measured separately via Ragas offline evaluation

---

## Retrieval & Chunking

| Setting | Value |
|---|---|
| Splitter | `RecursiveCharacterTextSplitter` |
| Chunk size | 1000 characters |
| Chunk overlap | 200 characters |
| Separators | `["\n\n", "\n", ".", " ", ""]` |
| Embedding model | `all-MiniLM-L6-v2` (384-dim, local inference) |
| Dense candidates | Chroma cosine similarity, top-`k=6` |
| Sparse candidates | BM25 (`rank_bm25`) over the role's permitted chunks, top-6 |
| Fusion | Reciprocal Rank Fusion (`k=60`) → top-8 candidates |
| Re-ranker | Cross-encoder `ms-marco-MiniLM-L-6-v2` → top-3 chunks |

The hybrid + re-rank path is implemented and measured — see [Evaluation](#evaluation-ragas) below: context precision moves `0.50 → 0.94` and context recall `0.50 → 1.00` against the dense-only top-3 baseline.

---

## Evaluation (Ragas)

Two pipelines are scored in one harness run: the **dense-only top-3 baseline** and the **shipped hybrid + cross-encoder pipeline**. Both are judged by Ragas 0.4 (Groq `gpt-oss-120b`) over the synthetic testset, so the improvement is measured rather than assumed. Full methodology, per-metric analysis and caveats: [`evaluation_report.md`](evaluation_report.md).

| Metric | Upgraded (baseline in brackets) |
|---|---|
| Faithfulness | 0.87 |
| Answer Relevancy | 0.79 |
| Context Recall | **1.00** (baseline `0.50`) |
| Context Precision | **0.94** (baseline `0.50`) |

Context precision and context recall both sat at `0.50` with dense-only top-3 — retrieval, not generation, was the bottleneck — and the hybrid + re-ranking upgrade lifts precision to `0.94` and recall to `1.00`. Faithfulness (`0.87`) and relevancy (`0.79`) confirm answers stay grounded and on-topic. These numbers come from a 10-question sample (`RAGAS_SAMPLE_SIZE=10`) and one upgraded precision row is `NaN`; re-run the harness for the full 30-question set.

---

## Observability (LangSmith)

Enabled with three environment variables — no code changes required. Every query automatically logs:
- The retrieved chunks for that query
- The exact prompt sent to the LLM
- Token usage (input/output) and estimated cost
- Latency per step (retrieval vs. LLM inference)
- Errors and retries

Used primarily to debug whether a wrong answer came from bad retrieval or bad generation.

---

## Running Locally

**Prerequisites:** Python 3.11+ (developed against 3.13)

```bash
git clone https://github.com/muralikrishna1729/RAG_RBAC_Guardrails_Monitoring.git
cd RAG_RBAC_Guardrails_Monitoring

python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Add GROQ_API_KEY and LangSmith keys to .env

python -m app.ingestion.ingest   # builds chroma_db/ from resources/data/
streamlit run app.py
```

Open `http://localhost:8501` and log in with a demo user to start chatting.

> Containerization is **not committed to this repo yet** — there is no `Dockerfile` or `docker-compose.yml`, so use the commands above to run it locally.

---

## UI & Theming

The Streamlit app ships with a custom dark theme split out of `app.py` so the presentation layer can evolve without touching RAG/guardrail/auth logic:

| File | Purpose |
|---|---|
| `.streamlit/config.toml` | Base Streamlit theme (dark palette, `#818cf8` accent) |
| `app/ui/theme.py` | Injects the stylesheet + `AVATARS` / role-pill helpers |
| `app/ui/assets/custom.css` | The stylesheet itself — chat bubbles, avatars, metric cards, sidebar, pills |
| `app/ui/components.py` | Reusable UI blocks: sidebar, chat header, empty state, sources, pipeline pill |
| `tools/ui_smoke_test.py` | Headless UI regression check (`python tools/ui_smoke_test.py`) |

The UI includes gradient title, emoji chat avatars with per-speaker bubble tints, a coloured role badge, KPI-style metric cards in the sidebar and security dashboard, an animated "retrieving" pill while an answer streams, an empty-state hint card, and a styled login card.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes | API key for Groq (`gpt-oss-120b`) inference and the guardrail scope judge |
| `JWT_SECRET_KEY` | Yes (API) | Signing secret for JWT access tokens — replace the built-in default before any real deployment |
| `ALGORITHM` | No | JWT signing algorithm (default `HS256`) |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | No | Access-token lifetime in minutes (default `60`) |
| `COMPANY_DOMAIN` | No | Trusted email domain the output guardrail leaves un-redacted (default `company.com`) |
| `DATABASE_URL` | No | SQLAlchemy URL for the user store (default `sqlite:///./app_auth.db`) |
| `GEMINI_API_KEY` | No | Only needed to regenerate the synthetic Ragas testset (`gemini-2.0-flash`) |
| `RAGAS_SAMPLE_SIZE` | No | Evaluate only the first N testset questions; unset = all |
| `SEMANTIC_CACHE_THRESHOLD` | No | Cosine similarity required for a semantic-cache hit (default `0.92`) |
| `LANGCHAIN_API_KEY` | No | Enables LangSmith tracing |
| `LANGCHAIN_TRACING_V2` | No | Set to `true` to activate tracing |
| `LANGCHAIN_ENDPOINT` | No | LangSmith endpoint (default `https://api.smith.langchain.com`) |
| `LANGCHAIN_PROJECT` | No | LangSmith project name for grouping traces |

---

## Deployment Status

Currently runs locally only. The `Dockerfile` / `docker-compose.yml` setup is **not in the repository**, so the containerized path — and the EC2 deployment that depends on it — is still outstanding: writing and testing the compose setup is the prerequisite step.

---

## What's Next

- Add the `Dockerfile` / `docker-compose.yml` setup, then deploy to AWS EC2 (`t2.medium`)
- Extend the FastAPI layer in `app/main.py` with an ingestion endpoint — JWT auth, chat, SSE streaming and the admin audit summary already exist
- Scale the Ragas evaluation from the 10-question sample to the full testset, and add answer-side baseline numbers for faithfulness/relevancy
- Move injection detection beyond regex patterns (trained classifier) and cover non-English phrasings

---

## What I Learned Building This

- Enforcing access control *inside* a vector database query, not as a filter applied after retrieval
- Why LLM-as-judge is a practical way to catch out-of-scope queries without hand-writing rules for every topic
- The difference between guardrails (real-time, rule-based) and evaluation (offline, metric-based) — and why you need both
- How chunk size and overlap choices directly show up as measurable retrieval quality in Ragas scores
- Being honest about what a security layer *doesn't* cover is as important as documenting what it does
