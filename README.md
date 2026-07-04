# RBAC RAG Chatbot

![Python](https://img.shields.io/badge/python-3.11-blue)
![LangChain](https://img.shields.io/badge/LangChain-RAG-1C3C3C)
![ChromaDB](https://img.shields.io/badge/ChromaDB-vector--store-orange)
![Groq](https://img.shields.io/badge/LLM-Groq_Llama_3.1-green)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)

An internal company chatbot that answers questions over department documents (finance, HR, marketing, engineering) while enforcing **role-based access control at the retrieval layer** — a user only ever gets chunks their role is permitted to see. Adds input/output guardrails against PII leakage and out-of-scope queries, full observability via LangSmith tracing, and offline quality evaluation via Ragas.

---

## Architecture

**Query flow:**
1. User logs in → role resolved (`finance`, `hr`, `marketing`, `engineering`, `admin`)
2. Query passes input guardrails (PII check, scope check)
3. Retriever queries ChromaDB with a **role-based metadata filter** — non-permitted chunks never leave the vector store
4. Retrieved chunks + query sent to Groq Llama 3.1 via LangChain
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
| Groq (Llama 3.1) | LLM inference | Fast inference for a responsive chat experience |
| Ragas | Offline evaluation | Measures faithfulness, relevancy, recall, precision on a synthetic testset |
| LangSmith | Observability | Automatic tracing of every chain step with zero extra code |
| Streamlit | UI | Fast to build a login + chat interface for an internal tool |
| Docker | Containerization | Reproducible environment, ready for EC2 deployment |

---

## Key Technical Decisions

**RBAC via ChromaDB metadata filtering, not middleware or separate collections** — every chunk is tagged with a `role` during ingestion. At query time, the retriever applies a filter (`{"$or": [{"role": role}, {"role": "general"}]}`) so a finance user's search never returns HR chunks — they're excluded inside the vector search itself, not hidden after the fact. Admin bypasses the filter entirely. One collection is simpler to maintain than five.

**Two-layer guardrails — input and output** — Input: regex-based PII detection (email, phone, Aadhaar patterns) blocks the query before it reaches the LLM, plus an LLM-as-judge scope check ("is this a company-business question?") to reject off-topic queries. Output: the same PII regex is re-applied to the LLM's response and matches are redacted, catching anything that leaked through generation. This is not a full enterprise guardrail suite — no prompt-injection or jailbreak detection yet — but it covers the core production risk for an internal tool: accidental PII exposure.

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
    rag_chain.py          # Loads ChromaDB, applies RBAC filter, builds LangChain chain with Groq
  auth/
    users.py              # User store with roles (finance, hr, marketing, engineering, admin)
  guardrails/
    guardrail.py           # PII regex detection (input/output), LLM-as-judge scope check
  evaluation/
    evaluate.py           # Generates synthetic testset, runs Ragas metrics
app.py                     # Streamlit UI — login page, chat interface, source display
resources/
  data/                    # Raw department documents (finance, hr, marketing, engineering, general)
chroma_db/                 # Auto-generated by ingestion — not committed to GitHub
Dockerfile                 # python:3.11-slim, Streamlit on 0.0.0.0:8501
docker-compose.yml          # Volume mounts for chroma_db/ and resources/
.dockerignore               # Excludes venv, __pycache__, .env
```

---

## RBAC Design

Every document chunk is tagged at ingestion time:

```python
doc.metadata["role"] = department  # "finance", "hr", "marketing", "engineering", "general"
```

At query time, the retriever filters based on the logged-in user's role:

```python
search_kwargs = {"k": 3}
if role != "admin":
    search_kwargs["filter"] = {"$or": [{"role": role}, {"role": "general"}]}
```

- `general` documents (company-wide policies) are visible to every role
- `admin` bypasses the filter and can retrieve across all departments
- The filter runs **inside ChromaDB at vector search time** — restricted chunks are never fetched, not just hidden from the response

---

## Guardrails

**Input (checked before the query reaches the LLM):**
- PII detection via regex — email, 10-digit phone, Aadhaar-style patterns — blocks the query immediately if matched
- Scope check via an LLM-as-judge prompt ("Is this question related to company business — HR, finance, marketing, engineering? Answer only YES or NO.") — rejects off-topic queries

**Output (checked before the response is shown):**
- Same PII regex patterns applied to the LLM's response; any match is replaced with `[REDACTED]`

**Known gaps (by design, not yet implemented):**
- No prompt-injection detection
- No adversarial jailbreak detection
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
| Retrieval method | Cosine similarity, top-`k=3` |

**Identified improvement path (not yet implemented):** hybrid search (BM25 + semantic) and a cross-encoder re-ranker would likely improve context precision beyond the current score.

---

## Evaluation (Ragas)

Measured on a synthetic testset generated from the ingested documents:

| Metric | Score |
|---|---|
| Faithfulness | 0.93 |
| Answer Relevancy | 0.77 |
| Context Recall | 0.70 |
| Context Precision | 0.57 |

Faithfulness and relevancy are strong — the model isn't hallucinating and answers are on-topic. Context precision is the weakest metric, meaning retrieval sometimes pulls in chunks that aren't tightly relevant; a re-ranking step is the planned next improvement.

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

**Prerequisites:** Docker Desktop, Python 3.11+

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

**With Docker:**
```bash
docker compose up -d --build
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes | API key for Groq Llama 3.1 inference |
| `LANGCHAIN_API_KEY` | No | Enables LangSmith tracing |
| `LANGCHAIN_TRACING_V2` | No | Set to `true` to activate tracing |
| `LANGCHAIN_PROJECT` | No | LangSmith project name for grouping traces |

---

## Deployment Status

Currently runs locally via Docker Compose. `Dockerfile` and `docker-compose.yml` are written and tested with volume mounts for `chroma_db/` and `resources/`. EC2 deployment (`t2.medium`) is the next step — planned but not yet executed.

---

## What's Next

- Deploy to AWS EC2 with the existing Docker setup
- Add a FastAPI layer (`POST /query`, `POST /ingest`, `GET /health`) so the chatbot can be consumed as an API, not just through the Streamlit UI
- Add a cross-encoder re-ranker to improve context precision
- Add prompt-injection detection to the guardrail layer

---

## What I Learned Building This

- Enforcing access control *inside* a vector database query, not as a filter applied after retrieval
- Why LLM-as-judge is a practical way to catch out-of-scope queries without hand-writing rules for every topic
- The difference between guardrails (real-time, rule-based) and evaluation (offline, metric-based) — and why you need both
- How chunk size and overlap choices directly show up as measurable retrieval quality in Ragas scores
- Being honest about what a security layer *doesn't* cover is as important as documenting what it does
