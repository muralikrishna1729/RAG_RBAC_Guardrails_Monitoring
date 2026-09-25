# 📊 Ragas Offline Quality Evaluation & Benchmark Report

This document presents a measured comparison of the **Baseline RAG Architecture** (dense vector search only, top-3) against the **Upgraded Hybrid Search + Cross-Encoder Re-Ranked RAG Architecture** that the codebase now ships.

Both pipelines are evaluated in the same run by [`app/evaluation/evaluate.py`](file:///c:/Users/Lenovo/Music/Projects/RAG_RBAC_Guardrails_Monitoring/app/evaluation/evaluate.py), so the comparison is reproducible rather than asserted.

---

## 🧪 Methodology

| Item | Value |
|---|---|
| Framework | Ragas 0.4 (`Faithfulness`, `AnswerRelevancy`, `ContextRecall`, `ContextPrecision`) |
| Judge LLM | Groq `openai/gpt-oss-120b`, `temperature=0` |
| Judge embeddings | Local CPU `all-MiniLM-L6-v2` (same model as retrieval) |
| Testset | `resources/test_data/synthetic_rag_testset.csv` — 30 synthetic Q&A pairs |
| Evaluated sample | First **10** questions (`RAGAS_SAMPLE_SIZE=10`) |
| Retrieval role | `admin` (no RBAC filter) so both pipelines draw from the same candidate pool |
| Artifacts | `resources/test_data/ragas_results.csv` (upgraded) · `resources/test_data/ragas_baseline.csv` (baseline) |

**Baseline pipeline:** `Chroma.as_retriever(k=3)` — dense cosine similarity only.
**Upgraded pipeline:** dense top-6 **+** BM25 top-6 → Reciprocal Rank Fusion → 8 candidates → cross-encoder (`ms-marco-MiniLM-L-6-v2`) → top-3 chunks.

The baseline is retrieval-only, so it is scored on the two retrieval-side metrics (`Context Precision`, `Context Recall`) against the same ground-truth references. `Faithfulness` and `Answer Relevancy` are generation-side metrics and are therefore only reported for the upgraded pipeline.

---

## 📈 Benchmark Summary Table

| Ragas Quality Metric | Baseline RAG (Vector Only, Top-3) | Upgraded RAG (BM25 + Dense RRF + Cross-Encoder) | Change |
|---|---|---|---|
| **Context Precision** | **0.50** | **0.94** | **+0.44** — noise in the retrieved chunks largely removed |
| **Context Recall** | **0.50** | **1.00** | **+0.50** — every required fact was retrieved |
| **Faithfulness** | _not scored_ (retrieval-only) | **0.87** | — |
| **Answer Relevancy** | _not scored_ (retrieval-only) | **0.79** | — |

Values are means over the evaluated questions, read directly from the two committed CSVs.

---

## 🔍 Metric Definitions & Analysis

### 1. Context Precision (0.50 → 0.94)
* **What it measures**: signal-to-noise ratio of the retrieved chunks — are the top-ranked chunks actually relevant to the question?
* **Why the baseline scored 0.50**: pure dense similarity (`all-MiniLM-L6-v2`) retrieves chunks that match the *tone* of the question but miss the exact identifiers — policy labels, fiscal quarters, employee IDs, acronyms. Per-question baseline precision was `0 0 1 0 0 0 1 1 1 1`.
* **How the upgrade fixed it**: BM25 contributes exact-token matches into the RRF candidate pool, and the cross-encoder then scores each `(query, chunk)` pair jointly, pushing genuinely relevant chunks into the top-3.

### 2. Context Recall (0.50 → 1.00)
* **What it measures**: whether the retriever fetched all the context needed to answer, judged against the ground truth.
* **Analysis**: the baseline missed the required chunk for half of the sample. The hybrid candidate pool (dense + lexical, 8 fused candidates before re-ranking) covered the ground-truth facts for every evaluated question. Note this moves **both** retrieval metrics — the earlier report framed precision as the sole bottleneck, but recall was equally weak at `0.50`, which is why widening the candidate pool matters as much as re-ordering it.

### 3. Faithfulness (0.87)
* **What it measures**: proportion of claims in the answer that are grounded in the retrieved context (hallucination detector).
* **Analysis**: `temperature=0` plus the prompt's explicit "answer strictly based on the context / say I don't have that information" fallback keeps groundedness high. The remaining headroom sits in a minority of answers where the model adds surrounding detail — per-question scores ranged 0.33 – 1.00.

### 4. Answer Relevancy (0.79)
* **What it measures**: how directly the answer addresses the question, without padding or evasion.
* **Analysis**: cleaner context feeds more focused completions. `AnswerRelevancy` runs with `strictness=1` because Groq rejects `n>1` completions, which Ragas' default relies on. The spread (0.26 – 1.00) shows the weakest answers are the ones where the model hedges instead of answering.

---

## ⚠️ Honest Caveats

* **Sample size is 10 questions**, not the full 30-question testset — small enough that a single question moves a mean by roughly 0.1. Drop `RAGAS_SAMPLE_SIZE` to score the whole set.
* **One upgraded `context_precision` row is `NaN`** (the judge returned no verdict for that pair). Ragas means skip NaNs, so the upgraded precision mean is computed over 9 rows.
* **No answer-side baseline**: the baseline harness collects retrieved contexts only, so the improvement is established for retrieval quality. `Faithfulness` / `Answer Relevancy` have no baseline counterpart in these artifacts.
* **Judge variance**: LLM-as-judge scoring is not perfectly deterministic even at `temperature=0`; absolute values drift slightly between runs.

---

## 🛠️ Reproduction Commands

```bash
# Full run over the whole synthetic testset
python -m app.evaluation.evaluate

# Faster run over the first N questions (what produced the committed CSVs)
RAGAS_SAMPLE_SIZE=10 python -m app.evaluation.evaluate      # bash
$env:RAGAS_SAMPLE_SIZE=10; python -m app.evaluation.evaluate # PowerShell
```

Outputs:

| File | Contents |
|---|---|
| `resources/test_data/ragas_results.csv` | Per-question upgraded scores + answers + re-ranked contexts |
| `resources/test_data/ragas_baseline.csv` | Per-question dense-only top-3 contexts + context metrics |
