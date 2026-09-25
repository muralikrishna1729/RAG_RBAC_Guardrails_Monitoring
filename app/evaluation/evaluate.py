import os
import sys
import pandas as pd
from dotenv import load_dotenv

# Console-safe printing (PowerShell pipes default to cp1252, which chokes on box-drawing chars)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langchain_community.document_loaders import DirectoryLoader, TextLoader, CSVLoader
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision

from datasets import Dataset

from app.embeddings import get_embeddings
from app.pipeline.rag_chain import build_rag_chain, retrieve_contexts

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_PATH    = "./resources/data"
TESTSET_PATH = "./resources/test_data/synthetic_rag_testset.csv"
CHROMA_PATH  = "./chroma_db"


# ── Shared wrappers (created once, reused everywhere) ─────────────────────────
def get_ragas_llm():
    return LangchainLLMWrapper(
        ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0,
        )
    )

def get_ragas_embeddings():
    return LangchainEmbeddingsWrapper(get_embeddings())


def generate_test_data(data_path: str) -> pd.DataFrame:
    """Load documents and generate a synthetic Q&A testset using Ragas."""

    md_loader = DirectoryLoader(
        data_path,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    csv_loader = DirectoryLoader(
        data_path,
        glob="**/*.csv",
        loader_cls=CSVLoader
    )

    documents = md_loader.load() + csv_loader.load()
    print(f"Loaded {len(documents)} documents for test generation.")

    # Use Gemini for generation (better quality than Groq for this task)
    generation_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.7
        )
    )

    embeddings = get_ragas_embeddings()

    generator = TestsetGenerator(
        llm=generation_llm,
        embedding_model=embeddings
    )

    run_config = RunConfig(
        max_retries=10,
        max_wait=90,
        timeout=120,
    )

    testset = generator.generate_with_langchain_docs(
        documents,
        testset_size=10,
        run_config=run_config
    )

    test_df = testset.to_pandas()
    print(f"\nGenerated {len(test_df)} test samples.")
    print(test_df[["question", "ground_truth"]].head())

    os.makedirs(os.path.dirname(TESTSET_PATH), exist_ok=True)
    test_df.to_csv(TESTSET_PATH, index=False)
    print(f"Testset saved to {TESTSET_PATH}")

    return test_df


def collect_rag_results(test_df: pd.DataFrame) -> dict:
    """Run the hybrid RAG chain per question; collect answers + reranked contexts."""

    rag_chain, _ = build_rag_chain(
        persist_directory=CHROMA_PATH,
        role="admin"
    )

    questions    = test_df["question"].tolist()
    ground_truths = test_df["ground_truth"].tolist() if "ground_truth" in test_df.columns else None

    answers  = []
    contexts = []

    print(f"\nRunning HYBRID RAG chain on {len(questions)} questions...")
    for i, question in enumerate(questions):
        answer = rag_chain.invoke(
            {"question": question},
            config={"run_name": "ragas_eval", "tags": ["eval", "hybrid"]},
        )
        docs    = retrieve_contexts("admin", question, persist_directory=CHROMA_PATH)
        answers.append(answer)
        contexts.append([doc.page_content for doc in docs])

        print(f"  [{i+1}/{len(questions)}] Done: {question[:60]}...")

    data_dict = {
        "question": questions,
        "answer":   answers,
        "contexts": contexts,
    }
    if ground_truths:
        data_dict["ground_truth"] = ground_truths

    return data_dict


def collect_baseline_contexts(test_df: pd.DataFrame, top_k: int = 3) -> list:
    """Dense-only top-k contexts - the pre-hybrid baseline pipeline."""
    from app.embeddings import get_embeddings
    from langchain_chroma import Chroma

    vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embeddings())
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    baseline = []
    for _, row in test_df.iterrows():
        docs = retriever.invoke(row["question"])
        baseline.append([doc.page_content for doc in docs])
    return baseline


def _avg(df: pd.DataFrame, col: str):
    return round(df[col].mean(), 3) if col in df.columns else None


def evaluate_rag_chain(test_df: pd.DataFrame, sample_size: int = None):
    """Evaluates the upgraded (hybrid) pipeline and the dense-only baseline with Ragas 0.4."""

    if sample_size:
        test_df = test_df.head(sample_size)

    ragas_llm        = get_ragas_llm()
    ragas_embeddings = get_ragas_embeddings()
    run_config = RunConfig(max_retries=10, max_wait=60, timeout=240)

    # ── Upgraded pipeline: hybrid retrieval + cross-encoder rerank ──
    data_dict = collect_rag_results(test_df)
    eval_dataset = Dataset.from_dict({
        "user_input":         data_dict["question"],
        "response":           data_dict["answer"],
        "retrieved_contexts": data_dict["contexts"],
        "reference":          data_dict.get("ground_truth", data_dict["question"]),
    })

    upgraded_metrics = [
        Faithfulness(llm=ragas_llm),
        # strictness=1: Groq rejects n>1 completions, which ragas' default uses
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings, strictness=1),
        ContextRecall(llm=ragas_llm),
        ContextPrecision(llm=ragas_llm),
    ]

    print("\nEvaluating UPGRADED pipeline (hybrid + rerank) with Ragas...")
    upgraded_scores = evaluate(
        dataset=eval_dataset,
        metrics=upgraded_metrics,
        run_config=run_config,
    )
    upgraded_df = upgraded_scores.to_pandas()
    upgraded_df.to_csv("./resources/test_data/ragas_results.csv", index=False)

    print("\n─── UPGRADED (hybrid + rerank) ────────────────────────────────")
    display_cols = ["question", "faithfulness", "answer_relevancy", "context_recall", "context_precision"]
    # ragas 0.4 uses snake_case metric names on the resulting frame
    for col in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        value = _avg(upgraded_df, col)
        if value is not None:
            print(f"  {col:<22}: {value}")
    print("  Full results saved to ./resources/test_data/ragas_results.csv")

    # ── Baseline pipeline: dense-only top-3 (no BM25/RRF, no rerank) ──
    baseline_contexts = collect_baseline_contexts(test_df)
    baseline_dataset = Dataset.from_dict({
        "user_input":         data_dict["question"],
        "retrieved_contexts": baseline_contexts,
        "reference":          data_dict.get("ground_truth", data_dict["question"]),
    })

    baseline_metrics = [
        ContextPrecision(llm=ragas_llm),
        ContextRecall(llm=ragas_llm),
    ]

    print("\nEvaluating BASELINE pipeline (dense-only top-3) with Ragas...")
    baseline_scores = evaluate(
        dataset=baseline_dataset,
        metrics=baseline_metrics,
        run_config=run_config,
    )
    baseline_df = baseline_scores.to_pandas()
    baseline_df.to_csv("./resources/test_data/ragas_baseline.csv", index=False)

    print("\n─── BASELINE (dense-only top-3) ───────────────────────────────")
    for col in ["context_precision", "context_recall"]:
        value = _avg(baseline_df, col)
        if value is not None:
            print(f"  {col:<22}: {value}")
    print("  Full results saved to ./resources/test_data/ragas_baseline.csv")

    print("\n─── SUMMARY (baseline -> upgraded) ────────────────────────────")
    for col in ["context_precision", "context_recall"]:
        b, u = _avg(baseline_df, col), _avg(upgraded_df, col)
        if b is not None and u is not None:
            delta = "" if not (b and u) else f"  ({(u - b) * 100:+.1f} pts)"
            print(f"  {col:<22}: {b} -> {u}{delta}")



if __name__ == "__main__":
    print("Starting RAG evaluation...\n")

    if os.path.exists(TESTSET_PATH):
        print(f"Existing testset found at {TESTSET_PATH}. Loading...")
        test_df = pd.read_csv(TESTSET_PATH)
        print(f"Loaded {len(test_df)} test samples.")
    else:
        print("No testset found. Generating new one (this may take a few minutes)...")
        test_df = generate_test_data(DATA_PATH)

    sample_size = int(os.getenv("RAGAS_SAMPLE_SIZE", "0")) or None
    if sample_size:
        print(f"Sampling {sample_size} questions (RAGAS_SAMPLE_SIZE).")
    evaluate_rag_chain(test_df, sample_size=sample_size)