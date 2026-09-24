from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os
from app.embeddings import get_embeddings

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

from app.auth.users import get_user_role
from app.guardrails.guardrail import check_input_guardrail, check_output_guardrail
from app.retrieval.hybrid_rerank import bm25_search, reciprocal_rank_fusion, rerank_documents

load_dotenv()

# Semantic cache: stores post-guardrail answers; a hit requires cosine similarity
# >= threshold against a cached question FOR THE SAME ROLE (answers never leak roles).
SEMANTIC_CACHE: list = []
SEMANTIC_CACHE_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.92"))
SEMANTIC_CACHE_MAX_SIZE = 128


def _cosine_similarity(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def _cache_lookup(role: str, question: str):
    """Returns a cached answer when a semantically similar question was asked for the same role."""
    query_vector = get_embeddings().embed_query(question)
    best_answer, best_score = None, 0.0
    for entry in SEMANTIC_CACHE:
        if entry["role"] != role:
            continue
        score = _cosine_similarity(query_vector, entry["vector"])
        if score > best_score:
            best_answer, best_score = entry["answer"], score
    if best_answer is not None and best_score >= SEMANTIC_CACHE_THRESHOLD:
        print(f"⚡ SEMANTIC CACHE HIT (similarity={best_score:.3f})")
        return best_answer
    return None


def _cache_store(role: str, question: str, answer: str) -> None:
    if len(SEMANTIC_CACHE) >= SEMANTIC_CACHE_MAX_SIZE:
        SEMANTIC_CACHE.pop(0)  # drop the oldest entry (bounded memory)
    SEMANTIC_CACHE.append({
        "role": role,
        "question": question,
        "answer": answer,
        "vector": get_embeddings().embed_query(question),
    })


def build_rag_chain(persist_directory: str, role:str):
    if not os.path.exists(persist_directory):
        raise FileNotFoundError(f"Vector database not found at {persist_directory}")
    vectorstore = Chroma(persist_directory= persist_directory ,embedding_function= get_embeddings())

    search_kwargs = {"k":6}
    if role != 'admin':
        search_kwargs["filter"] = {"$or" : [{"role":role},{"role":"general"}]}
    else:
        print("Admin access granted: Searching all departments.")

    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)   

    # llm = HuggingFaceEndpoint(
    #     repo_id="meta-llama/Llama-3.1-8B-Instruct",
    #     task="text-generation",
    #     temperature = 0.1 ,# low temp = more factual

    # )
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template(
        """
        You are a professional and friendly Company AI Assistant.

        Instructions:
            1. GREETINGS: If the user says 'Hi' or 'Hello', reply: "Hello! I'm your company assistant. How can I help you today?"
            2. IDENTITY: If asked 'Who are you?'or 'How do you doing' like that reply: "I am a RAG-powered assistant specialized in company policies.I'm Here to assist you"
            3. EXCEPTIONS: 
               - HR Email: hr@company.com
               - Leave Portal: https://portal.company.com/leave
            4. RAG: For all other questions, use the context below. 
            5. FALLBACK: If info is not in context or exceptions, say "I don't have that information."

        STRICT RULES:
            1. Use the context below to answer questions.
            2. If the answer is NOT in the context AND not in the 'EXCEPTIONS' above, 
               say: "I'm sorry, I don't have that information in our records."
            3. Do not make up facts.

        Context: {context}
        Question: {question}
        Answer:
        """
    )

    def retrieve_and_rerank(query:str):
        dense_docs = retriever.invoke(query)
        sparse_docs = bm25_search(query, role, top_k=6, vectorstore=vectorstore)
        fused = reciprocal_rank_fusion(dense_docs, sparse_docs)
        candidates = [doc for doc, _score in fused[:8]]
        return format_docs(rerank_documents(query, candidates, top_k=3))

    # chain = ({"context": retriever | format_docs ,"question":RunnablePassthrough()}| prompt | ChatHuggingFace(llm=llm) | StrOutputParser())
    chain = (
        {"context": lambda x: retrieve_and_rerank(x["question"]), "question": lambda x: x["question"]}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever

def format_docs(docs):
    if not docs:
        return "No relevant matching documents were found."
    return "\n\n".join(doc.page_content for doc in docs)

def get_retrieved_sources(role:str, question:str, persist_directory:str = "./chroma_db"):
    try:
        chain,retriever = build_rag_chain(persist_directory, role)
        docs = retriever.invoke(question)
        docs = rerank_documents(question, docs, top_k =3)
        sources = []
        for doc in docs:
            dept = doc.metadata.get("role", "general")
            source_file = doc.metadata.get("source", "company doc")
            snippet = doc.page_content[:150].replace("\n"," ")+"..."
            sources.append(f"[{dept.upper()}] {os.path.basename(source_file)}:\"{snippet}\"")
        return sources
    except Exception as e:
        return [f"Source info unavailable: {str(e)}"]
    

def check_access(docs,role):
    if not docs:
        return "I don't have access to that information or no matching documents were found."
    else:
        return format_docs(docs)

def ask_question(username:str, question:str):
    role = get_user_role(username)
    if not role:
        return "Access Denied: User not recognized."
    violation = check_input_guardrail(question)
    if violation:
        return violation
    
    # Semantic Cache Lookup
    cached = _cache_lookup(role, question)
    if cached:
        return cached
    
    chain,_ = build_rag_chain("./chroma_db",role = role)
    raw_response = chain.invoke(
        {"question": question},
        config={
            "run_name": "rbac_rag_chat",
            "tags": [f"role:{role}"],
            "metadata": {"username": username, "role": role},
        },
    )
    safe_response = check_output_guardrail(raw_response)
    # Store in Semantic Cache
    _cache_store(role, question, safe_response)
    return safe_response

def stream_rag_question(role: str, question: str, persist_directory: str = "./chroma_db", username: str = None):
    violation = check_input_guardrail(question)
    if violation:
        yield f" {violation}"
        return

    # Cache Lookup
    cached = _cache_lookup(role, question)
    if cached:
        yield cached
        return

    chain, _ = build_rag_chain(persist_directory, role=role)
    accumulated = ""
    for chunk in chain.stream(
        {"question": question},
        config={
            "run_name": "rbac_rag_stream",
            "tags": [f"role:{role}"],
            "metadata": {"username": username, "role": role},
        },
    ):
        accumulated += chunk
        yield chunk



    _cache_store(role, question, accumulated)

if __name__== "__main__":
    response = ask_question("Madhu", "How many candidates are onboarded in this year to the company?")
    if not response:
        print("I don't have access to that information or no matching documents were found.")
    print(response)






