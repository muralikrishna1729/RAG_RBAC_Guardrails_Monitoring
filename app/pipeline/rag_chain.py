from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os
import hashlib
from app.embeddings import get_embeddings

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

from app.auth.users import get_user_role
from app.guardrails.guardrail import check_input_guardrail, check_output_guardrail
from app.retrieval.hybrid_rerank import rerank_documents

load_dotenv()

SEMANTIC_CACHE:dict = {}

def get_cache_key(role:str, question:str)->str:
    cleaned = question.lower().strip()
    return hashlib.md5(f"{role}:{cleaned}".encode()).hexdigest()


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
        docs =retriever.invoke(query)
        rerank_docs = rerank_documents(query, docs, top_k=3)
        return format_docs(rerank_docs)

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
    cache_key = get_cache_key(role, question)
    if cache_key in SEMANTIC_CACHE:
        print(f" CACHE HIT for key: {cache_key}")
        return SEMANTIC_CACHE[cache_key]
    
    chain,_ = build_rag_chain("./chroma_db",role = role)
    raw_response = chain.invoke({"question": question})
    safe_response = check_output_guardrail(raw_response)
    # Store in Semantic Cache
    SEMANTIC_CACHE[cache_key] = safe_response
    return safe_response

def stream_rag_question(role: str, question: str, persist_directory: str = "./chroma_db"):
    violation = check_input_guardrail(question)
    if violation:
        yield f" {violation}"
        return

    # Cache Lookup
    cache_key = get_cache_key(role, question)
    if cache_key in SEMANTIC_CACHE:
        yield SEMANTIC_CACHE[cache_key]
        return

    chain, _ = build_rag_chain(persist_directory, role=role)
    accumulated = ""
    for chunk in chain.stream({"question": question}):
        accumulated += chunk
        yield chunk



    SEMANTIC_CACHE[cache_key] = accumulated

if __name__== "__main__":
    response = ask_question("Madhu", "How many candidates are onboarded in this year to the company?")
    if not response:
        print("I don't have access to that information or no matching documents were found.")
    print(response)






