import re 
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()
import os


PII_PATTERNS = [
    r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',  # email
    r'\b\d{10}\b',                                      # phone
    r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'                  # aadhaar
]
PROMPT_INJECTION_PATTERNS = [
    r"ignore (all )?previous instructions",
    r"system prompt override",
    r"you are now (in )?dan mode",
    r"disregard (the )?above",
    r"reveal (your )?system prompt",
    r"jailbreak",
]
GREETINGS = ["hi", "hello", "hey", "good morning", "good afternoon", "hola"]
COMPANY_DOMAIN = os.getenv("COMPANY_DOMAIN", "company.com").lower()

_llm_instance = None
def get_llm():
    global _llm_instance
    if _llm_instance is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return None 
        _llm_instance = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
    return _llm_instance

def detect_pii(text:str)->bool:
    for pattern in PII_PATTERNS:
        if re.search(pattern, text):
            return True 
    return False


def detect_prompt_injection(text:str)->bool:
    """Scans for adversarial jailbreaks and system override attempts"""
    lowered = text.lower()
    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            return True
    return False


def check_scope(question: str) -> bool:
    llm = get_llm()
    if not llm:
        return True
    prompt = ChatPromptTemplate.from_template(
        """
        You are a security classifier. Determine if the question is related to HR, Finance, Marketing, Engineering, IT support, policies, or general Company Operations.
        Question: {question}
        Answer only with the word 'YES' or 'NO'.
        """
    )
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"question":question})
    return "YES" in result.strip().upper()


def check_input_guardrail(question: str)->str:
    if question.lower().strip() in GREETINGS:
        return None
    if detect_prompt_injection(question):
        return "Security Violation: Potential prompt injection or adversarial override detected."
    if detect_pii(question):
        return "Security Violation: Query contains sensitive personal information. Please remove it."
    if not check_scope(question):
        return "Out-of-Scope: I can only answer questions related to company operations."
    return None


def check_output_guardrail(response: str) -> str:
    for pattern in PII_PATTERNS:
        response = re.sub(pattern, "[REDACTED]", response)
    email_pattern = PII_PATTERNS[0]
    found_emails = re.findall(email_pattern, response)
    for email in found_emails:
        if not email.lower().endswith(f"@{COMPANY_DOMAIN}"):
            response = response.replace(email, "[REDACTED]")
    return response

if __name__ == "__main__":
    print(check_input_guardrail("What is john@company.com salary?"))
    print(check_input_guardrail("Who won IPL 2024?"))
    print(check_input_guardrail("What is the leave policy?"))
    print(check_output_guardrail("Contact HR at hr@company.com or call 9876543210"))





