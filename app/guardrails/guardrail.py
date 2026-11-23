import logging
import os
import re

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

APP_ENV = os.getenv("APP_ENV", "development").strip().lower()
COMPANY_DOMAIN = os.getenv("COMPANY_DOMAIN", "company.com").lower()
if APP_ENV == "production" and not os.getenv("GROQ_API_KEY", "").strip():
    raise RuntimeError("GROQ_API_KEY must be configured in production.")

PII_PATTERNS = [
    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    r"\b\d{10}\b",
    r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
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

logger = logging.getLogger(__name__)
_llm_instance = None


def get_llm():
    """Create the scope-classifier client lazily with bounded timeout/retries."""
    global _llm_instance
    if _llm_instance is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return None
        _llm_instance = ChatGroq(
            model="openai/gpt-oss-20b",
            temperature=0,
            timeout=15,
            max_retries=1,
        )
    return _llm_instance


def detect_pii(text: str) -> bool:
    return any(re.search(pattern, text) for pattern in PII_PATTERNS)


def detect_prompt_injection(text: str) -> bool:
    """Scan for common adversarial jailbreak and system-override phrases."""
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in PROMPT_INJECTION_PATTERNS)


def check_scope(question: str) -> bool:
    llm = get_llm()
    if not llm:
        # Local development can run without the remote classifier; production cannot.
        return APP_ENV != "production"

    prompt = ChatPromptTemplate.from_template(
        """
        You are a security classifier. Determine if the question is related to HR,
        Finance, Marketing, Engineering, IT support, policies, or general Company
        Operations.
        Question: {question}
        Answer only with the word YES or NO.
        """
    )
    chain = prompt | llm | StrOutputParser()
    try:
        result = chain.invoke({"question": question})
        return result.strip().upper().strip(" .\n\t") == "YES"
    except Exception:
        # Fail closed if the scope classifier is unavailable.
        logger.exception("Scope classifier request failed; blocking this query.")
        return False


def check_input_guardrail(question: str) -> str:
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
    """Redact PII while preserving trusted company-domain email addresses."""
    email_pattern = PII_PATTERNS[0]
    trusted_domain = f"@{COMPANY_DOMAIN}"

    def redact_email(match) -> str:
        email = match.group(0)
        return email if email.lower().endswith(trusted_domain) else "[REDACTED]"

    response = re.sub(email_pattern, redact_email, response)
    for pattern in PII_PATTERNS[1:]:
        response = re.sub(pattern, "[REDACTED]", response)
    return response


if __name__ == "__main__":
    print(check_input_guardrail("What is john@company.com salary?"))
    print(check_input_guardrail("Who won IPL 2024?"))
    print(check_input_guardrail("What is the leave policy?"))
    print(check_output_guardrail("Contact HR at hr@company.com or call 9876543210"))
