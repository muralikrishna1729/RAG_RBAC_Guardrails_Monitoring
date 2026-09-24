from app.pipeline.rag_chain import get_retrieved_sources
from app.pipeline.rag_chain import stream_rag_question
import streamlit as st 
from app.auth.users import get_user_role
from app.guardrails.guardrail import check_input_guardrail, check_output_guardrail
from app.utils.audit_logger import log_audit_event
from app.pipeline.rag_chain import build_rag_chain
import os

st.set_page_config(page_title="Company AI Assistant", page_icon="🤖", layout="centered")

@st.cache_resource
def get_cached_rag_chain(role):
    """
        This prevents reloading the embedding model on every click
    """
    return build_rag_chain("./chroma_db", role)


if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_info" not in st.session_state:
    st.session_state.user_info = {"username": "", "role": ""}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def show_login_page():
    st.title("🔐 Company AI Assistant")
    with st.container(border = True):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login", use_container_width=True):
            user_role = get_user_role(username)
            if user_role: #and verify_user(user, password): 
                st.session_state.logged_in = True
                st.session_state.user_info = {"username": username, "role": user_role}
                st.rerun()
            else:
                st.error("Invalid credentials. Please check your username/password.")


def show_chat_page():
    role = st.session_state.user_info['role']
    username = st.session_state.user_info['username']
    with st.sidebar:
        st.title("👤 Security & Role Context")
        st.markdown(f"**User:** `{username}`")
        st.markdown(f"**Role:** **{role.upper()}**")
        st.success(f"🔒 Vector Store Filter Active: `{role}` & `general` chunks")
        st.info("⚡ Semantic Cache & Cross-Encoder Active")
        st.divider()
        if st.button("Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.chat_history = []
            st.rerun()

    st.title("💬 Company RAG Assistant")
    st.caption(f"Context-aware assistant for {st.session_state.user_info['role']} department")

    # 1. Display Chat History
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("📄 Sources used"):
                    for src in message["sources"]:
                        st.write(f"• {src}")

    # 2. Handle New Input
    if prompt := st.chat_input("Ask a question about company policy..."):
        # Save and display user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # --- PRE-PROCESSING: Greetings & Simple Responses ---
            GREETING_RESPONSES = {
                "hi": "Hello! I'm your Company AI. How can I help you today?",
                "hello": "Hi there! Ready to help with your company queries.",
                "who are you": "I am the official RAG Assistant for our company.",
            }
            
            clean_prompt = prompt.lower().strip().rstrip("?")
            
            if clean_prompt in GREETING_RESPONSES:
                response_content = GREETING_RESPONSES[clean_prompt]
                sources = []
            violation_msg = check_input_guardrail(prompt)
            if violation_msg:
                response_text = f"⚠️ {violation_msg}"
                st.warning(response_text)
                sources = []
                log_audit_event(username, role, "CHAT", prompt, f"BLOCKED: {violation_msg}", [])
            else: 
                response_text = st.write_stream(stream_rag_question(role, prompt))
                response_text = check_output_guardrail(response_text)
                sources = get_retrieved_sources(role, prompt)
                log_audit_event(username, role, "CHAT", prompt, "PASSED", sources)
                

            if sources:
                with st.expander("📄 Source Documents & Re-Ranked Metadata"):
                    for src in sources:
                        st.write(f"• {src}")
            
            st.session_state.chat_history.append({"role":"assistant", "content": response_text, "sources": sources})
            
if not st.session_state.logged_in:
    show_login_page()
else:
    show_chat_page()