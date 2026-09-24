from app.pipeline.rag_chain import get_retrieved_sources
from app.pipeline.rag_chain import stream_rag_question
import streamlit as st 
from app.auth.users import get_user_role, verify_credentials
from app.guardrails.guardrail import check_input_guardrail, check_output_guardrail
from app.utils.audit_logger import log_audit_event
from app.pipeline.rag_chain import build_rag_chain
import os
import pandas as pd

from app.utils.audit_reader import read_audit_events, summarize_events

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
            if verify_credentials(username, password):
                user_role = get_user_role(username)
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
                response_text = st.write_stream(stream_rag_question(role, prompt, username=username))
                response_text = check_output_guardrail(response_text)
                sources = get_retrieved_sources(role, prompt)
                log_audit_event(username, role, "CHAT", prompt, "PASSED", sources)
                

            if sources:
                with st.expander("📄 Source Documents & Re-Ranked Metadata"):
                    for src in sources:
                        st.write(f"• {src}")
            
            st.session_state.chat_history.append({"role":"assistant", "content": response_text, "sources": sources})
            
def show_audit_dashboard():
    """Admin-only security dashboard built from security_audit.log."""
    st.title("📊 Security & Audit Dashboard")
    st.caption("Admin-only view over `security_audit.log` (RBAC-gated).")

    events = read_audit_events()
    if not events:
        st.info("No audit events recorded yet. Chat activity will appear here.")
        return

    summary = summarize_events(events)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total events", summary["total_events"])
    c2.metric("Passed", summary["passed"])
    c3.metric("Blocked", summary["blocked"])
    block_rate = (summary["blocked"] / summary["total_events"] * 100) if summary["total_events"] else 0.0
    c4.metric("Block rate", f"{block_rate:.1f}%")

    left, right = st.columns(2)
    with left:
        st.subheader("Queries by role")
        st.bar_chart(pd.Series(summary["by_role"]))
    with right:
        st.subheader("By action")
        st.bar_chart(pd.Series(summary["by_action"]))

    st.subheader("Blocked requests by reason")
    if summary["blocked_reasons"]:
        st.dataframe(
            pd.DataFrame(
                [{"reason": k, "count": v} for k, v in sorted(summary["blocked_reasons"].items(), key=lambda kv: -kv[1])]
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.success("No blocked requests recorded.")

    st.subheader("Recent events")
    df = pd.DataFrame(events)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp", ascending=False)
    cols = [c for c in ["timestamp", "username", "role_claim", "action", "guardrail_status", "query_length"] if c in df.columns]
    st.dataframe(df[cols].head(20), use_container_width=True, hide_index=True)


if not st.session_state.logged_in:
    show_login_page()
else:
    if st.session_state.user_info.get("role") == "admin":
        page = st.sidebar.radio("Go to", ["💬 Chat", "📊 Security Dashboard"], label_visibility="collapsed")
        if page == "📊 Security Dashboard":
            show_audit_dashboard()
        else:
            show_chat_page()
    else:
        show_chat_page()