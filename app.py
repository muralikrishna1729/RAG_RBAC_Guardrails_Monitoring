from app.pipeline.rag_chain import generate_response_with_sources, build_rag_chain
import streamlit as st
from app.auth.users import get_user_role, verify_credentials
from app.guardrails.guardrail import check_input_guardrail
from app.utils.audit_logger import log_audit_event
import pandas as pd

from app.utils.audit_reader import read_audit_events, summarize_events
from app.ui import (
    ADMIN_NAV_PAGES,
    AVATARS,
    PAGE_DASHBOARD,
    inject_custom_css,
    render_chat_header,
    render_chat_history,
    render_empty_state,
    render_pipeline_pill,
    render_sidebar,
    render_sources,
)

st.set_page_config(page_title="Company AI Assistant", page_icon="🤖", layout="centered")
inject_custom_css()


@st.cache_resource
def get_cached_rag_chain(role):
    """Cache the role-specific RAG chain and its loaded embedding resources."""
    return build_rag_chain("./chroma_db", role)


if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_info" not in st.session_state:
    st.session_state.user_info = {"username": "", "role": ""}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def show_login_page():
    st.title("🔐 Company AI Assistant")
    st.caption("Sign in with your company credentials to reach your role's knowledge base.")
    with st.container(border=True, key="login_card"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login", use_container_width=True, type="primary"):
            if verify_credentials(username, password):
                user_role = get_user_role(username)
                st.session_state.logged_in = True
                st.session_state.user_info = {"username": username, "role": user_role}
                st.rerun()
            else:
                st.error("Invalid credentials. Please check your username/password.")


def show_chat_page():
    role = st.session_state.user_info["role"]
    username = st.session_state.user_info["username"]

    render_chat_header(role)
    render_chat_history(st.session_state.chat_history)
    if not st.session_state.chat_history:
        render_empty_state(role)

    if prompt := st.chat_input("Ask a question about company policy..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=AVATARS["user"]):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar=AVATARS["assistant"]):
            thinking = render_pipeline_pill()
            greeting_responses = {
                "hi": "Hello! I'm your Company AI. How can I help you today?",
                "hello": "Hi there! Ready to help with your company queries.",
                "who are you": "I am the official RAG Assistant for our company.",
            }
            clean_prompt = prompt.lower().strip().rstrip("?")

            if clean_prompt in greeting_responses:
                response_text = greeting_responses[clean_prompt]
                sources = []
                log_audit_event(username, role, "CHAT", prompt, "PASSED", sources)
            else:
                violation_msg = check_input_guardrail(prompt)
                if violation_msg:
                    response_text = f"⚠️ {violation_msg}"
                    st.warning(response_text)
                    sources = []
                    log_audit_event(username, role, "CHAT", prompt, f"BLOCKED: {violation_msg}", [])
                else:
                    response_text, sources = generate_response_with_sources(
                        role,
                        prompt,
                        username=username,
                        input_guardrail_checked=True,
                    )
                    st.write_stream(iter([response_text]))
                    log_audit_event(username, role, "CHAT", prompt, "PASSED", sources)

            thinking.empty()
            if sources:
                render_sources(sources, label="📄 Source Documents & Re-Ranked Metadata")

            st.session_state.chat_history.append(
                {"role": "assistant", "content": response_text, "sources": sources}
            )


def show_audit_dashboard():
    """Admin-only security dashboard built from the rotated JSON audit log."""
    st.title("📊 Security & Audit Dashboard")
    st.caption("Admin-only view over the security audit log (RBAC-gated).")

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
                [{"reason": key, "count": value} for key, value in sorted(
                    summary["blocked_reasons"].items(), key=lambda item: -item[1]
                )]
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
    cols = [
        column for column in [
            "timestamp", "username", "role", "role_claim", "action", "guardrail_status", "query_length"
        ] if column in df.columns
    ]
    st.dataframe(df[cols].head(20), use_container_width=True, hide_index=True)


if not st.session_state.logged_in:
    show_login_page()
else:
    # Streamlit sessions can outlive role changes. Refresh authorization from the DB
    # on every rerun so a demoted/deleted user does not keep stale UI permissions.
    current_username = st.session_state.user_info.get("username", "")
    current_role = get_user_role(current_username)
    if not current_role:
        st.session_state.logged_in = False
        st.session_state.user_info = {"username": "", "role": ""}
        st.session_state.chat_history = []
        st.warning("Your account is no longer available. Please sign in again.")
        st.rerun()
    elif current_role != st.session_state.user_info.get("role"):
        st.session_state.user_info["role"] = current_role
        st.session_state.chat_history = []

    user_role = st.session_state.user_info.get("role", "")
    is_admin = user_role == "admin"
    selected_page = render_sidebar(
        st.session_state.user_info.get("username", ""),
        user_role,
        nav_pages=ADMIN_NAV_PAGES if is_admin else None,
    )

    if is_admin and selected_page == PAGE_DASHBOARD:
        show_audit_dashboard()
    else:
        show_chat_page()
