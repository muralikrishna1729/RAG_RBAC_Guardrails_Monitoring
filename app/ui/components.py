"""Reusable Streamlit UI building blocks for the Company RAG Assistant.

Pure presentation: these helpers only read arguments / session state and write
widgets. No retrieval, guardrail, auth or audit logic lives here, so the backend
pipeline under ``app/`` stays untouched by UI work.
"""

import html

import streamlit as st

from app.ui.theme import AVATARS, role_pill_html

# Sidebar navigation labels (admin-only page switcher).
PAGE_CHAT = "💬 Chat"
PAGE_DASHBOARD = "📊 Security Dashboard"
ADMIN_NAV_PAGES = [PAGE_CHAT, PAGE_DASHBOARD]

DEFAULT_PIPELINE_LABEL = "🔎 Retrieving role-scoped context (hybrid search + re-rank)…"

# Role-aware prompt ideas for the pre-first-message hint card.
_EMPTY_STATE_EXAMPLES = {
    "finance": [
        "Summarise this quarter's financial performance",
        "What is the revenue breakdown by segment?",
    ],
    "hr": [
        "How many days of annual leave do employees get?",
        "Walk me through the onboarding process",
    ],
    "marketing": [
        "What were the marketing KPIs for the last quarter?",
        "Summarise the latest campaign performance",
    ],
    "engineering": [
        "What is the deployment and on-call process?",
        "Summarise the engineering architecture guidelines",
    ],
    "admin": [
        "Compare policy changes across departments",
        "What do the department documents say about budgets?",
    ],
    "general": [
        "What does the employee handbook say about remote work?",
        "How does the careers process work?",
    ],
}


def render_sidebar(username: str, role: str, nav_pages=None):
    """Renders the session / security sidebar.

    Layout: brand, signed-in identity with a role pill, (optional) admin page
    switcher, system-status metric cards, then logout.

    Returns the selected navigation page when ``nav_pages`` is provided (admin
    users), otherwise ``None``.
    """
    selected_page = None

    with st.sidebar:
        st.markdown("### 🤖 Company AI Assistant")
        st.markdown(f"**Signed in as** `{html.escape(username or 'guest')}`")
        st.markdown(role_pill_html(role), unsafe_allow_html=True)

        if nav_pages:
            selected_page = st.radio(
                "Navigate",
                nav_pages,
                label_visibility="collapsed",
                key="nav_page",
            )

        st.divider()
        st.markdown("**System status**")
        left, right = st.columns(2)
        left.metric(
            "Access",
            (role or "general").title(),
            help="Vector store is filtered to your role plus general chunks.",
        )
        right.metric(
            "Cache",
            "Active ⚡",
            help="Semantic cache and cross-encoder re-ranking are enabled.",
        )
        st.caption("🔒 Vector store filtered by role · Cross-encoder re-ranking on")
        st.divider()

        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.chat_history = []
            st.rerun()

    return selected_page


def render_chat_header(role: str) -> None:
    """Gradient page title plus a role-aware tagline."""
    st.markdown("# 💬 Company RAG Assistant")
    st.markdown(
        "<p class='app-tagline'>Context-aware assistant for the "
        f"<b>{html.escape((role or 'general').title())}</b> department · "
        "secured by role-based retrieval</p>",
        unsafe_allow_html=True,
    )


def render_empty_state(role: str) -> None:
    """Hint card shown until the first question is asked."""
    examples = _EMPTY_STATE_EXAMPLES.get(
        (role or "general").lower(), _EMPTY_STATE_EXAMPLES["general"]
    )
    bullets = "".join(f"<li>{html.escape(example)}</li>" for example in examples)
    st.markdown(
        '<div class="empty-state">'
        "<div class='empty-state-title'>👋 Ask me anything about company policy, HR, "
        "or your department's docs.</div>"
        "<div class='empty-state-sub'>I'll cite sources automatically, and retrieval is "
        "filtered to what your role is allowed to see.</div>"
        f"<ul class='empty-state-examples'>{bullets}</ul>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_sources(sources, label: str = "📄 Sources used") -> None:
    """Collapsible list of the documents an answer was grounded in."""
    if not sources:
        return
    with st.expander(label):
        for source in sources:
            st.write(f"• {source}")


def render_chat_history(messages) -> None:
    """Replays previous turns with the same bubbles/avatars as live answers."""
    for message in messages:
        message_role = message.get("role", "assistant")
        with st.chat_message(message_role, avatar=AVATARS.get(message_role, "💬")):
            st.markdown(message.get("content", ""))
            render_sources(message.get("sources"))


def render_pipeline_pill(label: str = DEFAULT_PIPELINE_LABEL):
    """Renders an animated "working" pill inside the assistant bubble.

    Returns the placeholder so the caller can drop it once the answer is ready::

        pill = render_pipeline_pill()
        ...
        pill.empty()
    """
    placeholder = st.empty()
    placeholder.markdown(
        f'<div class="pipeline-pill"><span class="pipeline-dot"></span>'
        f"{html.escape(label, quote=False)}</div>",
        unsafe_allow_html=True,
    )
    return placeholder
