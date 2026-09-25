"""Streamlit presentation layer for the Company RAG Assistant.

Everything visual lives here - stylesheet, avatars and reusable UI building
blocks - so `app.py` can stay a thin orchestration file and the backend
(retrieval, guardrails, auth, audit) is never touched by styling changes.

    from app.ui import inject_custom_css, render_sidebar, render_chat_history
"""

from app.ui.components import (
    ADMIN_NAV_PAGES,
    DEFAULT_PIPELINE_LABEL,
    PAGE_CHAT,
    PAGE_DASHBOARD,
    render_chat_header,
    render_chat_history,
    render_empty_state,
    render_pipeline_pill,
    render_sidebar,
    render_sources,
)
from app.ui.theme import AVATARS, inject_custom_css, load_custom_css, role_pill_html

__all__ = [
    "ADMIN_NAV_PAGES",
    "AVATARS",
    "DEFAULT_PIPELINE_LABEL",
    "PAGE_CHAT",
    "PAGE_DASHBOARD",
    "inject_custom_css",
    "load_custom_css",
    "render_chat_header",
    "render_chat_history",
    "render_empty_state",
    "render_pipeline_pill",
    "render_sidebar",
    "render_sources",
    "role_pill_html",
]
