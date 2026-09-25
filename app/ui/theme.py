"""Theme plumbing for the Streamlit UI.

The stylesheet is a real artifact (``app/ui/assets/custom.css``) rather than an
inline string, so it can be diffed, linted and previewed on its own.

Usage in ``app.py``::

    st.set_page_config(page_title="Company AI Assistant", page_icon="🤖")
    inject_custom_css()
"""

import html
from pathlib import Path

import streamlit as st

_CSS_PATH = Path(__file__).resolve().parent / "assets" / "custom.css"

# Chat avatars. Emoji resolve through Streamlit's AvatarType.EMOJI branch, so no
# image assets are needed for the coloured user/assistant bubbles.
AVATARS = {"user": "🧑‍💼", "assistant": "🤖"}


def load_custom_css() -> str:
    """Returns the packaged stylesheet text.

    Deliberately not cached: the file is a few KB, and re-reading it on each
    rerun means CSS tweaks appear on the next interaction (no server restart).
    """
    return _CSS_PATH.read_text(encoding="utf-8")


def inject_custom_css() -> None:
    """Injects the packaged stylesheet into the app.

    Must be called *after* ``st.set_page_config``.
    """
    st.markdown(f"<style>{load_custom_css()}</style>", unsafe_allow_html=True)


def role_pill_html(role: str) -> str:
    """Returns the coloured badge markup for the signed-in role.

    The ``role-<slug>`` class is styled in ``custom.css`` so each department gets
    its own accent colour.
    """
    safe_role = (role or "general").strip()
    slug = "".join(ch for ch in safe_role.lower() if ch.isalnum() or ch in "-_") or "general"
    return f'<span class="role-pill role-{slug}">{html.escape(safe_role.upper())}</span>'
