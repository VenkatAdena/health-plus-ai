"""Streamlit chat portal — the human face of the HealthPlus RAG stack.

WHY this file looks the way it does:

Streamlit re-executes this entire script on every user interaction (a
"rerun"). Anything that must survive between reruns therefore lives in one
of two places:

- ``st.cache_resource`` for heavy, process-wide singletons. The ChatService
  keeps conversation memory *in the object itself*, and building it loads
  the embedding model and constructs the Claude client — so it must be
  built exactly once and reused, never rebuilt per rerun.
- ``st.session_state`` for per-browser-tab state: the visible chat log
  (display only — the service records its own memory for the LLM) and the
  conversation id that keys that service-side memory.

Every failure path the backend documents (missing API key, Claude outage,
bad query, bad PDF) is caught and shown as a friendly message. A user of
this app should never see a Python traceback.
"""

from __future__ import annotations

import logging
import uuid

import streamlit as st

from healthplus.application import KnowledgeBaseService, build_chat_service
from healthplus.config import get_settings
from healthplus.core.exceptions import ConfigurationError, HealthPlusError, LLMError
from healthplus.core.logging import configure_logging
from healthplus.knowledge_base.models import DocumentCategory

logger = logging.getLogger(__name__)

# Sentinel label for "no category filter" in the sidebar selectbox.
ALL_CATEGORIES = "All categories"

DISCLAIMER = "Informational assistant for hospital operations — not medical advice."

# set_page_config must be the first Streamlit call in the script.
st.set_page_config(
    page_title="HealthPlus AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS — injected once per page load
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    /* ── Google Font ─────────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Root variables ──────────────────────────────────────────────── */
    :root {
        --hp-blue:        #0EA5E9;
        --hp-blue-dark:   #0284C7;
        --hp-blue-light:  #E0F2FE;
        --hp-navy:        #0F172A;
        --hp-slate:       #475569;
        --hp-muted:       #94A3B8;
        --hp-surface:     #F8FAFC;
        --hp-card:        #FFFFFF;
        --hp-border:      #E2E8F0;
        --hp-success:     #10B981;
        --hp-warning:     #F59E0B;
        --hp-error:       #EF4444;
        --hp-radius:      12px;
        --hp-radius-sm:   8px;
        --hp-shadow:      0 1px 3px rgba(0,0,0,.08), 0 1px 2px rgba(0,0,0,.06);
        --hp-shadow-md:   0 4px 12px rgba(14,165,233,.12), 0 2px 4px rgba(0,0,0,.06);
        --hp-transition:  all .2s cubic-bezier(.4,0,.2,1);
    }

    /* ── Base reset ──────────────────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* ── App background ──────────────────────────────────────────────── */
    .stApp {
        background: var(--hp-surface) !important;
    }

    /* ── Main content wrapper ────────────────────────────────────────── */
    .main .block-container {
        padding: 0 2.5rem 5rem !important;
        max-width: 900px;
    }

    /* Remove Streamlit's built-in top gap */
    [data-testid="stAppViewBlockContainer"] {
        padding-top: 0 !important;
    }
    .stMainBlockContainer {
        padding-top: 0 !important;
    }

    /* ── Page hero header ────────────────────────────────────────────── */
    .hp-hero {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 1.5rem 2rem;
        background: linear-gradient(135deg, #0EA5E9 0%, #0284C7 50%, #075985 100%);
        border-radius: 0 0 var(--hp-radius) var(--hp-radius);
        margin-top: 0;
        margin-bottom: 1.5rem;
        box-shadow: var(--hp-shadow-md);
    }
    .hp-hero-icon {
        font-size: 2.5rem;
        filter: drop-shadow(0 2px 4px rgba(0,0,0,.2));
    }
    .hp-hero-text h1 {
        color: #fff !important;
        font-size: 1.75rem !important;
        font-weight: 700 !important;
        margin: 0 0 .2rem !important;
        line-height: 1.2 !important;
    }
    .hp-hero-text p {
        color: rgba(255,255,255,.8) !important;
        font-size: .85rem !important;
        margin: 0 !important;
    }

    /* ── Sidebar ─────────────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: var(--hp-card) !important;
        border-right: 1px solid var(--hp-border) !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding: 0.75rem 1.25rem 1.5rem !important;
    }

    /* Sidebar section label */
    .hp-sidebar-label {
        font-size: .7rem;
        font-weight: 600;
        letter-spacing: .08em;
        text-transform: uppercase;
        color: var(--hp-muted);
        margin: 1.25rem 0 .6rem;
    }

    /* Sidebar brand strip */
    .hp-sidebar-brand {
        display: flex;
        align-items: center;
        gap: .6rem;
        padding: .75rem 1rem;
        background: linear-gradient(135deg, #0EA5E9, #0284C7);
        border-radius: var(--hp-radius-sm);
        margin-bottom: 1.25rem;
    }
    .hp-sidebar-brand span {
        color: #fff;
        font-weight: 700;
        font-size: 1rem;
    }
    .hp-sidebar-brand small {
        color: rgba(255,255,255,.75);
        font-size: .7rem;
        display: block;
        line-height: 1.2;
    }

    /* KB metric card */
    .hp-metric-card {
        background: var(--hp-blue-light);
        border: 1px solid #BAE6FD;
        border-radius: var(--hp-radius-sm);
        padding: .75rem 1rem;
        margin-bottom: .75rem;
        text-align: center;
    }
    .hp-metric-card .hp-metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--hp-blue-dark);
        line-height: 1;
    }
    .hp-metric-card .hp-metric-label {
        font-size: .72rem;
        font-weight: 500;
        color: var(--hp-slate);
        margin-top: .2rem;
        text-transform: uppercase;
        letter-spacing: .06em;
    }

    /* Category pill list */
    .hp-cat-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: .35rem .5rem;
        border-radius: 6px;
        font-size: .8rem;
        margin-bottom: .25rem;
        background: var(--hp-surface);
        border: 1px solid var(--hp-border);
        transition: var(--hp-transition);
    }
    .hp-cat-row:hover { background: var(--hp-blue-light); border-color: #7DD3FC; }
    .hp-cat-name { color: var(--hp-navy); font-weight: 500; }
    .hp-cat-count {
        background: var(--hp-blue);
        color: #fff;
        font-size: .68rem;
        font-weight: 600;
        padding: .15rem .45rem;
        border-radius: 999px;
        min-width: 24px;
        text-align: center;
    }

    /* Sidebar divider */
    .hp-divider {
        height: 1px;
        background: var(--hp-border);
        margin: 1rem 0;
        border: none;
    }

    /* ── Buttons ─────────────────────────────────────────────────────── */
    .stButton > button {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: .85rem !important;
        border-radius: var(--hp-radius-sm) !important;
        padding: .55rem 1.1rem !important;
        transition: var(--hp-transition) !important;
        border: none !important;
    }
    /* Primary (default Streamlit blue maps to our theme) */
    .stButton > button[kind="primary"],
    .stButton > button:not([kind="secondary"]) {
        background: linear-gradient(135deg, var(--hp-blue), var(--hp-blue-dark)) !important;
        color: #fff !important;
        box-shadow: 0 2px 8px rgba(14,165,233,.35) !important;
    }
    .stButton > button:not([kind="secondary"]):hover {
        background: linear-gradient(135deg, var(--hp-blue-dark), #075985) !important;
        box-shadow: 0 4px 14px rgba(14,165,233,.45) !important;
        transform: translateY(-1px) !important;
    }
    /* Clear / secondary style button */
    .stButton > button[kind="secondary"] {
        background: var(--hp-surface) !important;
        color: var(--hp-slate) !important;
        border: 1px solid var(--hp-border) !important;
    }
    .stButton > button[kind="secondary"]:hover {
        background: #FEE2E2 !important;
        border-color: #FECACA !important;
        color: var(--hp-error) !important;
    }
    /* Disabled */
    .stButton > button:disabled {
        opacity: .45 !important;
        cursor: not-allowed !important;
        transform: none !important;
        box-shadow: none !important;
    }

    /* ── Select boxes & radio ────────────────────────────────────────── */
    .stSelectbox > div > div,
    .stRadio > div {
        border-radius: var(--hp-radius-sm) !important;
        border-color: var(--hp-border) !important;
        font-size: .85rem !important;
    }
    .stRadio > div > label {
        font-size: .82rem !important;
        font-weight: 500 !important;
    }

    /* ── File uploader ───────────────────────────────────────────────── */
    [data-testid="stFileUploader"] {
        border: 2px dashed var(--hp-border) !important;
        border-radius: var(--hp-radius) !important;
        background: var(--hp-surface) !important;
        transition: var(--hp-transition) !important;
        padding: .5rem !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: var(--hp-blue) !important;
        background: var(--hp-blue-light) !important;
    }

    /* ── Chat messages ───────────────────────────────────────────────── */
    [data-testid="stChatMessage"] {
        border-radius: var(--hp-radius) !important;
        padding: 1rem 1.25rem !important;
        margin-bottom: .75rem !important;
        box-shadow: var(--hp-shadow) !important;
        border: 1px solid var(--hp-border) !important;
        transition: var(--hp-transition) !important;
        animation: hp-msg-in .25s ease forwards;
    }
    [data-testid="stChatMessage"]:hover {
        box-shadow: var(--hp-shadow-md) !important;
    }
    /* User message */
    [data-testid="stChatMessage"][data-testid*="user"],
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: linear-gradient(135deg, #EFF6FF, #DBEAFE) !important;
        border-color: #BFDBFE !important;
    }
    /* Assistant message */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background: var(--hp-card) !important;
        border-color: var(--hp-border) !important;
    }

    @keyframes hp-msg-in {
        from { opacity: 0; transform: translateY(6px); }
        to   { opacity: 1; transform: translateY(0);   }
    }

    /* ── Chat input ──────────────────────────────────────────────────── */
    [data-testid="stChatInput"] {
        border-radius: var(--hp-radius) !important;
        border: 2px solid var(--hp-border) !important;
        background: var(--hp-card) !important;
        box-shadow: var(--hp-shadow-md) !important;
        transition: var(--hp-transition) !important;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: var(--hp-blue) !important;
        box-shadow: 0 0 0 4px rgba(14,165,233,.15), var(--hp-shadow-md) !important;
    }
    [data-testid="stChatInput"] textarea {
        font-size: .9rem !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* ── Source expander ─────────────────────────────────────────────── */
    [data-testid="stExpander"] {
        border: 1px solid var(--hp-border) !important;
        border-radius: var(--hp-radius-sm) !important;
        background: var(--hp-surface) !important;
        margin-top: .6rem !important;
    }
    [data-testid="stExpander"] summary {
        font-size: .8rem !important;
        font-weight: 600 !important;
        color: var(--hp-slate) !important;
        padding: .5rem .75rem !important;
    }

    /* Source card inside expander */
    .hp-source-card {
        background: var(--hp-card);
        border: 1px solid var(--hp-border);
        border-left: 3px solid var(--hp-blue);
        border-radius: var(--hp-radius-sm);
        padding: .65rem .9rem;
        margin-bottom: .5rem;
        transition: var(--hp-transition);
    }
    .hp-source-card:hover {
        border-left-color: var(--hp-blue-dark);
        box-shadow: var(--hp-shadow);
    }
    .hp-source-title {
        font-size: .82rem;
        font-weight: 600;
        color: var(--hp-navy);
        margin-bottom: .2rem;
    }
    .hp-source-meta {
        display: flex;
        gap: .5rem;
        flex-wrap: wrap;
        margin-bottom: .4rem;
    }
    .hp-badge {
        font-size: .68rem;
        font-weight: 600;
        padding: .1rem .45rem;
        border-radius: 999px;
        background: var(--hp-blue-light);
        color: var(--hp-blue-dark);
        border: 1px solid #7DD3FC;
    }
    .hp-badge-score {
        background: #F0FDF4;
        color: #16A34A;
        border-color: #86EFAC;
    }
    .hp-source-snippet {
        font-size: .78rem;
        color: var(--hp-slate);
        line-height: 1.5;
    }

    /* ── Welcome / empty state ───────────────────────────────────────── */
    .hp-welcome {
        text-align: center;
        padding: 3rem 2rem;
        color: var(--hp-muted);
    }
    .hp-welcome-icon { font-size: 3.5rem; margin-bottom: 1rem; }
    .hp-welcome h3 {
        font-size: 1.15rem !important;
        font-weight: 600 !important;
        color: var(--hp-slate) !important;
        margin-bottom: .5rem !important;
    }
    .hp-welcome p {
        font-size: .85rem;
        color: var(--hp-muted);
        max-width: 380px;
        margin: 0 auto;
    }

    /* Suggestion chips */
    .hp-chips {
        display: flex;
        flex-wrap: wrap;
        gap: .5rem;
        justify-content: center;
        margin-top: 1.5rem;
    }
    .hp-chip {
        display: inline-flex;
        align-items: center;
        gap: .35rem;
        padding: .45rem .85rem;
        background: var(--hp-card);
        border: 1px solid var(--hp-border);
        border-radius: 999px;
        font-size: .78rem;
        font-weight: 500;
        color: var(--hp-slate);
        cursor: default;
        transition: var(--hp-transition);
        white-space: nowrap;
    }
    .hp-chip:hover {
        background: var(--hp-blue-light);
        border-color: #7DD3FC;
        color: var(--hp-blue-dark);
    }

    /* ── Metrics / st.metric override ───────────────────────────────── */
    [data-testid="stMetric"] {
        background: var(--hp-blue-light) !important;
        border: 1px solid #BAE6FD !important;
        border-radius: var(--hp-radius-sm) !important;
        padding: .75rem 1rem !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        color: var(--hp-blue-dark) !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: .72rem !important;
        font-weight: 600 !important;
        color: var(--hp-slate) !important;
        text-transform: uppercase !important;
        letter-spacing: .06em !important;
    }

    /* ── Alerts ──────────────────────────────────────────────────────── */
    [data-testid="stAlert"] {
        border-radius: var(--hp-radius-sm) !important;
        border-left-width: 4px !important;
        font-size: .85rem !important;
    }

    /* ── Spinner ─────────────────────────────────────────────────────── */
    [data-testid="stSpinner"] > div {
        color: var(--hp-blue) !important;
    }

    /* ── Responsive tweaks ───────────────────────────────────────────── */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 0.5rem 1rem 4rem !important;
        }
        .hp-hero {
            padding: 1rem 1.25rem;
            gap: .75rem;
        }
        .hp-hero-text h1 { font-size: 1.35rem !important; }
        .hp-chips { gap: .4rem; }
        .hp-chip { font-size: .72rem; padding: .35rem .65rem; }
    }
    @media (max-width: 480px) {
        .hp-hero-text h1 { font-size: 1.15rem !important; }
        .hp-hero-icon { font-size: 2rem; }
    }

    /* ── Hide Streamlit chrome ────────────────────────────────────────── */
    /* Safe hides only — do NOT touch header or any data-testid that
       could affect sidebar toggle buttons (varies by Streamlit version) */
    #MainMenu { visibility: hidden !important; }
    footer    { visibility: hidden !important; }

    /* Make the header bar visually disappear but stay fully functional
       so ALL native sidebar collapse/expand controls keep working */
    header {
        background-color: transparent !important;
        box-shadow: none !important;
        border-bottom: none !important;
    }

    /* Hide the Deploy button */
    [data-testid="stDeployButton"]  { display: none !important; }
    [data-testid="stToolbarActions"] { display: none !important; }
    .stDeployButton                  { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Process-wide singletons (survive reruns via st.cache_resource)
# ---------------------------------------------------------------------------


@st.cache_resource
def _bootstrap() -> None:
    """One-time process setup: runtime directories and logging.

    Wrapped in cache_resource so reruns don't re-configure logging (the
    configure function is idempotent anyway, but there is no reason to pay
    for it on every interaction).
    """
    settings = get_settings()
    settings.ensure_directories()
    configure_logging(level=settings.log_level, log_dir=settings.log_dir)


@st.cache_resource(show_spinner="Opening the knowledge base…")
def _get_knowledge_base() -> KnowledgeBaseService:
    """The KB service used by the sidebar (stats + PDF uploads).

    Deliberately separate from the chat service: uploads and stats only
    need the vector store and embedder, NOT the Claude API key — so the
    sidebar keeps working even when the key is missing.
    """
    return KnowledgeBaseService()


@st.cache_resource(show_spinner="Starting the assistant…")
def _get_chat_service(provider: str = "claude"):
    """One ChatService per provider for this process.

    Cached separately per provider value so switching between Claude and
    OpenAI does not rebuild the service each time — just the first switch.
    Raises ConfigurationError if the required API key is missing; Streamlit
    does not cache exceptions, so fixing .env and rerunning recovers cleanly.
    """
    return build_chat_service(provider=provider)


_bootstrap()


# ---------------------------------------------------------------------------
# Per-tab state (survives reruns via st.session_state)
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    # Display log only. Each entry: {"role": ..., "content": ..., "sources": [...]}.
    # The ChatService keeps its own memory for the LLM — we never feed this back.
    st.session_state["messages"] = []

if "conversation_id" not in st.session_state:
    # Keys the service-side conversation memory. A fresh id = a fresh memory.
    st.session_state["conversation_id"] = uuid.uuid4().hex


def _reset_conversation() -> None:
    """Clear the visible log AND rotate the conversation id.

    Rotating the id matters: the service's memory is keyed by it, so a new
    id means the LLM also starts from a blank history — clearing only the
    display would leave the model still 'remembering' the old chat.
    """
    st.session_state["messages"] = []
    st.session_state["conversation_id"] = uuid.uuid4().hex


def _render_sources(sources: list[dict]) -> None:
    """Render retrieval citations as styled source cards inside an expander.

    Takes plain dicts (SearchResult.model_dump()) so the same function works
    for both the just-streamed answer and messages replayed from the log.
    """
    if not sources:
        return
    with st.expander(f"📄 Sources ({len(sources)})", expanded=False):
        for i, src in enumerate(sources, start=1):
            snippet = src["text"][:220].strip()
            if len(src["text"]) > 220:
                snippet += "…"
            cat_label = src["category"].replace("_", " ").title()
            st.markdown(
                f"""
                <div class="hp-source-card">
                    <div class="hp-source-title">[{i}] {src['source']}</div>
                    <div class="hp-source-meta">
                        <span class="hp-badge">📄 Page {src['page_number']}</span>
                        <span class="hp-badge">{cat_label}</span>
                        <span class="hp-badge hp-badge-score">Score {src['score']:.2f}</span>
                    </div>
                    <div class="hp-source-snippet">{snippet}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Sidebar: knowledge-base stats, category filter, conversation & upload tools
# ---------------------------------------------------------------------------

# The sidebar depends only on the KB service (no Claude key needed), so we
# build it BEFORE attempting the chat service — a missing API key must not
# take down stats or uploads.
knowledge_base: KnowledgeBaseService | None = None
try:
    knowledge_base = _get_knowledge_base()
except HealthPlusError as exc:
    st.sidebar.error(f"Knowledge base unavailable: {exc}")

with st.sidebar:
    # Brand strip
    st.markdown(
        """
        <div class="hp-sidebar-brand">
            <span style="font-size:1.4rem">🏥</span>
            <div>
                <span>HealthPlus AI</span>
                <small>Diagnostic Intelligence</small>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Knowledge base stats ──────────────────────────────────────────
    st.markdown('<div class="hp-sidebar-label">Knowledge Base</div>', unsafe_allow_html=True)

    if knowledge_base is not None:
        st.metric("Total chunks", knowledge_base.chunk_count)
        counts = knowledge_base.category_counts()
        if counts:
            for category_name, count in sorted(counts.items()):
                label = category_name.replace("_", " ").title()
                st.markdown(
                    f'<div class="hp-cat-row">'
                    f'<span class="hp-cat-name">{label}</span>'
                    f'<span class="hp-cat-count">{count}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No documents ingested yet — upload a PDF below.")
    else:
        st.caption("Knowledge base unavailable.")

    st.markdown('<hr class="hp-divider">', unsafe_allow_html=True)

    # ── Chat settings ─────────────────────────────────────────────────
    st.markdown('<div class="hp-sidebar-label">Chat Settings</div>', unsafe_allow_html=True)

    provider_choice = st.radio(
        "LLM provider",
        options=["Claude (Anthropic)", "ChatGPT (OpenAI)"],
        horizontal=True,
        help="Switch the language model powering the assistant.",
        label_visibility="collapsed",
    )
    selected_provider = "openai" if "OpenAI" in provider_choice else "claude"

    # The filter narrows retrieval, not the LLM: answer(category=...) only
    # searches chunks tagged with the chosen category.
    filter_choice = st.selectbox(
        "Search within category",
        options=[ALL_CATEGORIES] + [c.value for c in DocumentCategory],
        help="Limit retrieval to one document category, or search everything.",
    )
    selected_category: str | None = (
        None if filter_choice == ALL_CATEGORIES else filter_choice
    )

    if st.button("🗑️ Clear conversation", use_container_width=True):
        _reset_conversation()
        st.rerun()

    st.markdown('<hr class="hp-divider">', unsafe_allow_html=True)

    # ── Document upload ───────────────────────────────────────────────
    st.markdown('<div class="hp-sidebar-label">Add a Document</div>', unsafe_allow_html=True)

    # A successful ingest ends in st.rerun() (so the stats above refresh),
    # which would wipe an inline st.success — so the message is parked in
    # session_state and shown on the run AFTER the rerun.
    if "last_ingest_message" in st.session_state:
        st.success(st.session_state.pop("last_ingest_message"))

    if knowledge_base is None:
        st.caption("Uploads are disabled while the knowledge base is unavailable.")
    else:
        uploaded_file = st.file_uploader("PDF document", type=["pdf"], label_visibility="collapsed")
        upload_category = st.selectbox(
            "Document category",
            options=list(DocumentCategory),
            format_func=lambda c: c.value.replace("_", " ").title(),
        )
        if st.button(
            "⬆️ Ingest document", use_container_width=True, disabled=uploaded_file is None
        ):
            try:
                with st.spinner("Ingesting document…"):
                    settings = get_settings()
                    uploads_dir = settings.data_dir / "uploads"
                    uploads_dir.mkdir(parents=True, exist_ok=True)
                    # Persist the upload to disk first: the pipeline's loader
                    # works on paths, and keeping the file gives us a
                    # re-ingestable record of what entered the KB.
                    saved_path = uploads_dir / uploaded_file.name
                    saved_path.write_bytes(uploaded_file.getvalue())
                    report = knowledge_base.ingest_pdf(
                        saved_path, category=upload_category
                    )
                st.session_state["last_ingest_message"] = (
                    f"✅ Ingested {report.source}: {report.pages} pages → "
                    f"{report.chunks} chunks in {report.duration_seconds}s."
                )
                # Rerun so the chunk/category stats at the top reflect the
                # freshly ingested document.
                st.rerun()
            except HealthPlusError as exc:
                # Covers DocumentLoadError and EmptyDocumentError (bad or
                # scanned PDFs) with the pipeline's own explanation.
                st.error(f"Could not ingest {uploaded_file.name}: {exc}")


# ---------------------------------------------------------------------------
# Main area: hero header, history, and the chat loop
# ---------------------------------------------------------------------------

provider_label = "Claude (Anthropic)" if selected_provider == "claude" else "ChatGPT (OpenAI)"
provider_icon  = "⚡" if selected_provider == "claude" else "🤖"

st.markdown(
    f"""
    <div class="hp-hero">
        <div class="hp-hero-icon">🏥</div>
        <div class="hp-hero-text">
            <h1>HealthPlus AI Assistant</h1>
            <p>{DISCLAIMER} &nbsp;·&nbsp; {provider_icon} {provider_label}</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Replay the visible history every rerun — Streamlit redraws from scratch.
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            _render_sources(message.get("sources", []))

# Empty-state welcome screen
if not st.session_state["messages"]:
    st.markdown(
        """
        <div class="hp-welcome">
            <div class="hp-welcome-icon">💬</div>
            <h3>How can I help you today?</h3>
            <p>Ask me anything about doctors, diagnostic tests, pricing, or hospital policies.</p>
            <div class="hp-chips">
                <span class="hp-chip">🩺 Available specialists</span>
                <span class="hp-chip">🧪 Lab test pricing</span>
                <span class="hp-chip">📋 Hospital policies</span>
                <span class="hp-chip">💊 Treatment options</span>
                <span class="hp-chip">🏨 Ward facilities</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# The chat service needs an API key for the selected provider. If it is
# missing we stop HERE — after the sidebar — so stats and uploads above
# remain fully usable.
try:
    chat_service = _get_chat_service(selected_provider)
except ConfigurationError as exc:
    key_hint = (
        "ANTHROPIC_API_KEY" if selected_provider == "claude" else "OPENAI_API_KEY"
    )
    st.error(
        f"Chat is not available: {exc}\n\n"
        f"To fix this, add `{key_hint}` to your `.env` file, then restart the app."
    )
    st.stop()
except HealthPlusError as exc:
    st.error(f"Chat is not available: {exc}")
    st.stop()

prompt = st.chat_input("Ask about doctors, tests, pricing, policies…")
if prompt:
    # Echo the user's message immediately so the app feels responsive while
    # retrieval and the Claude call happen below.
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state["messages"].append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        try:
            answer = chat_service.answer(
                prompt,
                st.session_state["conversation_id"],
                category=selected_category,
            )
        except ValueError as exc:
            # The query processor rejects empty or over-long questions.
            st.warning(str(exc))
        except HealthPlusError as exc:
            st.error(f"Something went wrong answering that: {exc}")
        else:
            source_dicts = [s.model_dump() for s in answer.sources]
            try:
                # write_stream renders tokens as they arrive and returns the
                # full concatenated text once the stream is exhausted.
                full_text = st.write_stream(answer.tokens)
            except LLMError as exc:
                # The Claude call failed mid-flight (network, rate limit,
                # server error). The service intentionally did NOT record
                # this turn in memory, so the conversation stays consistent.
                st.error(f"The assistant could not reply: {exc}")
            else:
                _render_sources(source_dicts)
                st.session_state["messages"].append(
                    {
                        "role": "assistant",
                        "content": full_text,
                        "sources": source_dicts,
                    }
                )
