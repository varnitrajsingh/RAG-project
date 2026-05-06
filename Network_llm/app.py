import streamlit as st
import os
import sys
from dotenv import load_dotenv

# ── Secret Loader (works locally via .env AND on Streamlit Cloud) ─────────────
load_dotenv()

def get_secret(key: str) -> str:
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return os.getenv(key, "")

# ── Inject secrets into environment (so Pipeline files can use os.getenv) ─────
for _key in ["OPENAI_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY", "HUGGINGFACE_API_KEY"]:
    _val = get_secret(_key)
    if _val:
        os.environ[_key] = _val

# ── Ensure required directories exist (important for cold starts on cloud) ────
os.makedirs("data/uploads", exist_ok=True)
os.makedirs("vectorstore", exist_ok=True)

# ── Add Pipeline folder to path ───────────────────────────────────────────────
pipeline_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Pipeline")
sys.path.insert(0, pipeline_path)

from ingest import process_pdf
from retrieve import query_pipeline

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Network Engineer Assistant",
    page_icon="🔧",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}
.stApp {
    background: #0f1117;
    color: #e2e8f0;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    max-width: 780px !important;
    padding: 2.5rem 2rem !important;
}

/* Header */
.app-header {
    text-align: center;
    padding: 2rem 0 1.5rem 0;
    border-bottom: 1px solid #1e2533;
    margin-bottom: 2rem;
}
.app-header .app-logo { font-size: 2.8rem; margin-bottom: 0.5rem; }
.app-header h1 {
    font-size: 1.75rem;
    font-weight: 700;
    color: #f1f5f9;
    letter-spacing: -0.02em;
    margin: 0 0 0.4rem 0;
}
.app-header p { font-size: 0.9rem; color: #64748b; margin: 0; }

/* Section Cards */
.section-card {
    background: #161b27;
    border: 1px solid #1e2533;
    border-radius: 12px;
    padding: 1.5rem 1.75rem;
    margin-bottom: 1.5rem;
}
.section-label {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #475569;
    margin-bottom: 1rem;
}

/* Status Badges */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.8rem;
    font-weight: 500;
    padding: 0.35rem 0.75rem;
    border-radius: 9999px;
    margin-top: 0.75rem;
}
.status-badge.success {
    background: rgba(16,185,129,0.12);
    color: #10b981;
    border: 1px solid rgba(16,185,129,0.2);
}
.status-badge.info {
    background: rgba(59,130,246,0.12);
    color: #3b82f6;
    border: 1px solid rgba(59,130,246,0.2);
}
.status-badge.warning {
    background: rgba(245,158,11,0.12);
    color: #f59e0b;
    border: 1px solid rgba(245,158,11,0.2);
}
.status-badge.error {
    background: rgba(239,68,68,0.12);
    color: #ef4444;
    border: 1px solid rgba(239,68,68,0.2);
}

/* File Uploader */
[data-testid="stFileUploader"] {
    background: #0f1117;
    border: 1.5px dashed #2d3748;
    border-radius: 10px;
    transition: border-color 0.2s ease;
}
[data-testid="stFileUploader"]:hover { border-color: #3b82f6; }
[data-testid="stFileUploader"] label {
    color: #94a3b8 !important;
    font-size: 0.875rem !important;
}

/* Text Input */
[data-testid="stTextInput"] input {
    background: #0f1117 !important;
    border: 1.5px solid #1e2533 !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 0.9rem !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.15) !important;
    outline: none !important;
}
[data-testid="stTextInput"] input::placeholder { color: #475569 !important; }
[data-testid="stTextInput"] label {
    color: #94a3b8 !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
}

/* Buttons */
[data-testid="stFormSubmitButton"] button,
.stButton > button {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 0.875rem !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.01em !important;
    width: 100% !important;
}
[data-testid="stFormSubmitButton"] button:hover,
.stButton > button:hover {
    background: linear-gradient(135deg, #1d4ed8, #1e3a8a) !important;
    box-shadow: 0 4px 15px rgba(37,99,235,0.35) !important;
    transform: translateY(-1px) !important;
}

/* Clear History Button */
.clear-btn button {
    background: transparent !important;
    border: 1px solid #2d3748 !important;
    color: #64748b !important;
    width: auto !important;
    padding: 0.4rem 1rem !important;
    font-size: 0.8rem !important;
}
.clear-btn button:hover {
    border-color: #ef4444 !important;
    color: #ef4444 !important;
    background: rgba(239,68,68,0.08) !important;
    box-shadow: none !important;
    transform: none !important;
}

/* Chat Messages */
[data-testid="stChatMessage"] {
    background: transparent !important;
    padding: 0.25rem 0 !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0f1117; }
::-webkit-scrollbar-thumb { background: #2d3748; border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: #3b82f6; }

hr { border-color: #1e2533 !important; margin: 1.25rem 0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="app-logo">🔧</div>
    <h1>Network Engineer Assistant</h1>
    <p>Upload network documentation and query it with natural language</p>
</div>
""", unsafe_allow_html=True)

# ── Upload Section ────────────────────────────────────────────────────────────
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-label">📄 Document</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"], label_visibility="collapsed")

if uploaded_file:
    file_path = f"data/uploads/{uploaded_file.name}"

    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.markdown(
        f'<div class="status-badge success">✓ Uploaded: {uploaded_file.name}</div>',
        unsafe_allow_html=True
    )

    if st.session_state.get("last_uploaded") != uploaded_file.name:
        with st.spinner("Building index…"):
            try:
                process_pdf(file_path)
                st.session_state["last_uploaded"] = uploaded_file.name
                st.markdown(
                    '<div class="status-badge success">✓ Document indexed and ready</div>',
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.markdown(
                    f'<div class="status-badge error">✕ Failed to process: {e}</div>',
                    unsafe_allow_html=True
                )
    else:
        st.markdown(
            '<div class="status-badge info">ℹ Already indexed</div>',
            unsafe_allow_html=True
        )

st.markdown('</div>', unsafe_allow_html=True)

# ── Chat Section ──────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-label">💬 Ask a Question</div>', unsafe_allow_html=True)

with st.form(key="query_form", clear_on_submit=True):
    query = st.text_input(
        "Question",
        placeholder="e.g. What port does BGP use?",
        label_visibility="collapsed"
    )
    submitted = st.form_submit_button("Send Message →")

if submitted and query.strip():
    if st.session_state.get("last_uploaded") is None:
        st.markdown(
            '<div class="status-badge warning">⚠ Upload a PDF document first</div>',
            unsafe_allow_html=True
        )
    else:
        with st.spinner("Thinking…"):
            try:
                answer = query_pipeline(query)
                st.session_state.history.append({"q": query, "a": answer})
            except Exception as e:
                st.markdown(
                    f'<div class="status-badge error">✕ Error: {e}</div>',
                    unsafe_allow_html=True
                )
elif submitted and not query.strip():
    st.markdown(
        '<div class="status-badge warning">⚠ Please enter a question</div>',
        unsafe_allow_html=True
    )

st.markdown('</div>', unsafe_allow_html=True)

# ── Chat History ──────────────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(
        f'<div class="section-label">🗂 Conversation History '
        f'<span style="margin-left:auto;color:#3b82f6;font-size:0.7rem;">'
        f'{len(st.session_state.history)} turn(s)</span></div>',
        unsafe_allow_html=True
    )

    for turn in reversed(st.session_state.history):
        with st.chat_message("user"):
            st.markdown(turn["q"])
        with st.chat_message("assistant"):
            st.markdown(turn["a"])

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)