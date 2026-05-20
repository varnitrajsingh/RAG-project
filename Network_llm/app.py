import streamlit as st
import os
import sys
import traceback
from dotenv import load_dotenv


load_dotenv()


def get_secret(key: str) -> str:
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return os.getenv(key, "")


for _key in ["OPENAI_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY", "HUGGINGFACE_API_KEY", "GEMINI_API_KEY"]:
    _val = get_secret(_key)
    if _val:
        os.environ[_key] = _val

# ── DEBUG: Startup ─────────────────────────────────────────────────────────────
print("=" * 60)
print("🔍 DEBUG: Environment Key Status at Startup")
print(f"  GEMINI_API_KEY set: {bool(os.environ.get('GEMINI_API_KEY'))}")
print(f"  GOOGLE_API_KEY set: {bool(os.environ.get('GOOGLE_API_KEY'))}")
print("=" * 60)

os.makedirs("data/uploads", exist_ok=True)
os.makedirs("vectorstore", exist_ok=True)

pipeline_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Pipeline")
sys.path.insert(0, pipeline_path)

print(f"🔍 DEBUG: Pipeline path: {pipeline_path}")
print(f"🔍 DEBUG: Pipeline path exists: {os.path.exists(pipeline_path)}")

try:
    from ingest import process_pdf
    print("✅ DEBUG: ingest imported successfully")
except Exception as e:
    print(f"❌ DEBUG: Failed to import ingest: {e}")
    traceback.print_exc()

try:
    from retrieve import query_pipeline
    print("✅ DEBUG: retrieve imported successfully")
except Exception as e:
    print(f"❌ DEBUG: Failed to import retrieve: {e}")
    traceback.print_exc()


st.set_page_config(
    page_title="Network Engineer Assistant",
    page_icon="🔧",
    layout="centered"
)

if "theme" not in st.session_state:
    st.session_state.theme = "dark"

theme = st.session_state.theme
is_dark = theme == "dark"

if is_dark:
    BG          = "#0d1117"
    SURFACE     = "#161b22"
    SURFACE2    = "#1c2128"
    BORDER      = "#30363d"
    TEXT        = "#e6edf3"
    TEXT_MUTED  = "#7d8590"
    TEXT_FAINT  = "#484f58"
    ACCENT      = "#2f81f7"
    ACCENT_GLOW = "rgba(47,129,247,0.18)"
    SUCCESS_BG  = "rgba(46,160,67,0.15)"
    SUCCESS_FG  = "#3fb950"
    SUCCESS_BD  = "rgba(46,160,67,0.3)"
    INFO_BG     = "rgba(47,129,247,0.12)"
    INFO_FG     = "#58a6ff"
    INFO_BD     = "rgba(47,129,247,0.25)"
    WARN_BG     = "rgba(210,153,34,0.15)"
    WARN_FG     = "#d29922"
    WARN_BD     = "rgba(210,153,34,0.3)"
    ERR_BG      = "rgba(248,81,73,0.15)"
    ERR_FG      = "#f85149"
    ERR_BD      = "rgba(248,81,73,0.3)"
    INPUT_BG    = "#0d1117"
    SCROLLBAR   = "#30363d"
    LOGO_FILTER = "drop-shadow(0 0 12px rgba(47,129,247,0.4))"
    GRADIENT_A  = "#2f81f7"
    GRADIENT_B  = "#79c0ff"
else:
    BG          = "#f6f8fa"
    SURFACE     = "#ffffff"
    SURFACE2    = "#f0f3f6"
    BORDER      = "#d0d7de"
    TEXT        = "#1f2328"
    TEXT_MUTED  = "#57606a"
    TEXT_FAINT  = "#8c959f"
    ACCENT      = "#0969da"
    ACCENT_GLOW = "rgba(9,105,218,0.15)"
    SUCCESS_BG  = "rgba(31,136,61,0.08)"
    SUCCESS_FG  = "#1a7f37"
    SUCCESS_BD  = "rgba(31,136,61,0.2)"
    INFO_BG     = "rgba(9,105,218,0.08)"
    INFO_FG     = "#0550ae"
    INFO_BD     = "rgba(9,105,218,0.2)"
    WARN_BG     = "rgba(154,103,0,0.08)"
    WARN_FG     = "#7d4e00"
    WARN_BD     = "rgba(154,103,0,0.2)"
    ERR_BG      = "rgba(207,34,46,0.08)"
    ERR_FG      = "#cf222e"
    ERR_BD      = "rgba(207,34,46,0.2)"
    INPUT_BG    = "#ffffff"
    SCROLLBAR   = "#d0d7de"
    LOGO_FILTER = "drop-shadow(0 0 10px rgba(9,105,218,0.25))"
    GRADIENT_A  = "#0969da"
    GRADIENT_B  = "#0550ae"


st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif !important;
}}
.stApp {{
    background: {BG};
    color: {TEXT};
}}
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{
    max-width: 800px !important;
    padding: 0 1.5rem 3rem 1.5rem !important;
}}
.app-hero {{
    text-align: center;
    padding: 2.5rem 1rem 2rem 1rem;
}}
.hero-icon {{
    font-size: 3rem;
    margin-bottom: 0.75rem;
    filter: {LOGO_FILTER};
    display: block;
}}
.hero-title {{
    font-size: 2rem;
    font-weight: 700;
    color: {TEXT};
    letter-spacing: -0.03em;
    margin: 0 0 0.5rem 0;
    line-height: 1.2;
}}
.hero-title span {{
    background: linear-gradient(135deg, {GRADIENT_A}, {GRADIENT_B});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}}
.hero-subtitle {{
    font-size: 0.95rem;
    color: {TEXT_MUTED};
    margin: 0 auto;
    line-height: 1.5;
    white-space: nowrap;
}}
.hero-divider {{
    height: 1px;
    background: linear-gradient(to right, transparent, {BORDER}, transparent);
    margin: 0 0 1.5rem 0;
    border: none;
}}
[data-testid="stFileUploader"] {{ margin-top: 0 !important; }}
[data-testid="stFileUploader"] > label {{
    display: none !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
}}
[data-testid="stFileUploader"] > div:first-of-type {{ margin-top: 0 !important; }}
[data-testid="stFileUploader"] > div {{
    background: {SURFACE2} !important;
    border: 1.5px dashed {BORDER} !important;
    border-radius: 10px !important;
    transition: border-color 0.2s ease, background 0.2s ease !important;
}}
[data-testid="stFileUploader"] > div:hover {{
    border-color: {ACCENT} !important;
    background: {ACCENT_GLOW} !important;
}}
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p {{
    color: {TEXT_MUTED} !important;
    font-size: 0.875rem !important;
}}
[data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"] {{
    background: {SURFACE} !important;
    border: 1px solid {BORDER} !important;
    color: {TEXT} !important;
    font-size: 0.8rem !important;
    border-radius: 6px !important;
    padding: 0.3rem 0.9rem !important;
}}
.section-card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 1.25rem 1.5rem 1.5rem 1.5rem;
    margin-bottom: 1.25rem;
    box-shadow: {"0 1px 3px rgba(0,0,0,0.3)" if is_dark else "0 1px 3px rgba(0,0,0,0.06)"};
}}
.section-label {{
    display: flex;
    align-items: center;
    gap: 0.45rem;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: {TEXT_FAINT};
    margin-bottom: 1rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid {BORDER};
}}
.section-label .label-count {{
    margin-left: auto;
    background: {SURFACE2};
    color: {TEXT_MUTED};
    border-radius: 9999px;
    padding: 0.1rem 0.55rem;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0;
    text-transform: none;
}}
.status-badge {{
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.8rem;
    font-weight: 500;
    padding: 0.4rem 0.85rem;
    border-radius: 8px;
    margin-top: 0.65rem;
}}
.status-badge.success {{ background: {SUCCESS_BG}; color: {SUCCESS_FG}; border: 1px solid {SUCCESS_BD}; }}
.status-badge.info    {{ background: {INFO_BG};    color: {INFO_FG};    border: 1px solid {INFO_BD}; }}
.status-badge.warning {{ background: {WARN_BG};    color: {WARN_FG};    border: 1px solid {WARN_BD}; }}
.status-badge.error   {{ background: {ERR_BG};     color: {ERR_FG};     border: 1px solid {ERR_BD}; }}
[data-testid="stTextInput"] input {{
    background: {INPUT_BG} !important;
    border: 1.5px solid {BORDER} !important;
    border-radius: 8px !important;
    color: {TEXT} !important;
    font-size: 0.9rem !important;
    padding: 0.65rem 1rem !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}}
[data-testid="stTextInput"] input:focus {{
    border-color: {ACCENT} !important;
    box-shadow: 0 0 0 3px {ACCENT_GLOW} !important;
    outline: none !important;
}}
[data-testid="stTextInput"] input::placeholder {{ color: {TEXT_FAINT} !important; }}
[data-testid="stTextInput"] > label {{
    display: none !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}}
[data-testid="stFormSubmitButton"] button {{
    background: {ACCENT} !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 0.875rem !important;
    font-weight: 600 !important;
    padding: 0.65rem 1.5rem !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
    letter-spacing: 0.01em !important;
    box-shadow: 0 2px 8px {ACCENT_GLOW} !important;
}}
[data-testid="stFormSubmitButton"] button:hover {{
    opacity: 0.88 !important;
    box-shadow: 0 4px 16px {ACCENT_GLOW} !important;
    transform: translateY(-1px) !important;
}}
.clear-btn > div > button {{
    background: transparent !important;
    border: 1px solid {BORDER} !important;
    color: {TEXT_MUTED} !important;
    width: auto !important;
    padding: 0.35rem 1rem !important;
    font-size: 0.78rem !important;
    border-radius: 6px !important;
    box-shadow: none !important;
    transition: all 0.18s ease !important;
}}
.clear-btn > div > button:hover {{
    border-color: {ERR_FG} !important;
    color: {ERR_FG} !important;
    background: {ERR_BG} !important;
    transform: none !important;
    box-shadow: none !important;
}}
[data-testid="stChatMessage"] {{
    background: {SURFACE2} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 10px !important;
    padding: 0.75rem 1rem !important;
    margin-bottom: 0.5rem !important;
}}
::-webkit-scrollbar {{ width: 5px; }}
::-webkit-scrollbar-track {{ background: {BG}; }}
::-webkit-scrollbar-thumb {{ background: {SCROLLBAR}; border-radius: 99px; }}
::-webkit-scrollbar-thumb:hover {{ background: {ACCENT}; }}
hr {{ border-color: {BORDER} !important; }}
</style>
""", unsafe_allow_html=True)


# ── Theme Toggle ──────────────────────────────────────────────────────────────
_, col_btn = st.columns([7, 1])
with col_btn:
    if st.button("☀️" if is_dark else "🌙", key="theme_toggle", help="Toggle theme"):
        st.session_state.theme = "light" if is_dark else "dark"
        st.rerun()


# ── Hero Header ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="app-hero">
    <span class="hero-icon">🔧</span>
    <h1 class="hero-title">Network Engineer <span>Assistant</span></h1>
    <p class="hero-subtitle">Upload network documentation and query it with natural language</p>
</div>
<div class="hero-divider"></div>
""", unsafe_allow_html=True)


# ── Upload Section ────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">📄 Document</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "x",
    type=["pdf"],
    label_visibility="hidden"
)

if uploaded_file:
    file_path = f"data/uploads/{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    print(f"🔍 DEBUG: File uploaded: {uploaded_file.name}")
    print(f"🔍 DEBUG: Saved to: {file_path} | Exists: {os.path.exists(file_path)}")

    st.markdown(
        f'<div class="status-badge success">✓ Uploaded: <strong>{uploaded_file.name}</strong></div>',
        unsafe_allow_html=True
    )

    if st.session_state.get("last_uploaded") != uploaded_file.name:
        with st.spinner("Building vector index…"):
            try:
                print(f"🔍 DEBUG: Calling process_pdf({file_path})")
                
                import shutil
                if os.path.exists("vectorstore"):
                    shutil.rmtree("vectorstore")
                os.makedirs("vectorstore", exist_ok=True)

                process_pdf(file_path)
                st.session_state["last_uploaded"] = uploaded_file.name
                print(f"✅ DEBUG: process_pdf done. last_uploaded = {st.session_state['last_uploaded']}")
                st.markdown(
                    '<div class="status-badge success">✓ Document indexed and ready</div>',
                    unsafe_allow_html=True
                )
            except Exception as e:
                print(f"❌ DEBUG: process_pdf failed: {e}")
                print(traceback.format_exc())
                st.markdown(
                    f'<div class="status-badge error">✕ Failed to process: {e}</div>',
                    unsafe_allow_html=True
                )
    else:
        print(f"🔍 DEBUG: Already indexed: {uploaded_file.name}")
        st.markdown(
            '<div class="status-badge info">ℹ Already indexed — ready to query</div>',
            unsafe_allow_html=True
        )

st.markdown('</div>', unsafe_allow_html=True)


# ── Chat Section ──────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

st.markdown('<div class="section-label">💬 Ask a Question</div>', unsafe_allow_html=True)

with st.form(key="query_form", clear_on_submit=True):
    query = st.text_input(
        "q",
        placeholder="e.g. What port does BGP use?",
        label_visibility="hidden"
    )
    submitted = st.form_submit_button("Send Message →")

if submitted and query.strip():
    if st.session_state.get("last_uploaded") is None:
        print("⚠️ DEBUG: Query submitted but no PDF uploaded")
        st.markdown(
            '<div class="status-badge warning">⚠ Upload and process a PDF first</div>',
            unsafe_allow_html=True
        )
    else:
        pdf_name = st.session_state.get("last_uploaded", "")
        print(f"🔍 DEBUG: Query = '{query}'")
        print(f"🔍 DEBUG: pdf_name = '{pdf_name}'")
        print(f"🔍 DEBUG: GEMINI_API_KEY present = {bool(os.environ.get('GEMINI_API_KEY'))}")
        with st.spinner("Thinking…"):
            try:
                print("🔍 DEBUG: Calling query_pipeline...")
                answer = query_pipeline(query, pdf_name)
                print(f"🔍 DEBUG: Answer type = {type(answer)}")
                print(f"🔍 DEBUG: Answer preview = {str(answer)[:150] if answer else 'NONE/EMPTY'}")
                st.session_state.history.append({"q": query, "a": answer})
                print("✅ DEBUG: Appended to history OK")
            except Exception as e:
                print(f"❌ DEBUG: query_pipeline exception: {e}")
                print(traceback.format_exc())
                st.markdown(
                    f'<div class="status-badge error">✕ Error: {e}</div>',
                    unsafe_allow_html=True
                )
elif submitted and not query.strip():
    print("⚠️ DEBUG: Empty query")
    st.markdown(
        '<div class="status-badge warning">⚠ Please enter a question</div>',
        unsafe_allow_html=True
    )

st.markdown('</div>', unsafe_allow_html=True)


# ── Conversation History ──────────────────────────────────────────────────────
if st.session_state.history:
    turn_count = len(st.session_state.history)
    st.markdown(
        f'<div class="section-label">🗂 Conversation'
        f'<span class="label-count">{turn_count} turn{"s" if turn_count != 1 else ""}</span>'
        f'</div>',
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