# streamlit_app.py
import sys, os, re, tempfile
import streamlit as st
import pandas as pd
from typing import List

# Calculate the absolute path to the project root ("rescheck")
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# backend services
from backend.services.parser import extract_resume_text, extract_text_from_pdf, extract_text_from_docx
from backend.services.nlp_utils import section_split, extract_skills
from backend.services.embeddings_manager import SBERTEmbeddingsProvider, OpenAIEmbeddingsProvider
from backend.services.scoring import Scorer
from backend.services.feedback import generate_feedback
from backend.services.vectorstore_wrapper import VectorStoreWrapper

# --- Page / theme setup ---
st.set_page_config(page_title="Resume Relevance", layout="wide", initial_sidebar_state="expanded")

CSS = """
<style>
    .stApp, .main {
        background-color: #0b0b0d;
        color: #e6e6e6;
    }
    .stButton>button { background-color: #111; color: #fff; border-radius:8px; }
    .title { font-family: 'Helvetica', sans-serif; color:#FFD700; font-weight:700; font-size:28px; }
    .card { background: linear-gradient(90deg, rgba(0,0,0,0.85), rgba(20,20,20,0.85)); border-radius:12px; padding:12px; margin-bottom:12px; box-shadow: 0 2px 6px rgba(0,0,0,0.5); }
    .chip { padding:6px 10px; border-radius:999px; color:#fff; font-weight:700; display:inline-block; }
    .chip-high { background: linear-gradient(90deg,#00b894,#00cec9); }
    .chip-medium { background: linear-gradient(90deg,#ffeaa7,#fab1a0); color:#1b1b1b; }
    .chip-low { background: linear-gradient(90deg,#ff7675,#d63031); }
    .tag { display:inline-block; padding:4px 8px; border-radius:6px; background:#111; margin:2px; color:#fff; border:1px solid #222;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

st.markdown("<div class='title'>Resume Relevance Checker</div>", unsafe_allow_html=True)
st.caption("Upload a Job Description and candidate resumes. The system returns a relevance score, missing skills, LLM feedback, and downloadable results.")

# --- Sidebar: Settings & API Key ---
with st.sidebar:
    st.header("Settings & API Key")
    if "api_key" not in st.session_state:
        st.session_state["api_key"] = ""
    api_key = st.text_input("API key (session only)", type="password", value=st.session_state["api_key"], key="api_key_input")
    use_groq = st.checkbox("Enable Groq LLM for feedback (requires key)", value=True)
    if api_key:
        st.session_state["api_key"] = api_key

    st.markdown("---")
    st.subheader("Embedding / Vector settings")
    store_kind = st.selectbox("Vector store (for later)", ["none", "chroma", "faiss", "pinecone"])
    embedding_choice = st.selectbox("Embeddings", ["sbert", "openai"])
    st.markdown("---")
    st.subheader("Scoring weights")
    hard_weight = st.slider("Hard-match weight", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
    soft_weight = round(1.0 - hard_weight, 2)
    st.markdown(f"**Soft-match weight:** {soft_weight}")
    st.markdown("---")
    if st.button("Clear session state"):
        for k in list(st.session_state.keys()):
            st.session_state.pop(k, None)
        st.experimental_rerun()

if use_groq and not st.session_state.get("api_key"):
    st.warning("Enter API key in sidebar to enable LLM feedback generation. You can still run embeddings with SBERT.")

# helper for verdict chips
def verdict_chip_html(verdict: str) -> str:
    if verdict == "High":
        return "<span class='chip chip-high'>✅ High</span>"
    if verdict == "Medium":
        return "<span class='chip chip-medium'>⚠️ Medium</span>"
    return "<span class='chip chip-low'>❌ Low</span>"

# cached providers
@st.cache_resource
def get_sbert_provider():
    return SBERTEmbeddingsProvider()

@st.cache_resource
def get_openai_provider(api_key):
    os.environ["OPENAI_API_KEY"] = api_key
    return OpenAIEmbeddingsProvider(api_key)

# embedding provider init
if embedding_choice == "sbert":
    try:
        emb_provider = get_sbert_provider()
    except Exception as e:
        st.error(f"Failed to initialize SBERT embeddings: {e}")
else:
    if st.session_state.get("api_key"):
        try:
            emb_provider = get_openai_provider(st.session_state["api_key"])
        except Exception:
            st.warning("OpenAI init failed, falling back to SBERT.")
            emb_provider = get_sbert_provider()
    else:
        st.warning("No API key found; using SBERT embeddings.")
        emb_provider = get_sbert_provider()

# scorer
scorer = Scorer(weights={"hard": hard_weight, "soft": soft_weight})

# --- JD Upload ---
# inside dashboard/streamlit_app.py — JD upload block (replace existing JD upload UI)
import streamlit as st, requests, tempfile, os, traceback

st.sidebar.header("Groq settings")
groq_url = st.sidebar.text_input("Groq API URL (full endpoint)", value=os.getenv("GROQ_API_URL",""))
groq_key = st.sidebar.text_input("Groq API Key (optional)", type="password")
use_groq = st.sidebar.checkbox("Use Groq for JD parsing", value=True)

API_BASE = st.sidebar.text_input("Backend API", value=os.getenv("API_BASE","http://localhost:8000"))

st.header("Upload Job Description (JD)")
jd_file = st.file_uploader("JD (pdf/docx/txt)", type=["pdf","docx","txt"])
if jd_file:
    raw_bytes = jd_file.read()
    st.subheader("Preview (first 800 chars)")
    try:
        # try to show text quickly
        import tempfile, os
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix="." + jd_file.name.split(".")[-1])
        tmp.write(raw_bytes); tmp.close()
        from backend.services.parser import extract_text
        txt = extract_text(tmp.name)
        os.unlink(tmp.name)
    except Exception:
        try:
            txt = raw_bytes.decode("utf-8", errors="ignore")
        except:
            txt = "<binary content>"
    st.code(txt[:800])

    if st.button("Upload JD to backend (use Groq)"):
        files = {"file": (jd_file.name, raw_bytes)}
        data = {"use_groq": str(bool(use_groq)).lower()}
        if groq_url:
            data["model_url"] = groq_url
        if groq_key:
            data["api_key"] = groq_key
        try:
            resp = requests.post(f"{API_BASE}/jd/upload", files=files, data=data, timeout=120)
            resp.raise_for_status()
            out = resp.json()
            st.success(f"JD uploaded (job_id={out['job_id']})")
            st.subheader("Parsed JD")
            st.json(out["parsed"])
            st.session_state["job_id"] = out["job_id"]
        except Exception as e:
            st.error("JD upload/parsing failed. See error below.")
            st.text(str(e))
            try:
                st.text(resp.text[:2000])
            except:
                st.text(traceback.format_exc())


    # --- Results ---
    st.markdown("## Results")
    df = pd.DataFrame(results)
    if df.empty:
        st.info("No successful results.")
    else:
        colf1, colf2, colf3 = st.columns([1,1,2])
        with colf1:
            min_score = st.slider("Minimum score", min_value=0, max_value=100, value=0)
        with colf2:
            verdict_filter = st.selectbox("Verdict", options=["All", "High", "Medium", "Low"], index=0)
        with colf3:
            text_filter = st.text_input("Filename contains", value="")

        df_display = df[df["score"] >= min_score]
        if verdict_filter != "All":
            df_display = df_display[df_display["verdict"] == verdict_filter]
        if text_filter.strip():
            df_display = df_display[df_display["filename"].str.contains(text_filter, case=False, na=False)]

        for _, row in df_display.iterrows():
            c1, c2 = st.columns([4,1])
            with c1:
                st.markdown(f"<div class='card'><b>{row['filename']}</b> — Score: <b>{row['score']}</b> | Hard: {row['hard']} | Soft: {row['soft']}</div>", unsafe_allow_html=True)
                if row['missing']:
                    st.markdown("**Missing skills:**")
                    tags_html = " ".join([f"<span class='tag'>{s}</span>" for s in row['missing']])
                    st.markdown(tags_html, unsafe_allow_html=True)
                else:
                    st.markdown("**Missing skills:** None")
                st.markdown("**Feedback (short):**")
                st.info(row['feedback'])
            with c2:
                st.markdown(verdict_chip_html(row['verdict']), unsafe_allow_html=True)

        csv = df_display.to_csv(index=False)
        st.download_button("Download filtered results (CSV)", csv, file_name="resume_relevance_results.csv", mime="text/csv")

        if store_kind != "none":
            if st.button("Save vectors to vector store"):
                try:
                    vs = VectorStoreWrapper(kind=store_kind)
                    for r in results:
                        vs.upsert(id=r['filename'], vector=emb_provider.embed([r['filename']])[0], metadata={"score": r['score'], "missing": r['missing']})
                    st.success("Saved vectors.")
                except Exception as e:
                    st.error(f"Failed to save vectors: {e}")
