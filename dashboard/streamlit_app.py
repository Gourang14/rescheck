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

st.markdown("<div class='title'> Resume Relevance Checker </div>", unsafe_allow_html=True)
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
st.markdown("## 1) Upload Job Description")
jd_file = st.file_uploader("Upload JD (PDF / DOCX / TXT)", type=["pdf", "docx", "txt"])
jd_text = ""
if jd_file:
    tmp_jd = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(jd_file.name)[1])
    tmp_jd.write(jd_file.getbuffer())
    tmp_jd.close()
    try:
        if jd_file.name.lower().endswith(".pdf"):
            jd_text = extract_text_from_pdf(tmp_jd.name)
        elif jd_file.name.lower().endswith(".docx"):
            jd_text = extract_text_from_docx(tmp_jd.name)
        else:
            jd_text = open(tmp_jd.name, "r", encoding="utf-8", errors="ignore").read()
    except Exception as e:
        st.error(f"Failed to extract JD text: {e}")
        jd_text = ""

if jd_text:
    st.markdown("**JD preview (first 800 chars)**")
    st.code(jd_text[:800] + ("..." if len(jd_text) > 800 else ""))

    jd_sections = section_split(jd_text)

    # --- Auto-skill extraction ---
    candidates = re.findall(r'\b[A-Z][a-zA-Z0-9\+\#\.]{2,}\b', jd_text)
    candidates = list(dict.fromkeys(candidates))  # dedupe
    auto_skills = candidates[:25]

    st.markdown("### 2) Skills (auto-detected — edit if needed)")
    col1, col2 = st.columns([2, 1])
    with col1:
        must_text = st.text_area("Must-have skills (comma separated)", value=",".join(auto_skills), height=120)
        nice_text = st.text_area("Nice-to-have skills (comma separated)", value="", height=80)
    with col2:
        st.markdown("**Quick actions**")
        if st.button("Use auto-detected skills"):
            st.session_state["must_text"] = ",".join(auto_skills)
            st.experimental_rerun()
        st.markdown("**Embedding provider**")
        st.write(f"Using: **{type(emb_provider).__name__}**")

    must_have = [s.strip() for s in (st.session_state.get("must_text", must_text) or "").split(",") if s.strip()]
    nice_to_have = [s.strip() for s in (nice_text or "").split(",") if s.strip()]
else:
    st.info("Please upload a Job Description to enable resume evaluation.")
    must_have, nice_to_have = [], []

st.markdown("---")

# --- Resume Upload ---
st.markdown("## 3) Upload Resumes")
uploaded = st.file_uploader("Upload Resumes (PDF/DOCX) — multiple allowed", accept_multiple_files=True, type=["pdf", "docx"])

if uploaded and not jd_text:
    st.warning("Upload a Job Description first.")
if uploaded and jd_text:
    st.info(f"Processing {len(uploaded)} resumes...")
    progress = st.progress(0)
    results, failed = [], []
    total = len(uploaded)

    for idx, f in enumerate(uploaded, start=1):
        status_placeholder = st.empty()
        status_placeholder.info(f"Parsing {f.name} ({idx}/{total})...")
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(f.name)[1])
        tmp.write(f.getbuffer())
        tmp.close()

        try:
            text = extract_resume_text(tmp.name)
        except Exception as e:
            status_placeholder.error(f"Failed to parse {f.name}: {e}")
            failed.append({"filename": f.name, "error": str(e)})
            progress.progress(int((idx / total) * 100))
            continue

        sections = section_split(text)

        with st.expander(f"Preview parsed sections — {f.name}", expanded=False):
            if isinstance(sections, dict):
        # new dict format
                for k, v in sections.items():
                    st.markdown(f"**{k.title()}**")
                    st.code(v[:800] + ("..." if len(v) > 800 else ""))
            elif isinstance(sections, list):
        # fallback if old list format
                for idx, sec in enumerate(sections, 1):
                    st.markdown(f"**Section {idx}**")
                    st.code(sec[:800] + ("..." if len(sec) > 800 else ""))
            else:
                st.warning("Could not parse sections properly.")

        # scoring
        try:
            hard = scorer.hard_score(text, must_have, nice_to_have)
        except Exception as e:
            hard = 0.0
            st.warning(f"Hard scoring failed for {f.name}: {e}")

        try:
            jd_vec = emb_provider.embed([jd_text])[0]
            resume_vec = emb_provider.embed([text])[0]
            soft = scorer.soft_score(jd_vec, resume_vec)
        except Exception as e:
            st.warning(f"Soft score failed for {f.name}: {e}")
            soft = 0.0

        final = scorer.final_score(hard, soft)
        verdict = scorer.verdict(final)

        # missing skills
        try:
            skill_map = extract_skills(text, must_have)
            matched = {k for k, v in skill_map.items() if v >= 70}
            missing = [k for k in must_have if k not in matched]
        except Exception:
            matched, missing = set(), must_have.copy()

        # feedback
        feedback_text = "Feedback disabled (no API key or LLM disabled)."
        if use_groq and st.session_state.get("api_key"):
            os.environ["GROQ_API_KEY"] = st.session_state["api_key"]
            try:
                feedback_text = generate_feedback(text, jd_text, missing)[:3000]
            except Exception as e:
                feedback_text = f"Feedback generation failed: {e}"

        results.append({
            "filename": f.name,
            "score": round(final, 2),
            "verdict": verdict,
            "hard": round(hard, 2),
            "soft": round(soft, 2),
            "missing": missing,
            "feedback": feedback_text
        })

        progress.progress(int((idx / total) * 100))
        status_placeholder.success(f"Processed {f.name}: score {final:.1f} — {verdict}")

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
