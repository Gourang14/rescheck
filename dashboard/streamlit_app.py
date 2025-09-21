# dashboard/streamlit_app.py
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
from backend.services.jd_parser import parse_jd_with_llm
# storage
from backend.services.storage import save_evaluation, get_all_evaluations

# --- Page / theme setup ---
st.set_page_config(page_title="Resume Relevance", layout="wide", initial_sidebar_state="expanded")

# --- Batman theme CSS (black) ---
st.set_page_config(page_title="Resume Relevance", layout="wide", initial_sidebar_state="expanded")
css = """
<style>
    .stApp, .main {
        background-color: #0b0b0d;
        color: #e6e6e6;
    }
    .css-18e3th9 { background-color: #0b0b0d; }
    .stButton>button { background-color: #111; color: #fff; border-radius:8px; }
    .title { font-family: 'Helvetica', sans-serif; color:#FFD700; font-weight:700;}
    .card { background: linear-gradient(90deg, rgba(0,0,0,0.85), rgba(20,20,20,0.85)); border-radius:12px; padding:12px; }
    .badge-high { color: #0af; font-weight:700; }
</style>
"""
st.markdown(css, unsafe_allow_html=True)

st.caption("#### Upload a Job Description and candidate resumes. The system returns a relevance score, missing skills, LLM feedback, and downloadable results.")


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
try:
    if embedding_choice == "sbert":
        emb_provider = get_sbert_provider()
    else:
        if st.session_state.get("api_key"):
            emb_provider = get_openai_provider(st.session_state["api_key"])
        else:
            st.warning("No API key found; using SBERT embeddings.")
            emb_provider = get_sbert_provider()
except Exception as e:
    st.error(f"Failed to initialize embeddings provider: {e}")
    try:
        emb_provider = get_sbert_provider()
    except Exception:
        emb_provider = None


# scorer (hard/soft combos)
scorer = Scorer(weights={"hard": hard_weight, "soft": soft_weight})


# --- Tabs: Evaluation UI + Stored Results ---
tab_eval, tab_db = st.tabs(["Run Evaluations", "Stored Results"])

# --------------- EVALUATION TAB ---------------
with tab_eval:
    # --- JD Upload ---
    st.markdown("## Upload Job Description")
    jd_file = st.file_uploader("Upload JD (PDF / DOCX / TXT)", type=["pdf", "docx", "txt"], key="jd_file")
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
        finally:
            try:
                os.unlink(tmp_jd.name)
            except Exception:
                pass

    if jd_text:
        st.markdown("**JD preview (first 800 chars)**")
        st.code(jd_text[:800] + ("..." if len(jd_text) > 800 else ""))

        candidates = re.findall(r'\b[A-Z][a-zA-Z0-9\+\#\.]{2,}\b', jd_text)
        candidates = list(dict.fromkeys(candidates))
        auto_skills = candidates[:25]
    else:
        auto_skills = []

    st.markdown("### Skills (auto-detected — edit if needed)")
    col1, col2 = st.columns([2, 1])
    with col1:
        must_text = st.text_area("Must-have skills (comma separated)", value=",".join(auto_skills), height=120, key="must_text")
        nice_text = st.text_area("Nice-to-have skills (comma separated)", value="", height=80, key="nice_text")
    with col2:
        st.markdown("**Quick actions**")
        if st.button("Use auto-detected skills"):
            st.session_state["must_text"] = ",".join(auto_skills)
            st.experimental_rerun()
        st.markdown("**Embedding provider**")
        st.write(f"Using: **{type(emb_provider).__name__ if emb_provider else 'None'}**")

    must_have = [s.strip() for s in (st.session_state.get("must_text", must_text) or "").split(",") if s.strip()]
    nice_to_have = [s.strip() for s in (nice_text or "").split(",") if s.strip()]

    st.markdown("---")

    # --- Resume Upload ---
    st.markdown("## Upload Resumes")
    uploaded = st.file_uploader("Upload Resumes (PDF/DOCX) — multiple allowed", accept_multiple_files=True, type=["pdf", "docx"], key="resumes_uploader")

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
                try:
                    os.unlink(tmp.name)
                except Exception:
                    pass
                continue
            finally:
                try:
                    os.unlink(tmp.name)
                except Exception:
                    pass

            sections = section_split(text)
            with st.expander(f"Preview parsed sections — {f.name}", expanded=False):
                if isinstance(sections, dict):
                    for k, v in sections.items():
                        st.markdown(f"**{k.title()}**")
                        st.code(v[:800] + ("..." if len(v) > 800 else ""))
                elif isinstance(sections, list):
                    for sidx, sec in enumerate(sections, 1):
                        st.markdown(f"**Section {sidx}**")
                        st.code(sec[:800] + ("..." if len(sec) > 800 else ""))
                else:
                    st.warning("Could not parse sections properly.")

            # Hard score
            try:
                hard = scorer.hard_score(text, must_have, nice_to_have)
            except Exception as e:
                hard = 0.0
                st.warning(f"Hard scoring failed for {f.name}: {e}")

            # Soft score
            try:
                if emb_provider is None:
                    soft = 0.0
                else:
                    jd_vec = emb_provider.embed([jd_text])[0]
                    resume_vec = emb_provider.embed([text])[0]
                    soft = scorer.soft_score(jd_vec, resume_vec)
            except Exception as e:
                st.warning(f"Embedding/soft score failed for {f.name}: {e}")
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

            eval_record = {
                "filename": f.name,
                "score": round(final, 2),
                "verdict": verdict,
                "hard": round(hard, 2),
                "soft": round(soft, 2),
                "missing": missing,
                "feedback": feedback_text
            }
            results.append(eval_record)

            # Save to DB
            try:
                save_evaluation(
                    resume_name=f.name,
                    eval_data={
                        "jd_title": (jd_text.splitlines()[0] if jd_text else "JD"),
                        "relevance_score": round(final, 2),
                        "semantic_score": round(soft, 2),
                        "hard_score": round(hard, 2),
                        "verdict": verdict,
                        "missing_skills": missing
                    }
                )
            except Exception as e:
                st.warning(f"Failed to save evaluation for {f.name}: {e}")

            progress.progress(int((idx / total) * 100))
            status_placeholder.success(f"Processed {f.name}: score {final:.1f} — {verdict}")

        # --- Results ---
        st.markdown("## Results (This Run)")
        df = pd.DataFrame(results)
        if df.empty:
            st.info("No successful results.")
        else:
            colf1, colf2, colf3 = st.columns([1,1,2])
            with colf1:
                min_score = st.slider("Minimum score", min_value=0, max_value=100, value=0, key="min_score_run")
            with colf2:
                verdict_filter = st.selectbox("Verdict", options=["All", "High", "Medium", "Low"], index=0, key="verdict_run")
            with colf3:
                text_filter = st.text_input("Filename contains", value="", key="text_filter_run")

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


# --------------- STORED RESULTS TAB ---------------
with tab_db:
    st.markdown("## Stored Evaluations")
    try:
        records = get_all_evaluations()
    except Exception as e:
        st.error(f"Failed to fetch stored evaluations: {e}")
        records = []

    if not records:
        st.info("No evaluations stored yet.")
    else:
        df_db = pd.DataFrame(
            [
                {
                    "Resume": r.resume_name,
                    "JD Title": r.jd_title,
                    "Score": r.relevance_score,
                    "Hard Score": r.hard_score,
                    "Soft Score": r.semantic_score,
                    "Verdict": r.verdict,
                    "Missing Skills": r.missing_skills,
                    "Saved ID": r.id
                }
                for r in records
            ]
        )

        col1, col2 = st.columns([1,1])
        with col1:
            role_filter = st.text_input("🔍 Filter by Job Title", key="role_filter_db")
        with col2:
            verdict_filter_db = st.selectbox("Filter by Verdict", ["All", "High", "Medium", "Low"], key="verdict_db")

        df_filtered = df_db.copy()
        if role_filter.strip():
            df_filtered = df_filtered[df_filtered["JD Title"].str.contains(role_filter, case=False, na=False)]
        if verdict_filter_db != "All":
            df_filtered = df_filtered[df_filtered["Verdict"] == verdict_filter_db]

        st.dataframe(df_filtered, use_container_width=True)

        csv_db = df_filtered.to_csv(index=False)
        st.download_button("Download DB results (CSV)", csv_db, file_name="stored_evaluations.csv", mime="text/csv")
