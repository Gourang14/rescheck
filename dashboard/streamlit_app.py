# dashboard/streamlit_app.py
import os, tempfile, requests, json, traceback
import streamlit as st

API_BASE = st.secrets.get("API_BASE", os.getenv("API_BASE", "http://localhost:8000"))

st.set_page_config(page_title="Resume Relevance (Groq)", layout="wide")
st.title("Resume Relevance — Groq LLM (Llama-38B)")

st.sidebar.header("Groq / API settings")
groq_key = st.sidebar.text_input("Groq API Key", type="password")
groq_url = st.sidebar.text_input("Groq API URL (full endpoint)")
use_groq_checkbox = st.sidebar.checkbox("Use Groq for parsing & evaluation", value=True)
api_base_input = st.sidebar.text_input("Backend API (optional)", value=API_BASE)
if api_base_input:
    API_BASE = api_base_input

st.header("Step 1 — Upload Job Description (JD)")
jd_file = st.file_uploader("JD (PDF/DOCX/TXT)", type=["pdf","docx","txt"])
if jd_file:
    st.subheader("JD preview")
    b = jd_file.read()
    # write temporary file for local preview
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix="." + jd_file.name.split(".")[-1])
    tmp.write(b); tmp.close()
    try:
        # show first 800 chars
        try:
            from backend.services.parser import extract_text
            txt = extract_text(tmp.name)
        except Exception:
            try:
                txt = b.decode("utf-8", errors="ignore")
            except:
                txt = "<binary content>"
        st.code(txt[:1000])
    finally:
        pass

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Upload JD to backend (use Groq)"):
            # call backend /jd/upload
            files = {"file": (jd_file.name, b)}
            data = {"use_groq": str(bool(use_groq_checkbox)).lower()}
            if groq_url: data["model_url"] = groq_url
            if groq_key: data["api_key"] = groq_key
            if st.checkbox("Also send job title extracted (optional)", value=False):
                data["title"] = st.text_input("Title (optional)")
            try:
                r = requests.post(f"{API_BASE}/jd/upload", files=files, data=data, timeout=60)
                r.raise_for_status()
                out = r.json()
                st.success(f"JD uploaded — job_id {out['job_id']}")
                st.json(out["parsed"])
                st.session_state["job_id"] = out["job_id"]
            except Exception as e:
                st.error(f"JD upload failed: {e}")
                st.text(traceback.format_exc())

    with col2:
        if st.button("Parse JD locally (heuristic)"):
            try:
                # call local parse_jd if app runs in same env
                from backend.services.jd_parser import parse_jd
                parsed = parse_jd(txt, use_groq=False)
                st.json(parsed)
            except Exception as e:
                st.error("Local JD parsing failed: " + str(e))
                st.text(traceback.format_exc())

st.header("Step 2 — Upload Resumes")
resumes = st.file_uploader("Upload resumes (multiple)", accept_multiple_files=True, type=["pdf","docx","txt"])
hard_weight = st.slider("Hard match weight", 0.0, 1.0, 0.6, 0.05)
soft_weight = round(1.0 - hard_weight, 2)

if st.button("Upload resumes & evaluate (via backend)"):
    if "job_id" not in st.session_state:
        st.error("Upload JD first and get job_id.")
    else:
        job_id = st.session_state["job_id"]
        results = []
        for f in resumes:
            try:
                b = f.read()
                files = {"file": (f.name, b)}
                r = requests.post(f"{API_BASE}/resume/upload", files=files, timeout=60)
                r.raise_for_status()
                resume_id = r.json()["resume_id"]
                # evaluate
                data = {"job_id": job_id, "resume_id": resume_id, "hard_weight": hard_weight, "soft_weight": soft_weight}
                if use_groq_checkbox:
                    if groq_url: data["model_url"] = groq_url
                    if groq_key: data["api_key"] = groq_key
                r2 = requests.post(f"{API_BASE}/evaluate", data=data, timeout=120)
                r2.raise_for_status()
                out = r2.json()["result"]
                results.append({"filename": f.name, "result": out})
            except Exception as e:
                st.error(f"Failed {f.name}: {e}")
        # sort and display
        results = sorted(results, key=lambda x: x["result"].get("final", 0), reverse=True)
        for r in results:
            st.markdown(f"### {r['filename']} — Score: **{r['result'].get('final',0)}** | {r['result'].get('verdict')}")
            st.write("Hard:", r['result'].get("hard"), "Soft:", r['result'].get("soft"))
            missing = r['result'].get("missing", {})
            st.write("Missing must:", missing.get("must", []))
            st.write("Missing good:", missing.get("good", []))
            st.info("\n".join(r['result'].get("feedback", [])[:6]))
