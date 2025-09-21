# dashboard/streamlit_app.py
"""
Streamlit app that:
 - Uploads a Job Description (PDF/DOCX/TXT)
 - Forces parsing via Groq LLM by sending model_url + api_key to backend /jd/upload
 - Uploads resumes, then calls backend /evaluate (and passes groq creds so evaluator uses Groq)
 - Shows parsed JD JSON, evaluation results, missing skills, feedback, and raw groq output on failure

Notes:
 - Make sure your backend (FastAPI app) is running and its /jd/upload and /evaluate endpoints accept
   model_url and api_key form fields (this app assumes that).
 - Use Streamlit sidebar to enter Groq URL and API key (temporary for testing).
"""

import os
import tempfile
import json
import requests
import traceback
import streamlit as st

st.set_page_config(page_title="Resume Relevance — Groq JD parsing", layout="wide")
st.title("Resume Relevance — Groq JD parsing (Llama-38B)")

# -------------------------
# Configurable endpoints
# -------------------------
DEFAULT_BACKEND = os.getenv("API_BASE", "http://localhost:8000")
API_BASE = st.sidebar.text_input("Backend API base URL", value=DEFAULT_BACKEND)
st.sidebar.markdown("**Groq (LLM) settings** — for parsing & evaluation")

groq_url = st.sidebar.text_input("Groq API URL (full endpoint)", value=os.getenv("GROQ_API_URL", ""))
groq_key = st.sidebar.text_input("Groq API Key", value=os.getenv("GROQ_API_KEY", ""), type="password")
use_groq_checkbox = st.sidebar.checkbox("Use Groq for JD parsing & evaluation (send creds to backend)", value=True)

st.sidebar.markdown("---")
st.sidebar.write("Tips:")
st.sidebar.write("- If Groq returns non-JSON, the backend will fallback to heuristic and return `parse_error` with raw response.")
st.sidebar.write("- For production, set env vars in server instead of sending keys from UI.")

# -------------------------
# Helpers
# -------------------------
def show_exception(e):
    st.error("Error: " + str(e))
    st.text(traceback.format_exc())

def preview_text_from_bytes(b: bytes, fname: str) -> str:
    """
    Try to provide a text preview from uploaded bytes.
    For PDFs this will just show a byte-decoded preview (best-effort).
    The authoritative extract should happen on backend using PyMuPDF/pdfplumber/docx2txt.
    """
    try:
        return b.decode("utf-8", errors="ignore")[:4000]
    except Exception:
        return f"<binary content: {fname} — preview not available>"

def upload_jd_to_backend(file_bytes: bytes, filename: str, title: str = "", use_groq: bool = True, model_url: str = None, api_key: str = None, backend_url: str = API_BASE):
    """
    Upload JD file to backend /jd/upload. This passes use_groq + model_url + api_key form fields.
    Returns JSON response or raises.
    """
    files = {"file": (filename, file_bytes)}
    data = {"use_groq": str(bool(use_groq)).lower()}
    if title:
        data["title"] = title
    if model_url:
        data["model_url"] = model_url
    if api_key:
        data["api_key"] = api_key
    resp = requests.post(f"{backend_url.rstrip('/')}/jd/upload", files=files, data=data, timeout=120)
    resp.raise_for_status()
    return resp.json()

def upload_resume_to_backend(file_bytes: bytes, filename: str, backend_url: str = API_BASE):
    files = {"file": (filename, file_bytes)}
    resp = requests.post(f"{backend_url.rstrip('/')}/resume/upload", files=files, timeout=120)
    resp.raise_for_status()
    return resp.json()

def evaluate_resume_backend(job_id: int, resume_id: int, hard_weight: float, soft_weight: float, model_url: str = None, api_key: str = None, backend_url: str = API_BASE):
    data = {
        "job_id": str(job_id),
        "resume_id": str(resume_id),
        "hard_weight": str(hard_weight),
        "soft_weight": str(soft_weight),
    }
    if model_url:
        data["model_url"] = model_url
    if api_key:
        data["api_key"] = api_key
    resp = requests.post(f"{backend_url.rstrip('/')}/evaluate", data=data, timeout=180)
    resp.raise_for_status()
    return resp.json()

# Optional: direct groq test from Streamlit (useful for debugging groq endpoint)
def groq_direct_test(prompt: str, model_url: str, api_key: str, timeout: int = 30):
    """
    Send prompt directly to Groq endpoint from Streamlit for quick testing.
    Tries common payload shapes (input, prompt, messages). Returns raw text response.
    """
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "Accept": "application/json"}
    bodies = [
        {"input": prompt, "max_output_tokens": 1024, "temperature": 0.0},
        {"prompt": prompt, "max_tokens": 1024, "temperature": 0.0},
        {"messages": [{"role": "user", "content": prompt}], "max_tokens": 1024, "temperature": 0.0}
    ]
    last_err = None
    for body in bodies:
        try:
            r = requests.post(model_url, headers=headers, json=body, timeout=timeout)
            r.raise_for_status()
            return r.text
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Direct Groq request failed: {last_err}")

# -------------------------
# UI: JD upload & parse
# -------------------------
st.header("1) Job Description (JD) — upload & parse (Groq)")

col1, col2 = st.columns([3, 1])
with col1:
    jd_file = st.file_uploader("Upload JD (pdf / docx / txt)", type=["pdf", "docx", "txt"])
    jd_title = st.text_input("Optional job title (sent to backend)", value="")
with col2:
    st.write("Groq debug")
    if st.button("Test Groq endpoint (quick)"):
        if not groq_url or not groq_key:
            st.warning("Please provide Groq URL & API key in the sidebar first.")
        else:
            with st.spinner("Calling Groq directly..."):
                try:
                    test_prompt = "Return a JSON object: { 'hello': 'world' }"
                    resp_text = groq_direct_test(test_prompt, groq_url, groq_key)
                    st.success("Groq responded (truncated):")
                    st.code(resp_text[:2000])
                except Exception as e:
                    show_exception(e)

if jd_file:
    jd_bytes = jd_file.read()
    st.subheader("Preview (first 1200 chars)")
    st.code(preview_text_from_bytes(jd_bytes, jd_file.name)[:1200])
    st.write("")
    parse_col, fallback_col = st.columns(2)
    with parse_col:
        if st.button("Parse JD (Groq)"):
            if not use_groq_checkbox:
                st.warning("Groq usage is disabled in the UI. Toggle 'Use Groq' in the sidebar if you want to use Groq.")
            else:
                st.info("Uploading JD to backend and requesting Groq-based parse...")
                try:
                    out = upload_jd_to_backend(jd_bytes, jd_file.name, title=jd_title, use_groq=True, model_url=groq_url or None, api_key=groq_key or None, backend_url=API_BASE)
                    st.success("JD uploaded and parsed — backend response:")
                    st.json(out.get("parsed", out))
                    # store job_id for later evaluation
                    job_id = out.get("job_id")
                    if job_id:
                        st.session_state["job_id"] = job_id
                        st.info(f"Saved job_id {job_id} to session.")
                except Exception as e:
                    st.error("JD parse/upload failed. Backend responded with an error — falling back to local heuristic parse (display only).")
                    show_exception(e)
                    # optional: call local heuristic parse (if available in same env)
                    try:
                        from backend.services.jd_parser import parse_jd as local_parse
                        parsed_local = local_parse(jd_bytes.decode("utf-8", errors="ignore"), use_groq=False)
                        st.subheader("Local heuristic parse result (fallback)")
                        st.json(parsed_local)
                    except Exception:
                        st.warning("Local heuristic parser not available in this Streamlit environment.")
    with fallback_col:
        if st.button("Parse JD (Local heuristic)"):
            try:
                # Try to call local heuristic if backend.services present (works when streamlit runs in same env)
                from backend.services.jd_parser import parse_jd as local_parse
                parsed_local = local_parse(preview_text_from_bytes(jd_bytes, jd_file.name), use_groq=False)
                st.json(parsed_local)
            except Exception as e:
                st.warning("Local parse not available here. Upload JD to backend instead.")
                show_exception(e)

# -------------------------
# UI: Resumes upload & evaluate
# -------------------------
st.header("2) Upload resumes and evaluate against uploaded JD")

resume_files = st.file_uploader("Upload resumes (multiple). After upload, click Evaluate.", accept_multiple_files=True, type=["pdf", "docx", "txt"])
hard_weight = st.slider("Hard match weight", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
soft_weight = round(1.0 - hard_weight, 2)
st.write(f"Soft weight = {soft_weight}")

if st.button("Upload resumes & evaluate (backend)"):
    if "job_id" not in st.session_state:
        st.error("No job_id in session. Upload & parse a JD first (Parse JD (Groq)).")
    else:
        job_id = int(st.session_state["job_id"])
        results = []
        for f in resume_files:
            try:
                st.info(f"Uploading {f.name} ...")
                b = f.read()
                r_up = upload_resume_to_backend(b, f.name, backend_url=API_BASE)
                resume_id = r_up.get("resume_id")
                st.success(f"Uploaded {f.name} (resume_id={resume_id}). Evaluating...")
                # call evaluate - pass groq creds to ensure evaluator uses Groq
                eval_out = evaluate_resume_backend(job_id, resume_id, hard_weight, soft_weight, model_url=groq_url or None, api_key=groq_key or None, backend_url=API_BASE)
                results.append({"filename": f.name, "result": eval_out.get("result", eval_out)})
            except Exception as e:
                st.error(f"Failed {f.name}: {e}")
                st.text(traceback.format_exc())

        if results:
            # sort by final score
            results = sorted(results, key=lambda x: x["result"].get("final", 0), reverse=True)
            st.header("Evaluation results (sorted by final score)")
            for r in results:
                res = r["result"]
                st.subheader(f"{r['filename']} — Score: {res.get('final', 'N/A')}  Verdict: {res.get('verdict','N/A')}")
                st.write("Hard:", res.get("hard"), "Soft:", res.get("soft"))
                missing = res.get("missing", {})
                st.write("Missing (must):", missing.get("must", []))
                st.write("Missing (good):", missing.get("good", []))
                fb = res.get("feedback", [])
                if fb:
                    st.write("Feedback:")
                    for item in fb[:6]:
                        st.markdown(f"- {item}")
                # show raw groq response if present (debugging)
                if isinstance(res.get("raw_groq_response"), str):
                    with st.expander("Raw Groq response (debug)"):
                        st.code(res.get("raw_groq_response")[:4000])

# -------------------------
# UI: View results stored in backend
# -------------------------
st.header("3) View stored evaluations (backend)")

job_id_input = st.number_input("Job ID to fetch results", min_value=0, step=1, value=int(st.session_state.get("job_id", 0)))
if st.button("Fetch results for job"):
    try:
        r = requests.get(f"{API_BASE.rstrip('/')}/results/{int(job_id_input)}", timeout=60)
        r.raise_for_status()
        out = r.json()
        st.json(out)
    except Exception as e:
        show_exception(e)

# -------------------------
# Footer / debug helpers
# -------------------------
st.markdown("---")
st.markdown("**Debug helpers**")
if st.checkbox("Show environment vars (debug)"):
    env_preview = {
        "API_BASE": API_BASE,
        "GROQ_API_URL (sidebar)": groq_url,
        "GROQ_API_KEY (sidebar set?)": bool(groq_key),
        "Use Groq (checkbox)": use_groq_checkbox
    }
    st.json(env_preview)

st.caption("If Groq parsing fails return contains parse_error/groq_raw or groq_exception keys. Copy raw to adjust groq_client payload shapes in backend.")
