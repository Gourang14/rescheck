# dashboard/streamlit_app.py
"""
Streamlit app: JD upload + resume upload + Groq-based parsing & evaluation (client-side).
No backend required — Streamlit calls the Groq endpoint directly when API URL + Key are provided.

Supports: PDF (PyMuPDF if installed), DOCX (docx2txt if installed), TXT.
Fallback: simple heuristic parser & scorer when Groq not provided or when Groq returns non-JSON.
"""

import os
import re
import json
import tempfile
import traceback
from typing import Optional, Dict, Any, List

import requests
import streamlit as st

# Optional helpers for better file parsing (install if you want)
try:
    import fitz  # PyMuPDF
    _HAS_FITZ = True
except Exception:
    _HAS_FITZ = False

try:
    import docx2txt
    _HAS_DOCX2TXT = True
except Exception:
    _HAS_DOCX2TXT = False

try:
    from rapidfuzz import fuzz
    _HAS_RAPIDFUZZ = True
except Exception:
    _HAS_RAPIDFUZZ = False

# -------------------------
# Utility: extract text from uploaded bytes
# -------------------------
def extract_text_from_bytes(b: bytes, filename: str) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        return _extract_pdf_bytes(b)
    if name.endswith(".docx"):
        return _extract_docx_bytes(b)
    # fallback to plain text decode
    try:
        return b.decode("utf-8", errors="ignore")
    except Exception:
        return ""

def _extract_pdf_bytes(b: bytes) -> str:
    if _HAS_FITZ:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.write(b); tmp.close()
        try:
            doc = fitz.open(tmp.name)
            pages = [p.get_text() for p in doc]
            doc.close()
            return "\n".join(pages)
        finally:
            try: os.unlink(tmp.name)
            except: pass
    # fallback: try to decode bytes
    try:
        return b.decode("utf-8", errors="ignore")
    except:
        return ""

def _extract_docx_bytes(b: bytes) -> str:
    if _HAS_DOCX2TXT:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".docx")
        tmp.write(b); tmp.close()
        try:
            txt = docx2txt.process(tmp.name) or ""
            return txt
        finally:
            try: os.unlink(tmp.name)
            except: pass
    try:
        return b.decode("utf-8", errors="ignore")
    except:
        return ""

# -------------------------
# Groq client (tries multiple payload shapes)
# -------------------------
DEFAULT_TIMEOUT = 60

def call_groq_raw(prompt: str, model_url: str, api_key: str, max_tokens: int = 1024, temperature: float = 0.0, timeout: int = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    """
    Returns dict: { success: bool, raw_text: str|None, parsed_json: dict|None, diagnostics: {...} }
    """
    diagnostics = {"model_url": model_url}
    if not model_url or not api_key:
        return {"success": False, "raw_text": None, "parsed_json": None, "diagnostics": {"error": "missing_url_or_key"}}

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "Accept": "application/json"}
    bodies = [
        {"input": prompt, "max_output_tokens": max_tokens, "temperature": temperature},
        {"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature},
        {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": temperature}
    ]

    last_err = None
    for i, body in enumerate(bodies, start=1):
        diagnostics[f"attempt_{i}_body"] = body
        try:
            resp = requests.post(model_url, headers=headers, json=body, timeout=timeout)
            diagnostics[f"attempt_{i}_status_code"] = resp.status_code
            diagnostics[f"attempt_{i}_headers"] = dict(resp.headers)
            text = resp.text
            diagnostics[f"attempt_{i}_raw_response_truncated"] = text[:2000]
            if resp.ok:
                # try parse JSON from text
                parsed = _try_extract_json(text)
                return {"success": True, "raw_text": text, "parsed_json": parsed, "diagnostics": diagnostics}
            else:
                last_err = f"HTTP {resp.status_code}"
                # continue to try other shapes
        except Exception as e:
            last_err = repr(e)
            diagnostics[f"attempt_{i}_exception"] = repr(e)
            continue

    diagnostics["last_error"] = last_err
    return {"success": False, "raw_text": None, "parsed_json": None, "diagnostics": diagnostics}

def _try_extract_json(text: str) -> Optional[Dict[str, Any]]:
    # direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # find first balanced {...}
    try:
        start = text.find("{")
        if start != -1:
            depth = 0
            for i in range(start, len(text)):
                ch = text[i]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        cand = text[start:i+1]
                        return json.loads(cand)
    except Exception:
        pass
    # regex fallback
    try:
        m = re.search(r'(\{(?:[^{}]|\n|\r)*\})', text)
        if m:
            return json.loads(m.group(1))
    except Exception:
        pass
    return None

# -------------------------
# Prompts
# -------------------------
JD_PROMPT_TEMPLATE = """
You are a strict JSON extractor for job descriptions.
Given this job description below, return a VALID JSON object with keys:
- title (string)
- must_have (array of short strings)
- good_to_have (array of short strings)
- qualifications (array of short strings)
- experience (string)

Keep skill names short (1-4 words). Do NOT add extra commentary. Return only JSON.

Job description:
\"\"\"{jd}\"\"\"
"""

EVAL_PROMPT_TEMPLATE = """
You are an objective resume evaluator. Inputs:
JD_PARSED_JSON:
{jd_json}

RESUME_TEXT (first 7000 chars):
\"\"\"{resume_excerpt}\"\"\"

Return ONLY valid JSON with keys:
- hard (number 0-100)
- soft (number 0-100)
- final (number 0-100)
- verdict (one of "High", "Medium", "Low")
- missing (object with keys: must (list), good (list))
- feedback (array of 3-6 short actionable strings)

final should be computed as: final = round(hard_weight*hard + soft_weight*soft, 2)
Use the provided hard_weight and soft_weight values.

hard_weight: {hard_weight}
soft_weight: {soft_weight}
"""

# -------------------------
# Heuristic fallback parser & scorer
# -------------------------
def heuristic_parse_jd(text: str) -> Dict[str, Any]:
    # naive extraction: find inline "must have" / "nice to have" lines else token extraction
    res = {"title": "", "must_have": [], "good_to_have": [], "qualifications": [], "experience": ""}
    for line in text.splitlines():
        if line.strip():
            res["title"] = line.strip()[:200]
            break
    must = re.findall(r'(?:must[-\s]*have|required)[:\-\s]*([^\n;]+)', text, flags=re.I)
    good = re.findall(r'(?:nice[-\s]*to[-\s]*have|good[-\s]*to[-\s]*have|preferred)[:\-\s]*([^\n;]+)', text, flags=re.I)
    quals = re.findall(r'(?:qualification|education|degree|certification)[:\-\s]*([^\n;]+)', text, flags=re.I)
    def split_and_clean(s):
        parts = re.split(r'[,\;/\|]', s)
        return [p.strip() for p in parts if p.strip()]
    for m in must: res["must_have"].extend(split_and_clean(m))
    for g in good: res["good_to_have"].extend(split_and_clean(g))
    for q in quals: res["qualifications"].extend(split_and_clean(q))
    exp = re.search(r'(\d+\+?\s+years?|\d+-\d+\s+years)', text, flags=re.I)
    if exp: res["experience"] = exp.group(0)
    # fallback token extraction if empty
    if not res["must_have"]:
        tokens = re.findall(r'\b[A-Za-z\+\#]{2,}(?:\s+[A-Za-z\+\#]{2,}){0,2}\b', text)
        noise = {"our","we","the","and","or","to","in","with","from","that","this","apply","role","job"}
        seen=set()
        for t in tokens:
            tl = t.lower()
            if tl in noise or len(t) <= 2: continue
            if tl in seen: continue
            seen.add(tl); res["must_have"].append(t)
            if len(res["must_have"])>=12: break
    return res

def heuristic_score(resume_text: str, jd_parsed: Dict[str, Any], hard_weight=0.6, soft_weight=0.4) -> Dict[str, Any]:
    must = jd_parsed.get("must_have", [])
    good = jd_parsed.get("good_to_have", [])
    found_must = []
    found_good = []
    if _HAS_RAPIDFUZZ:
        for s in must:
            if fuzz.partial_ratio(s.lower(), resume_text.lower()) >= 70: found_must.append(s)
        for s in good:
            if fuzz.partial_ratio(s.lower(), resume_text.lower()) >= 70: found_good.append(s)
    else:
        for s in must:
            if s.lower() in resume_text.lower(): found_must.append(s)
        for s in good:
            if s.lower() in resume_text.lower(): found_good.append(s)
    must_cov = (len(found_must)/len(must)) if must else 1.0
    good_cov = (len(found_good)/len(good)) if good else 1.0
    hard = round((0.75*must_cov + 0.25*good_cov)*100, 2)
    soft = round(hard * 0.9, 2)
    final = round(hard_weight*hard + soft_weight*soft, 2)
    verdict = "High" if final >= 75 else "Medium" if final >= 50 else "Low"
    missing = {"must": [m for m in must if m not in found_must], "good": [g for g in good if g not in found_good]}
    feedback = [f"Add a short project/cert demonstrating: {m}" for m in missing["must"][:4]]
    return {"hard": hard, "soft": soft, "final": final, "verdict": verdict, "missing": missing, "feedback": feedback}

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(layout="wide", page_title="Resume Relevance — Groq")
st.title("Resume Relevance — Groq-powered JD parsing & resume scoring")

# Sidebar: Groq settings
st.sidebar.header("Groq (LLM) settings")
groq_url = st.sidebar.text_input("Groq API URL (full endpoint)", value=os.getenv("GROQ_API_URL",""))
groq_key = st.sidebar.text_input("Groq API Key", value=os.getenv("GROQ_API_KEY",""), type="password")
use_groq = st.sidebar.checkbox("Use Groq for parse & evaluation", value=bool(groq_url and groq_key))

st.sidebar.markdown("---")
st.sidebar.markdown("Optional: Install PyMuPDF and docx2txt for better PDF/DOCX extraction.")

# Main: JD upload or paste
st.header("1) Job Description (JD)")
col1, col2 = st.columns([3,1])
with col1:
    jd_file = st.file_uploader("Upload JD (PDF / DOCX / TXT) OR paste JD text below", type=["pdf","docx","txt"])
    jd_text_area = st.text_area("Or paste JD text here", height=250)
with col2:
    st.write("Quick actions")
    if st.button("Clear session parsed JD"):
        st.session_state.pop("parsed_jd", None)
        st.success("Cleared parsed JD from session.")

jd_text = ""
if jd_file:
    try:
        b = jd_file.read()
        jd_text = extract_text_from_bytes(b, jd_file.name)
    except Exception as e:
        st.error("Failed to read uploaded JD: " + str(e))
        jd_text = ""
# prefer pasted text if present
if jd_text_area.strip():
    jd_text = jd_text_area

if not jd_text.strip():
    st.info("Upload or paste a Job Description to parse.")

# Parse actions
if jd_text.strip():
    st.subheader("JD preview")
    st.code(jd_text[:2000])

    parse_col, fallback_col = st.columns(2)
    with parse_col:
        if st.button("Parse JD with Groq"):
            if not (use_groq and groq_url and groq_key):
                st.warning("Groq not configured or disabled. Enable Groq in sidebar or provide URL + Key.")
            else:
                st.info("Calling Groq to parse JD (strict JSON). This may take a few seconds...")
                resp = call_groq_raw(JD_PROMPT_TEMPLATE.format(jd=jd_text), model_url=groq_url, api_key=groq_key, max_tokens=800, temperature=0.0)
                if resp["success"] and resp["parsed_json"]:
                    parsed = resp["parsed_json"]
                    st.success("Groq returned JSON for JD parsing.")
                    st.json(parsed)
                    st.session_state["parsed_jd"] = parsed
                else:
                    st.error("Groq parse failed — falling back to heuristic. See diagnostics below.")
                    st.json(resp["diagnostics"])
                    parsed = heuristic_parse_jd(jd_text)
                    parsed["parse_fallback_reason"] = resp["diagnostics"].get("last_error", "groq_failed")
                    st.subheader("Heuristic JD parse (fallback)")
                    st.json(parsed)
                    st.session_state["parsed_jd"] = parsed
    with fallback_col:
        if st.button("Parse JD (local heuristic)"):
            parsed = heuristic_parse_jd(jd_text)
            st.success("Parsed JD using local heuristic.")
            st.json(parsed)
            st.session_state["parsed_jd"] = parsed

# -------------------------
# Resumes upload and evaluation
# -------------------------
st.header("2) Upload resumes and evaluate (multiple)")
resume_files = st.file_uploader("Upload resumes (PDF/DOCX/TXT). Select multiple.", accept_multiple_files=True, type=["pdf","docx","txt"])

hard_weight = st.slider("Hard match weight", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
soft_weight = round(1.0 - hard_weight, 2)
st.write("Soft weight:", soft_weight)

if st.button("Evaluate resumes"):
    if "parsed_jd" not in st.session_state:
        st.error("No parsed JD found. Parse a JD first (step 1).")
    elif not resume_files:
        st.error("Upload at least one resume.")
    else:
        jd_parsed = st.session_state["parsed_jd"]
        results = []
        for f in resume_files:
            try:
                st.info(f"Processing {f.name} ...")
                rb = f.read()
                resume_text = extract_text_from_bytes(rb, f.name)
                # try Groq evaluation if configured
                if use_groq and groq_url and groq_key:
                    prompt = EVAL_PROMPT_TEMPLATE.format(jd_json=json.dumps(jd_parsed, ensure_ascii=False), resume_excerpt=resume_text[:7000], hard_weight=hard_weight, soft_weight=soft_weight)
                    resp = call_groq_raw(prompt, model_url=groq_url, api_key=groq_key, max_tokens=1000, temperature=0.0)
                    if resp["success"] and resp["parsed_json"]:
                        parsed_eval = resp["parsed_json"]
                        # normalize numeric fields if present as strings
                        try:
                            parsed_eval["hard"] = float(parsed_eval.get("hard", 0.0))
                            parsed_eval["soft"] = float(parsed_eval.get("soft", 0.0))
                            parsed_eval["final"] = float(parsed_eval.get("final", round(parsed_eval["hard"]*hard_weight + parsed_eval["soft"]*soft_weight,2)))
                        except Exception:
                            parsed_eval = heuristic_score(resume_text, jd_parsed, hard_weight, soft_weight)
                        results.append({"file": f.name, "result": parsed_eval, "raw_groq": resp["diagnostics"]})
                    else:
                        st.warning(f"Groq evaluation failed for {f.name}; using heuristic fallback. See diagnostics.")
                        st.json(resp["diagnostics"])
                        parsed_eval = heuristic_score(resume_text, jd_parsed, hard_weight, soft_weight)
                        parsed_eval["eval_fallback_reason"] = resp["diagnostics"].get("last_error", "groq_failed")
                        results.append({"file": f.name, "result": parsed_eval, "raw_groq": resp["diagnostics"]})
                else:
                    parsed_eval = heuristic_score(resume_text, jd_parsed, hard_weight, soft_weight)
                    results.append({"file": f.name, "result": parsed_eval, "raw_groq": None})
            except Exception as e:
                st.error(f"Failed to evaluate {f.name}: {e}")
                st.text(traceback.format_exc())

        # display results sorted by final score
        results = sorted(results, key=lambda x: x["result"].get("final", 0), reverse=True)
        st.header("Evaluation results")
        for r in results:
            res = r["result"]
            st.subheader(f"{r['file']}  — Final: {res.get('final','N/A')}  Verdict: {res.get('verdict','N/A')}")
            st.write("Hard:", res.get("hard"), "Soft:", res.get("soft"))
            missing = res.get("missing", {})
            st.write("Missing (must):", missing.get("must", []))
            st.write("Missing (good):", missing.get("good", []))
            fb = res.get("feedback", [])
            if fb:
                st.write("Feedback:")
                for item in fb[:6]:
                    st.markdown(f"- {item}")
            if r.get("raw_groq"):
                with st.expander("Groq diagnostics (raw)"):
                    st.json(r["raw_groq"])

st.markdown("---")
st.caption("Notes: Provide Groq API URL + Key in sidebar to use Groq. If Groq fails, a local heuristic is used as fallback.")

