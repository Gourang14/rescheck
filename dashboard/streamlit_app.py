# dashboard/streamlit_app.py
"""
Streamlit app: JD upload + resume upload + automatic LLM endpoint discovery.
You only provide an API key (and optionally choose provider). The app will:
 - try known Groq endpoints automatically (no URL input required)
 - optionally use OpenAI if chosen or key looks like `sk-...`
 - fallback to local heuristic parser/scorer if remote calls fail

Notes:
 - This runs entirely client-side (Streamlit process). Do not paste production keys on shared machines.
 - If you're behind a corporate proxy, set HTTP(S)_PROXY env vars before launching Streamlit.
"""

import os, re, json, tempfile, traceback
from typing import Optional, Dict, Any, List
import requests
import streamlit as st

# Optional: better file parsing
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

st.set_page_config(layout="wide", page_title="Resume Relevance — Auto Endpoint (Groq/OpenAI)")
st.title("Resume Relevance — Auto Endpoint (Groq/OpenAI)")

# -------------------------
# Candidate Groq endpoints (we'll try these automatically)
# -------------------------
GROQ_CANDIDATE_URLS = [
    "https://api.groq.ai/v1/models/llama-38b/generate",
    "https://api.groq.ai/v1/engines/llama-38b/completions",
    "https://api.groq.ai/v1/models/llama-3-8b/generate",
    # vendor partner endpoints could be added here if you have them
]

# -------------------------
# Helpers: file extraction
# -------------------------
def extract_text_from_bytes(b: bytes, filename: str) -> str:
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return _extract_pdf_bytes(b)
    if lower.endswith(".docx"):
        return _extract_docx_bytes(b)
    try:
        return b.decode("utf-8", errors="ignore")
    except:
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
    try:
        return b.decode("utf-8", errors="ignore")
    except:
        return ""

def _extract_docx_bytes(b: bytes) -> str:
    if _HAS_DOCX2TXT:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".docx")
        tmp.write(b); tmp.close()
        try:
            return docx2txt.process(tmp.name) or ""
        finally:
            try: os.unlink(tmp.name)
            except: pass
    try:
        return b.decode("utf-8", errors="ignore")
    except:
        return ""

# -------------------------
# JSON extraction helper
# -------------------------
def _try_extract_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except Exception:
        pass
    # try find first balanced {...}
    try:
        start = text.find("{")
        if start != -1:
            depth = 0
            for i in range(start, len(text)):
                ch = text[i]
                if ch == "{": depth += 1
                elif ch == "}": depth -= 1
                if depth == 0:
                    cand = text[start:i+1]
                    return json.loads(cand)
    except Exception:
        pass
    # regex fallback
    try:
        m = re.search(r'(\{(?:[^{}]|\n|\r)*\})', text)
        if m: return json.loads(m.group(1))
    except Exception:
        pass
    return None

# -------------------------
# Callers: try Groq candidate endpoints then OpenAI fallback
# -------------------------
DEFAULT_TIMEOUT = 40

def try_groq_auto(api_key: str, prompt: str, max_tokens: int = 1024, temperature: float = 0.0) -> Dict[str, Any]:
    """
    Try candidate Groq endpoints in sequence and return dict:
      { success: bool, url: str|None, raw: str|None, json: dict|None, diagnostics: {...} }
    """
    diagnostics = {"attempts": []}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "Accept": "application/json"}
    bodies = [
        {"input": prompt, "max_output_tokens": max_tokens, "temperature": temperature},
        {"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature},
        {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": temperature}
    ]
    for url in GROQ_CANDIDATE_URLS:
        for i, body in enumerate(bodies, start=1):
            attempt = {"url": url, "body_shape": i}
            try:
                r = requests.post(url, headers=headers, json=body, timeout=DEFAULT_TIMEOUT)
                attempt["status"] = r.status_code
                attempt["ok"] = r.ok
                attempt["truncated_text"] = r.text[:1500]
                diagnostics["attempts"].append(attempt)
                if r.ok:
                    parsed = _try_extract_json(r.text)
                    return {"success": True, "url": url, "raw": r.text, "json": parsed, "diagnostics": diagnostics}
            except Exception as e:
                attempt["exception"] = repr(e)
                diagnostics["attempts"].append(attempt)
                continue
    return {"success": False, "url": None, "raw": None, "json": None, "diagnostics": diagnostics}

def try_openai_direct(api_key: str, prompt: str, model: str = "gpt-4o-mini", max_tokens: int = 1024, temperature: float = 0.0) -> Dict[str, Any]:
    """
    Try OpenAI via the openai REST-compatible endpoint using requests if the openai package is not present.
    This function will attempt the canonical chat completions response shape.
    """
    # first try official OpenAI REST endpoint
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": temperature}
    try:
        r = requests.post(url, headers=headers, json=body, timeout=DEFAULT_TIMEOUT)
        if r.ok:
            text = r.text
            # try parse common shapes: choices[0].message.content or direct text
            try:
                js = r.json()
                content = None
                if "choices" in js and len(js["choices"])>0:
                    ch = js["choices"][0]
                    if isinstance(ch.get("message"), dict):
                        content = ch["message"].get("content")
                    elif "text" in ch:
                        content = ch.get("text")
                if content is None:
                    # fallback to raw text
                    content = json.dumps(js)
            except Exception:
                content = r.text
            parsed = _try_extract_json(content) or _try_extract_json(r.text)
            return {"success": True, "raw": r.text, "json": parsed, "diagnostics": {"url": url, "status": r.status_code}}
        else:
            return {"success": False, "raw": r.text, "json": None, "diagnostics": {"url": url, "status": r.status_code}}
    except Exception as e:
        return {"success": False, "raw": None, "json": None, "diagnostics": {"exception": repr(e)}}

# -------------------------
# Prompts (same as earlier)
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
# Simple heuristic fallback (same as earlier)
# -------------------------
def heuristic_parse_jd(text: str) -> Dict[str, Any]:
    res = {"title": "", "must_have": [], "good_to_have": [], "qualifications": [], "experience": ""}
    for line in text.splitlines():
        if line.strip():
            res["title"] = line.strip()[:200]; break
    must = re.findall(r'(?:must[-\s]*have|required)[:\-\s]*([^\n;]+)', text, flags=re.I)
    good = re.findall(r'(?:nice[-\s]*to[-\s]*have|good[-\s]*to[-\s]*have|preferred)[:\-\s]*([^\n;]+)', text, flags=re.I)
    quals = re.findall(r'(?:qualification|education|degree|certification)[:\-\s]*([^\n;]+)', text, flags=re.I)
    def split_and_clean(s): return [p.strip() for p in re.split(r'[,\;/\|]', s) if p.strip()]
    for m in must: res["must_have"].extend(split_and_clean(m))
    for g in good: res["good_to_have"].extend(split_and_clean(g))
    for q in quals: res["qualifications"].extend(split_and_clean(q))
    exp = re.search(r'(\d+\+?\s+years?|\d+-\d+\s+years)', text, flags=re.I)
    if exp: res["experience"] = exp.group(0)
    if not res["must_have"]:
        tokens = re.findall(r'\b[A-Za-z\+\#]{2,}(?:\s+[A-Za-z\+\#]{2,}){0,2}\b', text)
        noise = {"our","we","the","and","or","to","in","with","from","that","this","apply","role","job"}
        out=[]; seen=set()
        for t in tokens:
            tl = t.lower()
            if tl in noise or len(t) <= 2: continue
            if tl in seen: continue
            seen.add(tl); out.append(t)
            if len(out)>=12: break
        res["must_have"].extend(out)
    return res

def heuristic_score(resume_text: str, jd_parsed: Dict[str, Any], hard_weight=0.6, soft_weight=0.4) -> Dict[str, Any]:
    must = jd_parsed.get("must_have", [])
    good = jd_parsed.get("good_to_have", [])
    found_must=[]; found_good=[]
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
    hard = round((0.75*must_cov + 0.25*good_cov)*100,2)
    soft = round(hard * 0.9,2)
    final = round(hard_weight*hard + soft_weight*soft,2)
    verdict = "High" if final >= 75 else "Medium" if final >= 50 else "Low"
    missing = {"must":[m for m in must if m not in found_must], "good":[g for g in good if g not in found_good]}
    feedback = [f"Add a short project/cert demonstrating: {m}" for m in missing["must"][:4]]
    return {"hard": hard, "soft": soft, "final": final, "verdict": verdict, "missing": missing, "feedback": feedback}

# -------------------------
# UI
# -------------------------
st.sidebar.header("API Key / Provider")
api_key = st.sidebar.text_input("API Key (required for remote parsing)", type="password")
provider = st.sidebar.selectbox("Provider (auto tries Groq first)", ["Auto (Groq then OpenAI)", "Groq only", "OpenAI only"])
st.sidebar.markdown("If you don't have an API key, you can use the local heuristic parser.")

st.header("1) Job Description (JD) upload or paste")
jd_file = st.file_uploader("Upload JD (PDF/DOCX/TXT) or paste below", type=["pdf","docx","txt"])
jd_text = st.text_area("Or paste JD text here", height=220)

if jd_file:
    try:
        jd_text = extract_text_from_bytes(jd_file.read(), jd_file.name)
    except Exception as e:
        st.error("Failed to read JD file: " + str(e))

if not jd_text.strip():
    st.info("Upload or paste a JD to proceed (or use heuristic parser).")

col1, col2 = st.columns([2,1])
with col1:
    parse_btn = st.button("Parse JD (auto)")
with col2:
    clear_btn = st.button("Clear parsed JD")

if clear_btn:
    st.session_state.pop("parsed_jd", None)
    st.success("Cleared parsed JD.")

if parse_btn:
    if not api_key and (provider != "OpenAI only"):
        st.warning("No API key provided — using local heuristic parser.")
        parsed = heuristic_parse_jd(jd_text)
        st.session_state["parsed_jd"] = parsed
        st.json(parsed)
    else:
        st.info("Attempting remote parse (auto-discovery). This will try candidate endpoints automatically.")
        parsed = None
        diagnostics = {}
        # Try flows depending on provider selection
        if provider in ("Auto (Groq then OpenAI)", "Groq only", "Auto (Groq then OpenAI)"):
            g = try_groq_auto(api_key, JD_PROMPT_TEMPLATE.format(jd=jd_text), max_tokens=800, temperature=0.0)
            diagnostics["groq_try"] = g["diagnostics"]
            if g["success"]:
                parsed = g["json"] or heuristic_parse_jd(jd_text)
                parsed["_remote_url"] = g["url"]
                parsed["_remote_raw_trunc"] = (g["raw"][:1500] if g.get("raw") else None)
        if parsed is None and provider in ("Auto (Groq then OpenAI)","OpenAI only"):
            o = try_openai_direct(api_key, JD_PROMPT_TEMPLATE.format(jd=jd_text), model="gpt-4o-mini", max_tokens=800, temperature=0.0)
            diagnostics["openai_try"] = o["diagnostics"]
            if o["success"]:
                parsed = o["json"] or heuristic_parse_jd(jd_text)
                # store raw
                parsed["_remote_url"] = o["diagnostics"].get("url")
                parsed["_remote_raw_trunc"] = (o.get("raw")[:1500] if o.get("raw") else None)

        if parsed:
            st.success("Parsed JD (remote or fallback):")
            st.json(parsed)
            st.session_state["parsed_jd"] = parsed
            if diagnostics:
                with st.expander("Diagnostics (which endpoints tried)"):
                    st.json(diagnostics)
        else:
            st.error("Remote parsing failed for all attempted endpoints — using local heuristic fallback.")
            parsed = heuristic_parse_jd(jd_text)
            parsed["parse_fallback_reason"] = "remote_failed"
            st.json(parsed)
            st.session_state["parsed_jd"] = parsed

# -------------------------
# Resumes upload & evaluate
# -------------------------
st.header("2) Upload resumes and evaluate (multiple)")
resume_files = st.file_uploader("Upload resumes (PDF/DOCX/TXT). Select multiple", accept_multiple_files=True, type=["pdf","docx","txt"])
hard_weight = st.slider("Hard match weight", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
soft_weight = round(1.0 - hard_weight, 2)

if st.button("Evaluate resumes"):
    if "parsed_jd" not in st.session_state:
        st.error("No parsed JD found. Parse a JD first.")
    elif not resume_files:
        st.error("Upload at least one resume.")
    else:
        jd_parsed = st.session_state["parsed_jd"]
        results = []
        for f in resume_files:
            try:
                txt = extract_text_from_bytes(f.read(), f.name)
                # Try remote eval if api_key present
                eval_result = None
                diag_eval = {}
                if api_key and provider != "Groq only":  # allow auto/openai flows
                    # prefer groq auto
                    g = try_groq_auto(api_key, EVAL_PROMPT_TEMPLATE.format(jd_json=json.dumps(jd_parsed, ensure_ascii=False), resume_excerpt=txt[:7000], hard_weight=hard_weight, soft_weight=soft_weight), max_tokens=1000, temperature=0.0)
                    diag_eval["groq_try"] = g["diagnostics"]
                    if g["success"] and g.get("json"):
                        eval_result = g["json"]
                        eval_result["_remote_url"] = g["url"]
                    else:
                        o = try_openai_direct(api_key, EVAL_PROMPT_TEMPLATE.format(jd_json=json.dumps(jd_parsed, ensure_ascii=False), resume_excerpt=txt[:7000], hard_weight=hard_weight, soft_weight=soft_weight), model="gpt-4o-mini", max_tokens=1000, temperature=0.0)
                        diag_eval["openai_try"] = o["diagnostics"]
                        if o["success"] and o.get("json"):
                            eval_result = o["json"]
                            eval_result["_remote_url"] = o["diagnostics"].get("url")
                # If remote not used or failed -> heuristic
                if not eval_result:
                    eval_result = heuristic_score(txt, jd_parsed, hard_weight, soft_weight)
                    eval_result["_eval_fallback"] = True
                results.append({"file": f.name, "result": eval_result, "diag": diag_eval})
            except Exception as e:
                st.error(f"Failed for {f.name}: {e}")
                st.text(traceback.format_exc())
        # show results sorted by final score
        results = sorted(results, key=lambda x: x["result"].get("final", 0), reverse=True)
        for r in results:
            res = r["result"]
            st.subheader(f"{r['file']} — Final: {res.get('final','N/A')}  Verdict: {res.get('verdict','N/A')}")
            st.write("Hard:", res.get("hard"), "Soft:", res.get("soft"))
            missing = res.get("missing", {})
            st.write("Missing (must):", missing.get("must", []))
            st.write("Missing (good):", missing.get("good", []))
            fb = res.get("feedback", [])
            if fb:
                st.write("Feedback:")
                for item in fb[:6]:
                    st.markdown(f"- {item}")
            if r["diag"]:
                with st.expander("Diagnostics for this resume (endpoints tried)"):
                    st.json(r["diag"])

st.markdown("---")
st.caption("This app auto-discovers endpoints (Groq candidate URLs) — you only provide an API key. If remote calls fail, a local heuristic is used.")
