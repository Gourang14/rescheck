# dashboard/streamlit_app.py
"""
Resume Relevance — Auto Groq usage (you only provide API key).
Behavior:
 - You paste API key (Groq or OpenAI) in the sidebar.
 - App auto-tries Groq candidate endpoints & request shapes.
 - If Groq fails, app will automatically try OpenAI (if key looks like sk- or user chooses).
 - If remote calls fail, app falls back to local heuristic parsing & scoring.
 - Upload JD (PDF/DOCX/TXT) and multiple resumes; get parsed JD, scores, verdict, and feedback.
"""

import os, re, json, tempfile, time, traceback
from typing import Optional, Dict, Any, List
import requests
import streamlit as st

# Optional libs for better extraction / fuzzy matching
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

# Candidate Groq endpoints the app will try automatically (no endpoint required from user)
GROQ_CANDIDATES = [
    "https://api.groq.ai/v1/models/llama-38b/generate",
    "https://api.groq.ai/v1/models/llama-3-8b/generate",
    "https://api.groq.ai/v1/engines/llama-38b/completions",
    # add other common shapes if you have provider-specific endpoints
]

DEFAULT_TIMEOUT = 40

# -------------------------
# Helpers: file extraction
# -------------------------
def extract_text_from_bytes(b: bytes, filename: str) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        return _extract_pdf_bytes(b)
    if name.endswith(".docx"):
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
# JSON extraction helper
# -------------------------
def _try_extract_json(text: str) -> Optional[Dict[str, Any]]:
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
        if m:
            return json.loads(m.group(1))
    except Exception:
        pass
    return None

# -------------------------
# Auto-call: tries candidate endpoints + payload shapes
# -------------------------
def try_remote_with_key(api_key: str, prompt: str, candidate_urls: List[str], timeout: int = DEFAULT_TIMEOUT, max_tokens=1024) -> Dict[str, Any]:
    """
    Tries candidate endpoints and payload shapes. Returns:
    { success: bool, url: str|None, parsed: dict|None, raw: str|None, diagnostics: {...} }
    """
    diagnostics = {"attempts": []}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    bodies = [
        {"input": prompt, "max_output_tokens": max_tokens, "temperature": 0.0},
        {"prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0},
        {"messages": [{"role":"user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0.0},
    ]
    last_err = None
    for url in candidate_urls:
        for shape_idx, body in enumerate(bodies, start=1):
            attempt = {"url": url, "shape_idx": shape_idx}
            try:
                r = requests.post(url, headers=headers, json=body, timeout=timeout)
                attempt["status"] = r.status_code
                attempt["ok"] = r.ok
                attempt["raw_trunc"] = r.text[:1500]
                diagnostics["attempts"].append(attempt)
                if r.ok:
                    parsed = _try_extract_json(r.text)
                    return {"success": True, "url": url, "parsed": parsed, "raw": r.text, "diagnostics": diagnostics}
                else:
                    last_err = f"HTTP {r.status_code}"
            except Exception as e:
                attempt["exception"] = repr(e)
                diagnostics["attempts"].append(attempt)
                last_err = repr(e)
                continue
    diagnostics["last_error"] = last_err
    return {"success": False, "url": None, "parsed": None, "raw": None, "diagnostics": diagnostics}

# -------------------------
# Plain OpenAI direct attempt (fallback)
# -------------------------
def try_openai_direct(api_key: str, prompt: str, timeout: int = DEFAULT_TIMEOUT, model="gpt-4o-mini", max_tokens=1024) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {"model": model, "messages":[{"role":"user","content":prompt}], "max_tokens": max_tokens, "temperature": 0.0}
    url = "https://api.openai.com/v1/chat/completions"
    try:
        r = requests.post(url, headers=headers, json=body, timeout=timeout)
        diag = {"status": r.status_code, "url": url}
        if r.ok:
            try:
                js = r.json()
                # try to find content
                content = None
                if "choices" in js and len(js["choices"]) > 0:
                    ch = js["choices"][0]
                    if isinstance(ch.get("message"), dict):
                        content = ch["message"].get("content")
                    elif "text" in ch:
                        content = ch.get("text")
                if content is None:
                    content = r.text
                parsed = _try_extract_json(content) or _try_extract_json(r.text)
                return {"success": True, "parsed": parsed, "raw": content, "diagnostics": diag}
            except Exception:
                return {"success": True, "parsed": None, "raw": r.text, "diagnostics": diag}
        else:
            return {"success": False, "parsed": None, "raw": r.text, "diagnostics": diag}
    except Exception as e:
        return {"success": False, "parsed": None, "raw": None, "diagnostics": {"exception": repr(e)}}

# -------------------------
# Prompts (strict JSON)
# -------------------------
JD_PROMPT = """You are a JSON extractor. Given the Job Description below, return ONLY a valid JSON object with keys:
- title (string)
- must_have (array of short canonical skill strings)
- good_to_have (array of short canonical skill strings)
- qualifications (array of strings for degrees/certs)
- experience (string)

Keep skill names short, canonical, and consistent. Return only JSON.

Job Description:
\"\"\"{jd}\"\"\"
"""

EVAL_PROMPT = """You are an objective resume evaluator. Given a JD JSON and a resume excerpt, return only JSON with keys:
- hard (0-100)
- soft (0-100)
- final (0-100)  # final = hard_weight*hard + soft_weight*soft
- verdict (one of "High","Medium","Low")
- missing: {{ "must": [...], "good": [...] }}
- feedback: [short actionable strings]

JD_JSON:
{jd_json}

Resume (first 7000 chars):
\"\"\"{resume_excerpt}\"\"\"

hard_weight: {hard_weight}
soft_weight: {soft_weight}
"""

# -------------------------
# Heuristic fallback (offline)
# -------------------------
def heuristic_parse_jd(text: str) -> Dict[str, Any]:
    out = {"title": "", "must_have": [], "good_to_have": [], "qualifications": [], "experience": ""}
    for line in text.splitlines():
        if line.strip():
            out["title"] = line.strip()[:200]; break
    must = re.findall(r'(?:must[-\s]*have|required)[:\-\s]*([^\n;]+)', text, flags=re.I)
    good = re.findall(r'(?:nice[-\s]*to[-\s]*have|preferred|good[-\s]*to[-\s]*have)[:\-\s]*([^\n;]+)', text, flags=re.I)
    quals = re.findall(r'(?:qualification|education|degree|certification)[:\-\s]*([^\n;]+)', text, flags=re.I)
    def split_and_clean(s): return [p.strip() for p in re.split(r'[,\;/\|]', s) if p.strip()]
    for m in must: out["must_have"].extend(split_and_clean(m))
    for g in good: out["good_to_have"].extend(split_and_clean(g))
    for q in quals: out["qualifications"].extend(split_and_clean(q))
    exp = re.search(r'(\d+\+?\s+years?|\d+-\d+\s+years)', text, flags=re.I)
    if exp: out["experience"] = exp.group(0)
    if not out["must_have"]:
        tokens = re.findall(r'\b[A-Za-z\+\#]{2,}(?:\s+[A-Za-z\+\#]{2,}){0,2}\b', text)
        noise = {"our","we","the","and","or","to","in","with","from","that","this","apply","role","job"}
        seen=set()
        for t in tokens:
            tl = t.lower()
            if tl in noise or len(t)<=2: continue
            if tl in seen: continue
            seen.add(tl); out["must_have"].append(t)
            if len(out["must_have"])>=12: break
    return out

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
    soft = round(hard * 0.92,2)
    final = round(hard_weight*hard + soft_weight*soft,2)
    verdict = "High" if final >=75 else "Medium" if final >=50 else "Low"
    missing = {"must":[m for m in must if m not in found_must], "good":[g for g in good if g not in found_good]}
    feedback = [f"Add a short project/cert demonstrating: {m}" for m in missing["must"][:4]]
    return {"hard": hard, "soft": soft, "final": final, "verdict": verdict, "missing": missing, "feedback": feedback}

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Resume Relevance (Auto Groq)", layout="wide")
st.title("Resume Relevance — give a key, nothing else (auto-handled)")

# Sidebar: only API key & provider preference
st.sidebar.header("API Key (enter only key)")
api_key_input = st.sidebar.text_input("API Key (Groq or OpenAI)", type="password")
force_openai = st.sidebar.checkbox("Force OpenAI (skip Groq auto-tries)", value=False)
use_auto_candidates = st.sidebar.checkbox("Auto-try Groq endpoints (recommended)", value=True)
st.sidebar.markdown("You do NOT need to enter endpoints. App will auto-try known Groq endpoints. If all fails it will fall back to a local heuristic parser.")

# JD upload/paste
st.header("1) Upload or paste Job Description (JD)")
col1, col2 = st.columns([3,1])
with col1:
    jd_file = st.file_uploader("Upload JD (PDF / DOCX / TXT) or paste below", type=["pdf","docx","txt"])
    jd_text_area = st.text_area("Paste JD text here (overrides file)", height=240)
with col2:
    if st.button("Clear parsed JD session"):
        st.session_state.pop("parsed_jd", None)
        st.success("Cleared parsed JD.")

jd_text = ""
if jd_file:
    try:
        jd_text = extract_text_from_bytes(jd_file.read(), jd_file.name)
    except Exception as e:
        st.error(f"Failed to read JD file: {e}")
if jd_text_area.strip():
    jd_text = jd_text_area

if not jd_text.strip():
    st.info("Upload or paste a Job Description to parse.")

# Parse JD action
if jd_text.strip():
    st.subheader("JD preview")
    st.code(jd_text[:1200])
    if st.button("Parse JD (auto)"):
        key = api_key_input.strip()
        parsed = None
        diagnostics = {}
        # if user requested OpenAI only or key looks like OpenAI sk-, try OpenAI
        try_openai_first = False
        if force_openai or (key.startswith("sk-") and len(key) > 20):
            try_openai_first = True

        if try_openai_first and key:
            st.info("Trying OpenAI (key looks like OpenAI or forced).")
            o = try_openai_direct(key, JD_PROMPT.format(jd=jd_text))
            diagnostics["openai"] = o["diagnostics"] if "diagnostics" in o else {}
            if o["success"] and o.get("parsed"):
                parsed = o["parsed"]
        # If not parsed yet and auto candidates enabled, try Groq candidates
        if parsed is None and key and use_auto_candidates and not try_openai_first:
            st.info("Auto-trying Groq candidate endpoints (no endpoint input required).")
            g = try_remote_with_key(key, JD_PROMPT.format(jd=jd_text), GROQ_CANDIDATES)
            diagnostics["groq"] = g["diagnostics"]
            if g["success"] and g.get("parsed"):
                parsed = g["parsed"]
        # If still none and key exists, try OpenAI fallback automatically
        if parsed is None and key and not try_openai_first:
            st.info("Groq attempts failed or returned non-JSON; trying OpenAI fallback.")
            o = try_openai_direct(key, JD_PROMPT.format(jd=jd_text))
            diagnostics["openai_fallback"] = o.get("diagnostics", {})
            if o["success"] and o.get("parsed"):
                parsed = o["parsed"]
        # If parsed still None -> heuristic
        if parsed is None:
            st.warning("Remote parsing failed or returned no structured JSON; using local heuristic fallback.")
            parsed = heuristic_parse_jd(jd_text)
            parsed["_parse_mode"] = "heuristic_fallback"
            parsed["_diagnostics"] = diagnostics
            st.json(parsed)
            st.session_state["parsed_jd"] = parsed
        else:
            # normalize and store
            parsed_norm = {
                "title": parsed.get("title","") if isinstance(parsed.get("title",""), str) else "",
                "must_have": parsed.get("must_have",[]) if isinstance(parsed.get("must_have",[]), list) else [],
                "good_to_have": parsed.get("good_to_have",[]) if isinstance(parsed.get("good_to_have",[]), list) else [],
                "qualifications": parsed.get("qualifications",[]) if isinstance(parsed.get("qualifications",[]), list) else [],
                "experience": parsed.get("experience","") if isinstance(parsed.get("experience",""), str) else ""
            }
            parsed_norm["_parse_mode"] = "remote"
            parsed_norm["_diagnostics"] = diagnostics
            st.success("Parsed JD (remote).")
            st.json(parsed_norm)
            st.session_state["parsed_jd"] = parsed_norm

# Resumes upload & evaluation
st.header("2) Upload resumes and evaluate")
resume_files = st.file_uploader("Upload resumes (multiple) (PDF/DOCX/TXT)", accept_multiple_files=True, type=["pdf","docx","txt"])
hard_weight = st.slider("Hard match weight", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
soft_weight = round(1.0 - hard_weight, 2)

if st.button("Evaluate resumes"):
    if "parsed_jd" not in st.session_state:
        st.error("Parse a JD first.")
    elif not resume_files:
        st.error("Upload at least one resume file.")
    else:
        jd_parsed = st.session_state["parsed_jd"]
        key = api_key_input.strip()
        results = []
        for f in resume_files:
            try:
                txt = extract_text_from_bytes(f.read(), f.name)
                parsed_eval = None
                diag = {}
                # If key provided, try remote evaluation
                if key:
                    # prefer Groq auto unless forced OpenAI or key indicates OpenAI
                    tried_openai = False
                    if force_openai or (key.startswith("sk-") and len(key)>20):
                        tried_openai = True
                        o = try_openai_direct(key, EVAL_PROMPT.format(jd_json=json.dumps(jd_parsed, ensure_ascii=False), resume_excerpt=txt[:7000], hard_weight=hard_weight, soft_weight=soft_weight))
                        diag["openai"] = o.get("diagnostics", {})
                        if o["success"] and o.get("parsed"):
                            parsed_eval = o["parsed"]
                    if parsed_eval is None and not tried_openai:
                        g = try_remote_with_key(key, EVAL_PROMPT.format(jd_json=json.dumps(jd_parsed, ensure_ascii=False), resume_excerpt=txt[:7000], hard_weight=hard_weight, soft_weight=soft_weight), GROQ_CANDIDATES)
                        diag["groq"] = g.get("diagnostics", {})
                        if g["success"] and g.get("parsed"):
                            parsed_eval = g["parsed"]
                        else:
                            # try openai fallback
                            o = try_openai_direct(key, EVAL_PROMPT.format(jd_json=json.dumps(jd_parsed, ensure_ascii=False), resume_excerpt=txt[:7000], hard_weight=hard_weight, soft_weight=soft_weight))
                            diag["openai_fallback"] = o.get("diagnostics", {})
                            if o["success"] and o.get("parsed"):
                                parsed_eval = o["parsed"]
                # If remote didn't produce parsed result -> heuristic score
                if parsed_eval is None:
                    parsed_eval = heuristic_score(txt, jd_parsed, hard_weight, soft_weight)
                    parsed_eval["_eval_mode"] = "heuristic"
                else:
                    # normalize numbers if present
                    try:
                        parsed_eval["hard"] = float(parsed_eval.get("hard", 0.0))
                        parsed_eval["soft"] = float(parsed_eval.get("soft", 0.0))
                        parsed_eval["final"] = float(parsed_eval.get("final", round(parsed_eval["hard"]*hard_weight + parsed_eval["soft"]*soft_weight,2)))
                    except Exception:
                        parsed_eval = heuristic_score(txt, jd_parsed, hard_weight, soft_weight)
                        parsed_eval["_eval_mode"] = "heuristic_after_parse_error"
                results.append({"file": f.name, "result": parsed_eval, "diagnostics": diag})
            except Exception as e:
                st.error(f"Failed to process {f.name}: {e}")
                st.text(traceback.format_exc())
        # display results sorted
        results = sorted(results, key=lambda r: r["result"].get("final",0), reverse=True)
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
            if r.get("diagnostics"):
                with st.expander("Diagnostics"):
                    st.json(r["diagnostics"])

st.markdown("---")
st.caption("You only need to supply an API key. App auto-tries Groq endpoints and OpenAI as fallback. If remote fails, it uses a local heuristic so the app always works.")
