# dashboard/streamlit_app.py
"""
Streamlit app — Groq-powered resume relevance checker (single-file).
Usage:
  - Provide GROQ_API_URL and GROQ_API_KEY in the sidebar (or set env vars first).
  - Upload a Job Description (PDF/DOCX/TXT) or paste text.
  - Click "Parse JD (Groq)" to get structured JSON.
  - Upload multiple resumes and click "Evaluate resumes" to get scores & feedback.

Notes:
  - This calls your Groq endpoint directly from Streamlit.
  - Prompts request STRICT JSON; responses are parsed and validated.
  - Fallback: local heuristic parsing/scoring if Groq is unavailable.
"""

import os, re, json, tempfile, time, traceback
from typing import Optional, Dict, Any, List
import requests
import streamlit as st

# Optional file parsers (install if you want improved extraction)
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

# Optional speedups for fuzzy matching
try:
    from rapidfuzz import fuzz
    _HAS_RAPIDFUZZ = True
except Exception:
    _HAS_RAPIDFUZZ = False

# Optional sentence transformer for a local soft score fallback (if installed)
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _HAS_SBERT = True
    _SBERT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
except Exception:
    _HAS_SBERT = False

# -------------------------
# Config / cache
# -------------------------
CACHE_FILE = "groq_rescheck_cache.json"

def load_cache():
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"jd": {}, "resumes": {}}

def save_cache(cache):
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

CACHE = load_cache()

# -------------------------
# Utility: file text extraction
# -------------------------
def extract_text_from_bytes(b: bytes, filename: str) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        return _extract_pdf_bytes(b)
    if name.endswith(".docx"):
        return _extract_docx_bytes(b)
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
    try:
        return b.decode("utf-8", errors="ignore")
    except:
        return ""

def _extract_docx_bytes(b: bytes) -> str:
    if _HAS_DOCX2TXT:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".docx")
        tmp.write(b); tmp.close()
        try:
            text = docx2txt.process(tmp.name) or ""
            return text
        finally:
            try: os.unlink(tmp.name)
            except: pass
    try:
        return b.decode("utf-8", errors="ignore")
    except:
        return ""

# -------------------------
# Groq client (robust): tries several payload shapes
# -------------------------
DEFAULT_TIMEOUT = 60

def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    # direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # balanced-brace extraction
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

def call_groq_endpoint(prompt: str, groq_url: str, groq_key: str, max_tokens: int = 1024, temperature: float = 0.0) -> Dict[str, Any]:
    """
    Returns a dict:
      {
         success: bool,
         raw: str or None,
         parsed_json: dict or None,
         diagnostics: {...}
      }
    """
    diagnostics = {"attempts": [], "time": time.time()}
    headers = {"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json", "Accept": "application/json"}
    # Try common payload shapes that Groq endpoints often accept
    bodies = [
        {"input": prompt, "max_output_tokens": max_tokens, "temperature": temperature},
        {"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature},
        {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": temperature}
    ]
    last_err = None
    for idx, body in enumerate(bodies, start=1):
        try:
            resp = requests.post(groq_url, headers=headers, json=body, timeout=DEFAULT_TIMEOUT)
            diagnostics["attempts"].append({"shape_idx": idx, "status": resp.status_code, "ok": resp.ok})
            text = resp.text
            diagnostics["attempts"][-1]["raw_trunc"] = text[:2000]
            if resp.ok:
                parsed = _extract_json_from_text(text)
                return {"success": True, "raw": text, "parsed_json": parsed, "diagnostics": diagnostics}
            else:
                last_err = f"HTTP {resp.status_code}"
        except Exception as e:
            last_err = repr(e)
            diagnostics["attempts"].append({"shape_idx": idx, "exception": repr(e)})
            continue
    diagnostics["last_error"] = last_err
    return {"success": False, "raw": None, "parsed_json": None, "diagnostics": diagnostics}

# -------------------------
# Prompts (strict JSON output)
# -------------------------
JD_PROMPT = """You are a JSON extractor. Given the job description below, return ONLY a valid JSON object with keys:
- title (string)
- must_have (array of short canonical skill strings)
- good_to_have (array of short canonical skill strings)
- qualifications (array of strings: degrees/certifications)
- experience (string, e.g., "1-3 years" or "")

Keep skill names short (1-3 words), canonical (e.g., "Python", "PyTorch", "NLP"). Return only JSON, no commentary.

Job Description:
\"\"\"{jd}\"\"\"
"""

EVAL_PROMPT = """You are a resume evaluator. Given a structured JD (JSON) and a resume text, return ONLY valid JSON with keys:
- hard (number 0-100) : keyword/requirements coverage
- soft (number 0-100) : semantic fit score
- final (number 0-100) : final weighted score (final = hard_weight*hard + soft_weight*soft)
- verdict (string): one of "High", "Medium", "Low"
- missing (object): { "must": [...], "good": [...] } lists of missing items from JD
- feedback (array of short actionable strings 3-6 items)

Inputs:
JD_PARSED_JSON:
{jd_json}

Resume (first 7000 chars):
\"\"\"{resume_excerpt}\"\"\"

Use weights:
hard_weight: {hard_weight}
soft_weight: {soft_weight}

Return only JSON.
"""

# -------------------------
# Heuristic fallback parser + scorer (offline)
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
            if tl in noise or len(t)<=2: continue
            if tl in seen: continue
            seen.add(tl); out.append(t)
            if len(out)>=12: break
        res["must_have"].extend(out)
    return res

def heuristic_score(resume_text: str, jd_parsed: Dict[str, Any], hard_weight=0.6, soft_weight=0.4) -> Dict[str, Any]:
    must = jd_parsed.get("must_have", [])
    good = jd_parsed.get("good_to_have", [])
    found_must = []
    found_good = []
    if _HAS_RAPIDFUZZ:
        for s in must:
            if fuzz.partial_ratio(s.lower(), resume_text.lower()) >= 70:
                found_must.append(s)
        for s in good:
            if fuzz.partial_ratio(s.lower(), resume_text.lower()) >= 70:
                found_good.append(s)
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
    verdict = "High" if final >= 75 else "Medium" if final >=50 else "Low"
    missing = {"must":[m for m in must if m not in found_must], "good":[g for g in good if g not in found_good]}
    feedback = [f"Add a short project/cert demonstrating: {m}" for m in missing["must"][:4]]
    return {"hard": hard, "soft": soft, "final": final, "verdict": verdict, "missing": missing, "feedback": feedback}

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(layout="wide", page_title="Groq Resume Relevance")
st.title("Groq Resume Relevance — JD parsing & resume scoring")

# Sidebar: Groq config
st.sidebar.header("Groq API settings")
groq_url = st.sidebar.text_input("Groq API URL (full endpoint)", value=os.getenv("GROQ_API_URL",""))
groq_key = st.sidebar.text_input("Groq API Key", value=os.getenv("GROQ_API_KEY",""), type="password")
st.sidebar.caption("If your Groq endpoint requires a different payload, the app tries common shapes (input, prompt, messages).")
st.sidebar.markdown("---")
st.sidebar.write("Cache is stored locally to avoid repeated LLM calls.")

# JD upload or paste
st.header("1) Job Description (JD)")
col_left, col_right = st.columns([3,1])
with col_left:
    jd_file = st.file_uploader("Upload JD (PDF / DOCX / TXT) or paste text below", type=["pdf","docx","txt"])
    jd_text_area = st.text_area("Paste JD text (overrides file preview)", height=260)
with col_right:
    if st.button("Clear cached JD"):
        CACHE["jd"].clear(); save_cache(CACHE); st.success("Cached JD cleared.")

jd_text = ""
if jd_file:
    try:
        jd_text = extract_text_from_bytes(jd_file.read(), jd_file.name)
    except Exception as e:
        st.error("Failed to extract JD: " + str(e))
if jd_text_area.strip():
    jd_text = jd_text_area

if jd_text.strip():
    st.subheader("JD preview (first 1200 chars)")
    st.code(jd_text[:1200])
    # parse actions
    parse_col, heur_col = st.columns(2)
    with parse_col:
        if st.button("Parse JD (Groq)"):
            # cache key - simple hash of content
            cache_key = "jd_" + str(abs(hash(jd_text)) % (10**12))
            if cache_key in CACHE["jd"]:
                parsed = CACHE["jd"][cache_key]
                st.success("Loaded parsed JD from cache.")
                st.json(parsed)
                st.session_state["parsed_jd"] = parsed
            else:
                if not groq_url or not groq_key:
                    st.warning("Groq URL or API key not provided. Using local heuristic fallback.")
                    parsed = heuristic_parse_jd(jd_text)
                    st.json(parsed)
                    st.session_state["parsed_jd"] = parsed
                else:
                    with st.spinner("Calling Groq for JD parsing..."):
                        out = call_groq_endpoint(JD_PROMPT.format(jd=jd_text), groq_url, groq_key, max_tokens=800, temperature=0.0)
                        if out["success"] and out["parsed_json"]:
                            parsed = out["parsed_json"]
                            # minimal normalization
                            parsed = {
                                "title": parsed.get("title", "") if isinstance(parsed.get("title",""), str) else "",
                                "must_have": parsed.get("must_have",[]) if isinstance(parsed.get("must_have",[]), list) else [],
                                "good_to_have": parsed.get("good_to_have",[]) if isinstance(parsed.get("good_to_have",[]), list) else [],
                                "qualifications": parsed.get("qualifications",[]) if isinstance(parsed.get("qualifications",[]), list) else [],
                                "experience": parsed.get("experience","") if isinstance(parsed.get("experience",""), str) else "",
                                "raw_text": jd_text
                            }
                            CACHE["jd"][cache_key] = parsed; save_cache(CACHE)
                            st.success("Parsed JD (Groq).")
                            st.json(parsed)
                            st.session_state["parsed_jd"] = parsed
                        else:
                            st.error("Groq parse failed — falling back to heuristic. See diagnostics.")
                            st.json(out["diagnostics"])
                            parsed = heuristic_parse_jd(jd_text)
                            parsed["parse_fallback_reason"] = out["diagnostics"].get("last_error", "groq_failed")
                            st.subheader("Heuristic JD parse (fallback)")
                            st.json(parsed)
                            st.session_state["parsed_jd"] = parsed
    with heur_col:
        if st.button("Parse JD (heuristic)"):
            parsed = heuristic_parse_jd(jd_text)
            st.success("Parsed JD using heuristic.")
            st.json(parsed)
            st.session_state["parsed_jd"] = parsed
else:
    st.info("Upload or paste a Job Description to parse.")

# Resumes upload & evaluate
st.header("2) Upload resumes & evaluate against parsed JD")
resume_files = st.file_uploader("Upload resumes (multiple) (PDF/DOCX/TXT)", accept_multiple_files=True, type=["pdf","docx","txt"])
hard_weight = st.slider("Hard match weight", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
soft_weight = round(1.0 - hard_weight, 2)
st.write("Soft weight:", soft_weight)

if st.button("Evaluate resumes"):
    if "parsed_jd" not in st.session_state:
        st.error("No parsed JD in session. Parse a JD first.")
    elif not resume_files:
        st.error("Upload one or more resumes.")
    else:
        jd_parsed = st.session_state["parsed_jd"]
        results = []
        for f in resume_files:
            try:
                content_bytes = f.read()
                resume_text = extract_text_from_bytes(content_bytes, f.name)
                # cache key derived from resume + jd
                cache_key = "eval_" + str(abs(hash((resume_text[:4000], json.dumps(jd_parsed, sort_keys=True)))) % (10**12))
                if cache_key in CACHE["resumes"]:
                    res = CACHE["resumes"][cache_key]
                    st.info(f"Loaded cached evaluation for {f.name}")
                    results.append({"file": f.name, "result": res, "cached": True})
                    continue

                if not groq_url or not groq_key:
                    # heuristic only
                    res = heuristic_score(resume_text, jd_parsed, hard_weight, soft_weight)
                    CACHE["resumes"][cache_key] = res; save_cache(CACHE)
                    results.append({"file": f.name, "result": res, "cached": False})
                else:
                    # call groq for evaluation
                    with st.spinner(f"Evaluating {f.name} with Groq..."):
                        prompt = EVAL_PROMPT.format(jd_json=json.dumps(jd_parsed, ensure_ascii=False), resume_excerpt=resume_text[:7000], hard_weight=hard_weight, soft_weight=soft_weight)
                        out = call_groq_endpoint(prompt, groq_url, groq_key, max_tokens=1000, temperature=0.0)
                        if out["success"] and out["parsed_json"]:
                            parsed_eval = out["parsed_json"]
                            # normalize numeric fields
                            try:
                                parsed_eval["hard"] = float(parsed_eval.get("hard", 0.0))
                                parsed_eval["soft"] = float(parsed_eval.get("soft", 0.0))
                                parsed_eval["final"] = float(parsed_eval.get("final", round(parsed_eval["hard"]*hard_weight + parsed_eval["soft"]*soft_weight,2)))
                            except Exception:
                                parsed_eval = heuristic_score(resume_text, jd_parsed, hard_weight, soft_weight)
                                parsed_eval["eval_fallback_reason"] = "groq_output_shape"
                            CACHE["resumes"][cache_key] = parsed_eval; save_cache(CACHE)
                            results.append({"file": f.name, "result": parsed_eval, "diagnostics": out["diagnostics"], "cached": False})
                        else:
                            # fallback
                            st.warning(f"Groq eval failed for {f.name} — using heuristic fallback. See diagnostics.")
                            st.json(out["diagnostics"])
                            parsed_eval = heuristic_score(resume_text, jd_parsed, hard_weight, soft_weight)
                            parsed_eval["eval_fallback_reason"] = out["diagnostics"].get("last_error", "groq_failed")
                            CACHE["resumes"][cache_key] = parsed_eval; save_cache(CACHE)
                            results.append({"file": f.name, "result": parsed_eval, "diagnostics": out["diagnostics"], "cached": False})
            except Exception as e:
                st.error(f"Failed processing {f.name}: {e}")
                st.text(traceback.format_exc())

        # show sorted results
        results = sorted(results, key=lambda x: x["result"].get("final", 0), reverse=True)
        st.header("Evaluation results")
        for r in results:
            res = r["result"]
            st.subheader(f"{r['file']} — Final: {res.get('final','N/A')}  Verdict: {res.get('verdict','N/A')}{' (cached)' if r.get('cached') else ''}")
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
                with st.expander("Groq diagnostics (for this file)"):
                    st.json(r["diagnostics"])

st.markdown("---")
st.caption("Tips: Use real Groq API URL & API key in the sidebar. If Groq fails or you lack a key, the app falls back to a local heuristic.")

