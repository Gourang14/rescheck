# dashboard/streamlit_app.py
import os
import re
import json
import tempfile
import traceback
from typing import Optional, Dict, Any, List

import requests
import streamlit as st

# Optional parsers: PyMuPDF (fitz) or pdfplumber for PDFs, docx2txt for docx
# Install as needed: pip install PyMuPDF pdfplumber docx2txt rapidfuzz sentence-transformers
try:
    import fitz  # PyMuPDF
    _HAS_FITZ = True
except Exception:
    _HAS_FITZ = False

try:
    import pdfplumber
    _HAS_PDFPLUMBER = True
except Exception:
    _HAS_PDFPLUMBER = False

try:
    import docx2txt
    _HAS_DOCX2TXT = True
except Exception:
    _HAS_DOCX2TXT = False

# Optional local tools
try:
    from rapidfuzz import fuzz
    _HAS_RAPIDFUZZ = True
except Exception:
    _HAS_RAPIDFUZZ = False

# -------------------------
# UI & Config
# -------------------------
st.set_page_config(page_title="Resume Relevance — Groq (LLM)", layout="wide")
st.title("Resume Relevance — Groq LLM (client-side)")

st.sidebar.header("Groq settings (enter to use Groq)")
groq_url = st.sidebar.text_input("Groq API URL (full endpoint)", value=os.getenv("GROQ_API_URL", ""))
groq_key = st.sidebar.text_input("Groq API Key", value=os.getenv("GROQ_API_KEY", ""), type="password")
use_groq = st.sidebar.checkbox("Use Groq for parsing & evaluation", value=True)

st.sidebar.caption("If Groq fails, a local heuristic will be used as a fallback.")

# -------------------------
# Helpers: file text extraction
# -------------------------
def extract_text_from_pdf_bytes(b: bytes) -> str:
    if _HAS_FITZ:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(b)
            tmp.flush()
            path = tmp.name
        try:
            doc = fitz.open(path)
            parts = [page.get_text() for page in doc]
            doc.close()
            return "\n".join(parts)
        finally:
            try:
                os.unlink(path)
            except Exception:
                pass
    if _HAS_PDFPLUMBER:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(b); tmp.flush(); path = tmp.name
        try:
            text = []
            with pdfplumber.open(path) as pdf:
                for p in pdf.pages:
                    text.append(p.extract_text() or "")
            return "\n".join(text)
        finally:
            try: os.unlink(path)
            except: pass
    # fallback: best-effort decode (may be messy)
    try:
        return b.decode("utf-8", errors="ignore")
    except Exception:
        return ""

def extract_text_from_docx_bytes(b: bytes) -> str:
    if _HAS_DOCX2TXT:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(b); tmp.flush(); path = tmp.name
        try:
            return docx2txt.process(path) or ""
        finally:
            try: os.unlink(path)
            except: pass
    try:
        return b.decode("utf-8", errors="ignore")
    except:
        return ""

def extract_text_from_bytes(b: bytes, filename: str) -> str:
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return extract_text_from_pdf_bytes(b)
    if lower.endswith(".docx"):
        return extract_text_from_docx_bytes(b)
    # txt or fallback
    try:
        return b.decode("utf-8", errors="ignore")
    except:
        return ""

# -------------------------
# Groq client (direct call from Streamlit)
# -------------------------
DEFAULT_TIMEOUT = 45

def _try_parse_json(resp_text: str) -> Optional[Dict[str, Any]]:
    # try direct json
    try:
        return json.loads(resp_text)
    except Exception:
        pass
    # try to find first balanced {...}
    try:
        start = resp_text.find('{')
        if start != -1:
            depth = 0
            for i in range(start, len(resp_text)):
                ch = resp_text[i]
                if ch == '{': depth += 1
                elif ch == '}': depth -= 1
                if depth == 0:
                    cand = resp_text[start:i+1]
                    return json.loads(cand)
    except Exception:
        pass
    # regex attempt (less reliable)
    try:
        m = re.search(r'(\{(?:[^{}]|\n|\r)*\})', resp_text)
        if m:
            return json.loads(m.group(1))
    except Exception:
        pass
    return None

def call_groq_raw(prompt: str, model_url: str, api_key: str, max_tokens: int = 1024, temperature: float = 0.0, timeout: int = DEFAULT_TIMEOUT) -> str:
    if not model_url or not api_key:
        raise RuntimeError("Groq URL and API key required for direct calls.")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "Accept": "application/json"}
    bodies = [
        {"input": prompt, "max_output_tokens": max_tokens, "temperature": temperature},
        {"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature},
        {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": temperature}
    ]
    last_exc = None
    for body in bodies:
        try:
            r = requests.post(model_url, headers=headers, json=body, timeout=timeout)
            r.raise_for_status()
            return r.text
        except Exception as e:
            last_exc = e
            continue
    raise RuntimeError(f"All Groq request shapes failed. Last error: {last_exc}")

def groq_generate_json(prompt: str, model_url: str, api_key: str, max_tokens: int = 1024, temperature: float = 0.0):
    raw = call_groq_raw(prompt, model_url=model_url, api_key=api_key, max_tokens=max_tokens, temperature=temperature)
    parsed = _try_parse_json(raw)
    if parsed is not None:
        return parsed
    # returned non-JSON; include raw text for debugging
    return {"raw_response": raw}

# -------------------------
# Prompts
# -------------------------
JD_PROMPT = """You are an assistant that MUST return only valid JSON and nothing else.
Given the following job description, extract and return JSON with keys:
- title (string)
- must_have (array of short strings)
- good_to_have (array of short strings)
- qualifications (array of strings)
- experience (string)

Job description:
\"\"\"{jd}\"\"\"

Return only JSON.
"""

EVAL_PROMPT = """You are an evaluation engine. Return ONLY valid JSON with keys:
- hard (number 0-100): hard keyword coverage
- soft (number 0-100): semantic fit score
- final (number 0-100): weighted final score (final = hard_weight*hard + soft_weight*soft)
- verdict (string): "High" | "Medium" | "Low"
- missing (object): {{ "must": [...], "good": [...] }}
- feedback (array of short strings): 3-6 concise actionable suggestions

Inputs:
JD_PARSED_JSON:
{jd_parsed}

RESUME_TEXT (first 7000 chars):
\"\"\"{resume_excerpt}\"\"\"

WEIGHTS:
hard_weight: {hard_weight}
soft_weight: {soft_weight}

Rules:
- Use only skills present in JD_PARSED when listing missing skills.
- Numbers must be between 0 and 100.
- Return only JSON.
"""

# -------------------------
# Local heuristic fallback (simple)
# -------------------------
def heuristic_parse_jd(raw_text: str) -> Dict[str, Any]:
    text = raw_text or ""
    res = {"title": "", "must_have": [], "good_to_have": [], "qualifications": [], "experience": ""}
    # title
    for line in text.splitlines():
        if line.strip():
            res["title"] = line.strip()[:200]; break
    # find inline lists
    def find(p):
        m = re.findall(p, text, flags=re.I)
        return " ; ".join(m) if m else ""
    must = find(r'(?:must[-\s]*have|required|requirements?)\s*[:\-–]\s*([^\n]+)')
    good = find(r'(?:nice[-\s]*to[-\s]*have|good[-\s]*to[-\s]*have|preferred)\s*[:\-–]\s*([^\n]+)')
    quals = find(r'(?:qualification|qualifications|education)\s*[:\-–]\s*([^\n]+)')
    def split_list(s): return [t.strip() for t in re.split(r'[,\n;/\|]', s) if t.strip()]
    if must: res["must_have"].extend(split_list(must))
    if good: res["good_to_have"].extend(split_list(good))
    if quals: res["qualifications"].extend(split_list(quals))
    m = re.search(r'(\d+\+?\s+years?|\d+-\d+\s+years|\d+\s+years? of experience)', text, flags=re.I)
    if m: res["experience"] = m.group(0)
    # fallback tokens
    if not res["must_have"]:
        tokens = re.findall(r'\b[A-Za-z\+\#]{2,}(?:\s+[A-Za-z\+\#]{2,}){0,2}\b', text)
        noise = set(w.lower() for w in ["job","role","apply","our","we","team","experience","work","will","that","the","and","or"])
        out=[]; seen=set()
        for t in tokens:
            tl = t.lower()
            if tl in noise: continue
            if tl in seen: continue
            seen.add(tl); out.append(t)
        res["must_have"].extend(out[:20])
    return res

def heuristic_score(resume_text: str, jd_parsed: Dict[str, Any]) -> Dict[str, Any]:
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
            if s.lower() in resume_text.lower():
                found_must.append(s)
        for s in good:
            if s.lower() in resume_text.lower():
                found_good.append(s)
    must_cov = (len(found_must) / len(must)) if must else 1.0
    good_cov = (len(found_good) / len(good)) if good else 1.0
    hard = round((0.75*must_cov + 0.25*good_cov)*100, 2)
    soft = round(hard * 0.9, 2)
    final = round(0.6*hard + 0.4*soft, 2)
    verdict = "High" if final >= 75 else "Medium" if final >= 50 else "Low"
    missing = {"must": [m for m in must if m not in found_must], "good": [g for g in good if g not in found_good]}
    feedback = [f"Add short project or certification demonstrating: {m}" for m in missing["must"][:4]]
    return {"hard": hard, "soft": soft, "final": final, "verdict": verdict, "missing": missing, "feedback": feedback}

# -------------------------
# Main app UI
# -------------------------
st.header("Upload Job Description (JD)")
col1, col2 = st.columns([3, 1])
with col1:
    jd_file = st.file_uploader("Upload JD (pdf/docx/txt)", type=["pdf", "docx", "txt"])
    jd_text_preview = ""
    jd_title = st.text_input("Optional job title (local only)", value="")
with col2:
    st.write("Groq quick test")
    if st.button("Ping Groq (quick test)"):
        if not groq_url or not groq_key:
            st.warning("Provide Groq URL and Key in the sidebar.")
        else:
            try:
                test_prompt = "Return JSON: {\"ping\":\"ok\"}"
                resp = call_groq_raw(test_prompt, model_url=groq_url, api_key=groq_key, max_tokens=32)
                st.success("Groq endpoint responded (raw, truncated):")
                st.code(resp[:2000])
            except Exception as e:
                show = str(e)
                st.error("Groq test failed: " + show)

if jd_file:
    b = jd_file.read()
    jd_text_preview = extract_text_from_bytes(b, jd_file.name)
    st.subheader("JD preview (first 1200 chars)")
    st.code(jd_text_preview[:1200])

    parse_btn_col, fallback_parse_col = st.columns(2)
    with parse_btn_col:
        if st.button("Parse JD with Groq"):
            if not groq_url or not groq_key or not use_groq:
                st.warning("Groq not configured or disabled. Using local heuristic fallback.")
                parsed = heuristic_parse_jd(jd_text_preview)
                st.json(parsed)
                st.session_state["parsed_jd"] = parsed
            else:
                try:
                    prompt = JD_PROMPT.format(jd=jd_text_preview)
                    parsed = groq_generate_json(prompt, model_url=groq_url, api_key=groq_key, max_tokens=800, temperature=0.0)
                    if isinstance(parsed, dict) and parsed.get("raw_response"):
                        st.warning("Groq returned non-JSON; showing raw response and fallback heuristic.")
                        st.code(parsed["raw_response"][:4000])
                        parsed = heuristic_parse_jd(jd_text_preview)
                        parsed["parse_error"] = "groq_raw_returned"
                        st.json(parsed)
                    else:
                        st.success("Parsed by Groq (JSON):")
                        st.json(parsed)
                    st.session_state["parsed_jd"] = parsed
                except Exception as e:
                    st.error("Groq parse failed; using heuristic fallback.")
                    st.text(str(e))
                    st.text(traceback.format_exc())
                    parsed = heuristic_parse_jd(jd_text_preview)
                    st.json(parsed)
                    st.session_state["parsed_jd"] = parsed
    with fallback_parse_col:
        if st.button("Parse JD locally (heuristic)"):
            parsed = heuristic_parse_jd(jd_text_preview)
            st.json(parsed)
            st.session_state["parsed_jd"] = parsed

# -------------------------
# Resumes: upload and evaluate locally (Groq or heuristic)
# -------------------------
st.header("Upload resumes and evaluate")
resume_files = st.file_uploader("Upload resumes (multiple)", accept_multiple_files=True, type=["pdf", "docx", "txt"])
hard_weight = st.slider("Hard match weight", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
soft_weight = round(1.0 - hard_weight, 2)
st.write("Soft weight:", soft_weight)

if st.button("Evaluate uploaded resumes"):
    if "parsed_jd" not in st.session_state:
        st.error("No parsed JD available. Parse a JD first.")
    else:
        jd_parsed = st.session_state["parsed_jd"]
        results = []
        for f in resume_files:
            try:
                st.info(f"Processing {f.name}...")
                rb = f.read()
                resume_text = extract_text_from_bytes(rb, f.name)
                # If Groq is enabled, call groq evaluator
                if use_groq and groq_url and groq_key:
                    try:
                        prompt = EVAL_PROMPT.format(jd_parsed=json.dumps(jd_parsed, ensure_ascii=False), resume_excerpt=resume_text[:7000], hard_weight=hard_weight, soft_weight=soft_weight)
                        parsed_eval = groq_generate_json(prompt, model_url=groq_url, api_key=groq_key, max_tokens=1000, temperature=0.0)
                        # If non-json returned, fallback to heuristic
                        if isinstance(parsed_eval, dict) and parsed_eval.get("raw_response"):
                            st.warning(f"Groq returned non-JSON for {f.name}. Using heuristic fallback and showing raw Groq output (truncated).")
                            st.code(parsed_eval["raw_response"][:4000])
                            parsed_eval = heuristic_score(resume_text, jd_parsed)
                        else:
                            # ensure numeric types
                            try:
                                parsed_eval["hard"] = float(parsed_eval.get("hard", 0.0))
                                parsed_eval["soft"] = float(parsed_eval.get("soft", 0.0))
                                parsed_eval["final"] = float(parsed_eval.get("final", parsed_eval["hard"]*hard_weight + parsed_eval["soft"]*soft_weight))
                            except Exception:
                                # ensure shape
                                parsed_eval = heuristic_score(resume_text, jd_parsed)
                        results.append({"filename": f.name, "result": parsed_eval})
                    except Exception as e:
                        st.error(f"Groq evaluation failed for {f.name}, using heuristic fallback.")
                        st.text(str(e))
                        st.text(traceback.format_exc())
                        parsed_eval = heuristic_score(resume_text, jd_parsed)
                        results.append({"filename": f.name, "result": parsed_eval})
                else:
                    # heuristic
                    parsed_eval = heuristic_score(resume_text, jd_parsed)
                    results.append({"filename": f.name, "result": parsed_eval})
            except Exception as e:
                st.error(f"Failed to process {f.name}: {e}")
                st.text(traceback.format_exc())

        # display results sorted by final score
        results = sorted(results, key=lambda x: x["result"].get("final", 0), reverse=True)
        for r in results:
            res = r["result"]
            st.markdown(f"### {r['filename']} — **Final:** {res.get('final', 'N/A')}  Verdict: {res.get('verdict','N/A')}")
            st.write("Hard:", res.get("hard"), "Soft:", res.get("soft"))
            missing = res.get("missing", {})
            st.write("Missing (must):", missing.get("must", []))
            st.write("Missing (good):", missing.get("good", []))
            fb = res.get("feedback", [])
            if fb:
                st.write("Feedback:")
                for item in fb[:6]:
                    st.markdown(f"- {item}")

st.markdown("---")
st.caption("This app performs JD parsing and resume scoring locally in Streamlit. Supply Groq API URL + key in the sidebar to use Groq. If Groq is not provided, a heuristic fallback will be used.")
