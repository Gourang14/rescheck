"""
JD parser: tries to use an LLM (OpenAI) to extract structured fields.
Falls back to heuristic regex extraction if LLM not available.
"""

import os
import re
from typing import Dict, List

# try OpenAI via the lightweight OpenAI client; fallback if not present
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False


def _heuristic_parse_jd(raw_text: str, title: str = "") -> Dict:
    """
    Simple regex/heuristic JD parsing: finds lines with keywords like 'must have', 'good to have'
    Returns:
        dict: { title, raw_text, must_have:[], good_to_have:[], qualifications:[], experience: str }
    """
    text = raw_text.replace("\r", "\n")
    res = {"title": title or "", "raw_text": raw_text, "must_have": [], "good_to_have": [], "qualifications": [], "experience": ""}

    # heuristics: look for sections or lines containing 'must', 'require', 'good to have', 'preferred'
    must_patterns = [r"(must[-\s]*have[:\s]*)(.+)", r"(required[:\s]*)(.+)", r"(requirements[:\s]*)(.+)"]
    prefer_patterns = [r"(good[-\s]*to[-\s]*have[:\s]*)(.+)", r"(preferred[:\s]*)(.+)"]

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    joined = "\n".join(lines)

    # gather phrase lists by scanning lines
    for l in lines:
        lower = l.lower()
        if 'must have' in lower or 'must-have' in lower or 'required' in lower or 'requirements' in lower:
            # extract words after colon
            parts = re.split(r'[:\-–]', l, maxsplit=1)
            if len(parts) > 1:
                res["must_have"].extend([x.strip() for x in re.split(r'[;,/]| and ', parts[1]) if x.strip()])
        if 'good to have' in lower or 'preferred' in lower:
            parts = re.split(r'[:\-–]', l, maxsplit=1)
            if len(parts) > 1:
                res["good_to_have"].extend([x.strip() for x in re.split(r'[;,/]| and ', parts[1]) if x.strip()])

    # fallback: try to detect skill tokens capitalized or common skill words
    if not res["must_have"]:
        # take top nouns / capitalized tokens as candidates
        candidates = re.findall(r'\b[A-Z][a-zA-Z0-9\+\#\.]{2,}\b', joined)
        res["must_have"] = list(dict.fromkeys(candidates))[:10]

    # qualifications and experience: look for 'B.Tech', 'Bachelor', 'years'
    quals = re.findall(r'(B\.?Tech|Bachelor|Master|M\.?Tech|MBA|Ph\.?D|BSc|MSc)[^\n]*', joined, flags=re.I)
    res["qualifications"] = list(dict.fromkeys([q.strip() for q in quals]))[:5]

    exp = re.search(r'(\d+\+?\s+years? of experience|\d+-\d+\s+years|\d+\s+years)', joined, flags=re.I)
    res["experience"] = exp.group(0) if exp else ""

    # reduce noise
    res["must_have"] = [s for s in res["must_have"] if len(s) > 1][:15]
    res["good_to_have"] = [s for s in res["good_to_have"] if len(s) > 1][:15]
    return res


def parse_jd_with_llm(raw_text: str, title: str = "") -> Dict:
    """
    Try to parse JD using LLM (OpenAI). If not available or an error occurs, fallback heuristics.
    The LLM prompt asks for a JSON object with keys: title, must_have, good_to_have, qualifications, experience
    """
    if not OPENAI_AVAILABLE or not os.getenv("OPENAI_API_KEY"):
        return _heuristic_parse_jd(raw_text, title)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = f"""
    Extract the following fields from this job description. Output strict JSON with keys:
    - title (string)
    - must_have (list of short strings)
    - good_to_have (list of short strings)
    - qualifications (list of short strings like degrees/certifications)
    - experience (string; e.g. "3-5 years")

    Job description:
    \"\"\"{raw_text}\"\"\"
    """

    try:
        resp = client.responses.create(
            model="gpt-4o-mini",  # change as available; or use GPT-4/GPT-3.5
            input=[
                {"role":"user","content": prompt}
            ],
            max_output_tokens=800
        )
        # Attempt to parse as JSON from text output
        raw_out = ""
        # openai.responses returns structured output choices
        for item in resp.output:
            if hasattr(item, "content") and isinstance(item.content, list):
                for c in item.content:
                    if c.get("type") == "output_text":
                        raw_out += c.get("text", "")
            elif isinstance(item, dict):
                raw_out += item.get("text", "")

        # try to find JSON substring
        import json, re
        match = re.search(r'(\{.*\})', raw_out, flags=re.S)
        if match:
            parsed = json.loads(match.group(1))
            # ensure lists exist
            parsed.setdefault("title", title)
            parsed.setdefault("must_have", [])
            parsed.setdefault("good_to_have", [])
            parsed.setdefault("qualifications", [])
            parsed.setdefault("experience", "")
            parsed["raw_text"] = raw_text
            return parsed
        else:
            # fallback heuristics
            return _heuristic_parse_jd(raw_text, title)
    except Exception:
        return _heuristic_parse_jd(raw_text, title)
