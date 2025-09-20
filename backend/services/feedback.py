# backend/services/feedback.py
import os, re
from typing import List

# stopwords or noise tokens often extracted from naive JD parsing
NOISE_TOKENS = set(["Job","Role","Description","Overview","Apply","Apply Now","Duration","Detailed","Internship","Interns","Walk","Bond","Detailed"])

def _clean_skill_token(token: str) -> str:
    token = re.sub(r'[^A-Za-z0-9\+\#\. ]', '', token).strip()
    return token

def _filter_skills(raw_skills: List[str]) -> List[str]:
    cleaned = []
    for s in raw_skills:
        s2 = _clean_skill_token(s)
        if not s2 or s2.lower() in (w.lower() for w in NOISE_TOKENS):
            continue
        # drop tokens that are single letters or numbers
        if len(s2) <= 2:
            continue
        cleaned.append(s2)
    # dedupe preserving order
    seen = set(); out = []
    for x in cleaned:
        if x.lower() not in seen:
            out.append(x); seen.add(x.lower())
    return out

def _heuristic_feedback(missing: List[str]) -> str:
    missing = _filter_skills(missing)
    if not missing:
        return "Good match. Consider strengthening measurable results (metrics) and adding 1–2 domain projects."
    suggestions = []
    # For each missing skill produce a 1-line actionable advise
    for s in missing[:6]:
        suggestions.append(f"- Build a 1–2 week mini-project demonstrating {s} (a short repo + README, 3–4 key metrics).")
    suggestions.append("- Add concise metrics (e.g., latency, accuracy, throughput) and links to code/sample outputs.")
    return "\n".join(suggestions)

# LLM-based feedback (optional)
def generate_feedback(resume_text: str, jd_text: str, missing: List[str], jd_title: str = "") -> str:
    # Clean the missing skill tokens first
    missing_clean = _filter_skills(missing)
    # if no LLM available or no key, deliver heuristic feedback
    try:
        from openai import OpenAI
        key = os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY")
        if not key:
            return _heuristic_feedback(missing)
        client = OpenAI(api_key=key)
        prompt = f"""You are a concise career coach. Given missing skills {missing_clean} for JD titled '{jd_title}', produce 4 short, concrete suggestions (1 sentence each) the candidate can accomplish in 2-6 weeks. Output plain text bullets only."""
        resp = client.responses.create(model="gpt-4o-mini", input=[{"role":"user","content":prompt}], max_output_tokens=200)
        out = ""
        for item in resp.output:
            if isinstance(item, dict) and item.get("type") == "output_text":
                out += item.get("text","")
            elif isinstance(item, str):
                out += item
        # fallback to heuristic if response empty
        if not out.strip():
            return _heuristic_feedback(missing_clean)
        return out.strip()
    except Exception:
        return _heuristic_feedback(missing)
