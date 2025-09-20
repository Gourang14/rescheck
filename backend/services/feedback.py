import os
from typing import List

# Edit depending on LLM availability
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

def _heuristic_feedback(missing: List[str], resume_text: str, jd_title: str = "") -> str:
    if not missing:
        return "Good match — no critical skills missing. Consider adding more domain-specific projects to stand out."
    bullets = []
    for s in missing[:6]:
        bullets.append(f"- Add a short project or certification demonstrating **{s}** (small project, 1-2 weeks).")
    bullets.append("Also highlight measurable outcomes (metrics, links, durations).")
    return "\n".join(bullets)

def generate_feedback(resume_text: str, jd_text: str, missing: List[str], jd_title: str = "") -> str:
    """
    Use an LLM (if available) to craft 4 concise actionable suggestions.
    Fallback to heuristic suggestions if LLM/API not available.
    """
    if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        prompt = f"""
You are a friendly career coach. Given a candidate resume text and the job description, provide 4 short, actionable suggestions (1 sentence each) the candidate can do in 4-8 weeks to improve fit for the job titled '{jd_title}'.
Mention specific skills from the missing list: {missing}.
Resume excerpt (first 800 chars):
{resume_text[:800]}

Return only JSON: {{ "suggestions": ["...", "...", "...", "..."] }}
"""
        try:
            resp = client.responses.create(model="gpt-4o-mini", input=[{"role":"user","content":prompt}], max_output_tokens=300)
            raw = ""
            for item in resp.output:
                if isinstance(item, dict) and item.get("type") == "output_text":
                    raw += item.get("text","")
                elif isinstance(item, str):
                    raw += item
            import json, re
            match = re.search(r'(\{.*\})', raw, flags=re.S)
            if match:
                parsed = json.loads(match.group(1))
                sug = parsed.get("suggestions", [])
                if sug:
                    return "\n".join([f"- {s}" for s in sug])
            # fallback to heuristics if parsing failed
        except Exception:
            pass

    # fallback
    return _heuristic_feedback(missing, resume_text, jd_title)