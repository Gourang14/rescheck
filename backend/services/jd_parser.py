# backend/services/jd_parser.py
import os, re, json
from typing import Dict, List

# Heuristic fallback parser (safe, no LLM)
def _heuristic_parse_jd(raw_text: str, title: str = "") -> Dict:
    text = raw_text.replace("\r", "\n")
    out = {"title": title or "", "raw_text": raw_text, "must_have": [], "good_to_have": [], "qualifications": [], "experience": ""}
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    joined = "\n".join(lines)

    # find salary/experience patterns
    exp = re.search(r'(\d+\+?\s+years? of experience|\d+-\d+\s+years|\d+\s+years)', joined, flags=re.I)
    out["experience"] = exp.group(0) if exp else ""

    # collect plausible skills: capitalized tokens and known skill tokens
    tokens = re.findall(r'\b[A-Za-z0-9\+\#\.]{2,}\b', joined)
    # common noise words to drop
    noise = set(["Job","Role","Description","Overview","Apply","Apply Now","Apply here","Detailed","Duration","Internship","Interns"])
    candidates = [t for t in tokens if len(t)>1 and t[0].isupper() and t not in noise]
    out["must_have"] = list(dict.fromkeys(candidates))[:12]

    # qualifications (degree tokens)
    quals = re.findall(r'\b(B\.?Tech|Bachelor|Master|M\.?Tech|MBA|Ph\.?D|BSc|MSc|B\.Sc|M\.Sc|BE)\b', joined, flags=re.I)
    out["qualifications"] = list(dict.fromkeys(quals))[:5]
    return out

# LLM parsing (LangChain/OpenAI). If no API key or failure -> fallback.
def parse_jd_with_llm(raw_text: str, title: str = "") -> Dict:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        return _heuristic_parse_jd(raw_text, title)

    # try to import LangChain & OpenAI; if not available fall back
    try:
        from langchain.chat_models import ChatOpenAI
        from langchain.prompts import ChatPromptTemplate
    except Exception:
        return _heuristic_parse_jd(raw_text, title)

    prompt = """Extract structured JSON from the job description. Return valid JSON with keys:
    title (string), must_have (list of short strings), good_to_have (list), qualifications (list), experience (string).
    JD:
    {jd}
    """

    try:
        llm = ChatOpenAI(temperature=0, openai_api_key=api_key, model="gpt-4o-mini")
        template = ChatPromptTemplate.from_template(prompt)
        chain = llm.generate([template.format_prompt(jd=raw_text).to_messages()])  # safe call
        # chain output text:
        out_text = chain.generations[0][0].text if chain.generations and chain.generations[0] else ""
        # extract JSON substring
        m = re.search(r'(\{.*\})', out_text, flags=re.S)
        if m:
            parsed = json.loads(m.group(1))
            # normalize keys
            return {
                "title": parsed.get("title", title),
                "raw_text": raw_text,
                "must_have": parsed.get("must_have", []) or [],
                "good_to_have": parsed.get("good_to_have", []) or [],
                "qualifications": parsed.get("qualifications", []) or [],
                "experience": parsed.get("experience", "") or ""
            }
    except Exception:
        pass

    return _heuristic_parse_jd(raw_text, title)
