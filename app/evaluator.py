import os
import re
import math
from rapidfuzz import fuzz
from openai import OpenAI

# initialize client (expects OPENAI_API_KEY in env)
_api_key = os.getenv("OPENAI_API_KEY")
if _api_key:
    client = OpenAI(api_key=_api_key)
else:
    client = None

# --- Text extraction helpers (used by main) ---
import pdfplumber
import docx2txt

def extract_text_from_pdf(path):
    text_pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ""
            text_pages.append(txt)
    return "\n".join(text_pages)

def extract_text_from_docx(path):
    return docx2txt.process(path)

def normalize_text(txt):
    txt = txt.replace('\r', '\n')
    txt = re.sub(r'\n{2,}', '\n', txt)
    return txt.strip()

# --- simple JD parser ---

def parse_jd_text(raw_text, title=None):
    """Return a dict: title, raw_text, must_have[], good_to_have[], education (string)
    Uses heuristics; for production replace with an LLM-assisted parser.
    """
    t = title or ""
    text = raw_text.lower()
    # heuristics: find lines containing 'must' or 'required' or 'responsibilities'
    must = []
    good = []
    education = ""

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for i, line in enumerate(lines):
        if any(k in line for k in ("must have", "must", "required", "requirements")):
            # collect subsequent 5 lines as likely list
            for s in lines[i+1:i+8]:
                if len(s.split()) <= 1:
                    continue
                must.append(s.strip("* -"))
        if any(k in line for k in ("nice to have", "good to have", "preferred")):
            for s in lines[i+1:i+8]:
                if len(s.split()) <= 1:
                    continue
                good.append(s.strip("* -"))
        if "education" in line or "qualification" in line:
            education = line
    # fallback: try pickup keywords from title and intro
    if not must:
        # naive: pick technologies from first 10 lines using small skill lexicon
        for l in lines[:10]:
            for w in _MASTER_SKILLS:
                if w in l:
                    if w not in must:
                        must.append(w)
    return {
        "title": t,
        "raw_text": raw_text,
        "must_have": list(dict.fromkeys(must)),
        "good_to_have": list(dict.fromkeys(good)),
        "education": education
    }

# --- section splitter & contact extraction ---
HEADINGS = ['education', 'experience', 'skills', 'projects', 'certifications', 'summary']

def split_sections(text):
    sections = {}
    lines = text.splitlines()
    current = 'general'
    sections[current] = []
    for line in lines:
        l = line.strip()
        if not l:
            continue
        low = l.lower()
        if any(low.startswith(h) for h in HEADINGS):
            current = low.split()[0]
            sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(l)
    for k in list(sections.keys()):
        sections[k] = '\n'.join(sections[k]).strip()
    return sections

def extract_email(text):
    m = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    return m.group(0) if m else None

def extract_phone(text):
    m = re.search(r'(\+?\d[\d\s\-]{6,}\d)', text)
    return m.group(0) if m else None

# --- skill matching ---
_MASTER_SKILLS = [
    "python","java","sql","c++","c#","machine learning","deep learning",
    "tensorflow","pytorch","pandas","numpy","scikit-learn","react","node.js",
    "docker","kubernetes","aws","azure","git","nlp","computer vision"
]

from rapidfuzz import fuzz

def find_skills_in_text(text, thresh=70):
    txt = text.lower()
    found = []
    for skill in _MASTER_SKILLS:
        score = fuzz.partial_ratio(skill.lower(), txt)
        if score >= thresh:
            found.append((skill, int(score)))
    return sorted(found, key=lambda x: -x[1])

# --- embeddings ---

def get_embedding_openai(text):
    if not client:
        raise RuntimeError("OpenAI API key not set. Set OPENAI_API_KEY environment variable or swap for local embeddings.")
    # guard: shorten text if extremely long (simple truncation for MVP)
    if len(text) > 20000:
        text = text[:20000]
    resp = client.embeddings.create(model="text-embedding-3-small", input=text)
    return resp.data[0].embedding

def cosine_sim(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(x*x for x in b))
    return dot / (na*nb) if na and nb else 0.0

# --- scoring ---

def score_resume_against_jd(resume_sections, jd_parsed):
    # jd_parsed: dict with must_have, good_to_have, education, raw_text
    must = jd_parsed.get("must_have", []) or []
    good = jd_parsed.get("good_to_have", []) or []

    # find which must skills exist in resume (simple fuzzy across sections)
    found_must = []
    resume_text = "\n".join((resume_sections or {}).values())
    for m in must:
        m_low = m.lower()
        # check against resume text directly
        if fuzz.partial_ratio(m_low, resume_text.lower()) >= 75:
            found_must.append(m)
    must_coverage = (len(found_must) / len(must)) if must else 1.0

    # education
    edu_req = (jd_parsed.get("education") or "").lower()
    edu_text = (resume_sections.get("education","") or "").lower()
    edu_match = 1.0 if edu_req and edu_req in edu_text else (1.0 if not edu_req else 0.0)

    # good-to-have coverage
    found_good = 0
    for g in good:
        if fuzz.partial_ratio(g.lower(), resume_text.lower()) >= 70:
            found_good += 1
    keyword_coverage = (found_good / len(good)) if good else 1.0

    # hard score composition
    hard_comp = must_coverage*0.7 + edu_match*0.2 + keyword_coverage*0.1
    hard_score = hard_comp * 100

    # soft score via embeddings
    jd_text = jd_parsed.get("raw_text", "")
    try:
        emb_jd = get_embedding_openai(jd_text)
        emb_res = get_embedding_openai(resume_text)
        soft_cos = cosine_sim(emb_jd, emb_res)
        soft_score = max(0.0, soft_cos) * 100
    except Exception as e:
        # if embeddings unavailable, fallback to keyword heuristic
        soft_score = keyword_coverage * 100

    final_score = 0.6 * hard_score + 0.4 * soft_score

    if final_score >= 75:
        verdict = "High"
    elif final_score >= 50:
        verdict = "Medium"
    else:
        verdict = "Low"

    missing_skills = [m for m in must if m not in found_must]

    return {
        "score": round(final_score, 2),
        "hard_score": round(hard_score, 2),
        "soft_score": round(soft_score, 2),
        "verdict": verdict,
        "missing_skills": missing_skills,
        "found_must": found_must,
    }