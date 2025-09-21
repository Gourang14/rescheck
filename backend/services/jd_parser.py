# backend/services/jd_parser.py
import re
from typing import List, Dict, Set

# connectors and company noise to trim or remove
CONNECTORS = [' from ', ' including ', ' that ', ' which ', ' and ', ' & ', ' , ', ';', ':', ' - ', ' – ', ' — ', ' to ', ' by ']
COMPANIES = ['palantir', 'mckinsey', 'quantumblack', 'axion', 'ray']  # lowercase
NOISE_WORDS = set([
    "apply","send","resume","our","we","team","since","largest","founding","deployed","onset",
    "job","role","description","overview","detailed","duration","internship","interns","work"
])
VERB_WORDS = set(["work","develop","design","build","use","deploy","create","partner","solve","revolutionize","accelerate"])

def normalize_token(tok: str) -> str:
    t = tok.strip()
    t = re.sub(r'^[\W_]+|[\W_]+$', '', t)  # trim punctuation edges
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def split_multi(s: str) -> List[str]:
    # split combined items like "NLP/LLMs" or "AI and NLP"
    parts = re.split(r'[\/\|,&]| and | & ', s, flags=re.I)
    return [p.strip() for p in parts if p.strip()]

def trim_after_connectors(s: str) -> str:
    s_low = s.lower()
    for c in CONNECTORS:
        if c.strip() and c in s_low:
            pos = s_low.find(c)
            s = s[:pos].strip()
            s_low = s.lower()
    # final punctuation trim
    return re.sub(r'^[\W_]+|[\W_]+$', '', s).strip()

def clean_candidate(s: str) -> str:
    s = s.replace("’", "'").replace("“","\"").replace("”","\"")
    s = re.sub(r'[\(\)\[\]]', '', s)
    s = normalize_token(s)
    s = trim_after_connectors(s)
    if not s:
        return ""
    low = s.lower()
    # remove company mentions
    for comp in COMPANIES:
        if comp in low:
            s = re.sub(re.escape(comp), '', s, flags=re.I).strip()
            low = s.lower()
    # drop noisy/verb-containing tokens
    if any(w in low for w in NOISE_WORDS):
        return ""
    if any(v in low for v in VERB_WORDS):
        return ""
    if len(s.split()) > 6:
        return ""
    # drop very short tokens
    if len(re.sub(r'[^A-Za-z0-9]', '', s)) <= 1:
        return ""
    return s

def initial_candidates(text: str) -> List[str]:
    cand = []
    # titlecase chunks (up to 4 words)
    cand += re.findall(r'\b(?:[A-Z][a-z0-9\+\#]{1,})(?:\s+[A-Z][a-z0-9\+\#]{1,}){0,3}\b', text)
    # acronyms like NLP, LLMs, AI
    cand += re.findall(r'\b[A-Z]{2,6}s?\b', text)
    # content inside parentheses
    for p in re.findall(r'\(([^)]+)\)', text):
        for part in re.split(r'[,\;/\|]', p):
            cand.append(part)
    # inline lists: "Must have: ..." etc.
    for m in re.findall(r'(?im)(?:must[-\s]*have|required)[:\-]\s*([^\n]*)', text):
        for part in re.split(r'[,\;/\|]', m):
            cand.append(part)
    for m in re.findall(r'(?im)(?:nice[-\s]*to[-\s]*have|good[-\s]*to[-\s]*have|preferred)[:\-]\s*([^\n]*)', text):
        for part in re.split(r'[,\;/\|]', m):
            cand.append(part)
    # also split long dash segments (short pieces may be tech)
    for part in re.split(r'[-–—]', text):
        if len(part.split()) <= 6:
            cand.append(part)
    return cand

def extract_skills_from_jd(raw_text: str) -> Dict:
    jd = raw_text or ""
    result = {"title": "", "must_have": [], "good_to_have": [], "qualifications": [], "experience": ""}

    # title: first non-empty line
    for line in jd.splitlines():
        if line.strip():
            result["title"] = line.strip()[:200]
            break

    # detect experience pattern if present
    m = re.search(r'(\d+\+?\s+years?|\d+-\d+\s+years|\d+\s+years? of experience)', jd, flags=re.I)
    if m:
        result["experience"] = m.group(0).strip()

    # build candidate pool
    raw = initial_candidates(jd)
    processed = []
    seen = set()
    for r in raw:
        for part in split_multi(r):
            cleaned = clean_candidate(part)
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            processed.append(cleaned)

    # find signal ngrams for likely tech phrases (robust catch of lower-case phrases)
    tokens = re.findall(r"[A-Za-z\+\#]{2,}", jd.lower())
    signal_phrases = set()
    signal_keywords = [
        "generative ai","generative","nlp","llm","llms","enterprise ai","computer vision",
        "electric vehicle","electric vehicles","medical device","medical devices","ai models","ai",
        "automotive","aerospace","supersonic","machine learning","deep learning"
    ]
    for n in range(1,4):
        for i in range(len(tokens)-n+1):
            seq = " ".join(tokens[i:i+n])
            for sig in signal_keywords:
                if sig in seq:
                    signal_phrases.add(sig)

    # canonicalize and format signal phrases
    final = []
    for s in sorted(signal_phrases, key=lambda x:(-len(x.split()), x)):
        if s.lower() in ["nlp","llm","llms","ai"]:
            final.append(s.upper())
        else:
            final.append(s.title())

    # merge processed items (titlecase / acronym results), with further cleaning
    for s in processed:
        s2 = trim_after_connectors(s)
        s2 = clean_candidate(s2)
        if not s2: continue
        # split "AI and NLP" -> ["AI","NLP"]
        for p in split_multi(s2):
            p_clean = clean_candidate(p)
            if not p_clean: continue
            # format acronyms nicely
            if p_clean.lower() in ["nlp","llm","llms","ai"]:
                p_clean = p_clean.upper()
            else:
                p_clean = p_clean.title() if len(p_clean.split())>1 else (p_clean.upper() if p_clean.isalpha() and len(p_clean)<=3 else p_clean.capitalize())
            if p_clean.lower() not in [f.lower() for f in final]:
                final.append(p_clean)

    # final dedupe preserving order
    seen2 = set(); final_clean = []
    for f in final:
        if f.lower() in seen2: continue
        seen2.add(f.lower()); final_clean.append(f)

    # heuristics: bucket must-have vs good-to-have by presence of core keywords
    must_keys = ["ai","generative","nlp","llm","machine learning","computer vision","electric","vehicle","medical","enterprise","models"]
    must = []; good = []
    for f in final_clean:
        low = f.lower()
        if any(k in low for k in must_keys):
            must.append(f)
        else:
            good.append(f)

    result["must_have"] = must[:15]
    result["good_to_have"] = good[:15]
    return result
