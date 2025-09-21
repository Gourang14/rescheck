# backend/services/jd_parser.py
import re
from typing import List, Dict

# noise tokens to filter out obvious junk words/phrases
NOISE_TOKENS = set([
    "job","role","description","overview","apply","apply now","detailed","duration",
    "internship","interns","walk","bond","work","attendance","students","position","the","and","or",
    "responsibilities","responsibility","responsible","will","years","experience","apply","send","resume"
])

# common alias mappings – extend this to canonicalize common names
ALIASES = {
    "tensorflow": ["tensor flow","tf"],
    "pytorch": ["py torch","torch"],
    "c++": ["cpp"],
    "c#": ["csharp"],
    "machine learning": ["ml"],
    "deep learning": ["dl"],
    "natural language processing": ["nlp"],
    "computer vision": ["cv"],
    "spark": ["apache spark"],
    "tableau": ["tableau desktop"],
    "sql": ["structured query language"]
}

# filter words that indicate a phrase is actually a verb/clause rather than a skill
VERB_FILTER = set(["work","works","will","develop","developing","design","building","build","using","use","experience","lead","leads"])

def normalize_skill(tok: str) -> str:
    s = tok.strip()
    s = re.sub(r'[\s\-/]+', ' ', s)                 # unify separators to spaces
    s = re.sub(r'[^\w\+\#\. ]', '', s)             # remove unwanted punctuation (keep +,#,.)
    s = s.strip()
    if not s:
        return ""
    s_low = s.lower()
    # canonicalize common aliases
    for canon, aliases in ALIASES.items():
        if s_low == canon: return canon
        for a in aliases:
            if s_low == a: return canon
    return s

def filter_noise(skills: List[str]) -> List[str]:
    out = []
    seen = set()
    for s in skills:
        if not s: continue
        s_norm = normalize_skill(s)
        if not s_norm: continue
        # basic filters
        if len(s_norm) <= 1: continue
        if any(tok in s_norm.lower() for tok in NOISE_TOKENS): continue
        if len(s_norm.split()) > 5: continue                 # drop very long phrases
        if any(v in s_norm.lower().split() for v in VERB_FILTER): continue
        if re.fullmatch(r'\d+(\.\d+)?', s_norm): continue
        if s_norm.lower() in seen: continue
        seen.add(s_norm.lower())
        out.append(s_norm)
    return out

def split_candidates(text: str) -> List[str]:
    # split on commas, bullets, semicolons, 'and', '/', pipes etc.
    parts = re.split(r'[,\n;\u2022•\t]| and |/|\|', text)
    tokens = [p.strip() for p in parts if p and p.strip()]
    return tokens

def extract_section_lists_block(raw: str, headings: List[str]) -> Dict[str,str]:
    """
    Find heading lines (e.g. "Skills:", "Must have:" or standalone headings)
    and collect the content following that heading until next heading/blank.
    """
    text = raw.replace('\r','\n')
    out = {}
    lines = text.splitlines()
    lower_headings = [h.lower() for h in headings]
    for i, line in enumerate(lines):
        low = line.strip().lower()
        for h in lower_headings:
            if low == h or low.startswith(h + ":") or low.startswith(h + " -") or low.startswith(h + " —"):
                content = ""
                # inline after colon on same line?
                if ":" in line:
                    content = line.split(":",1)[1].strip()
                j = i+1
                while j < len(lines) and lines[j].strip() and not any(lines[j].strip().lower().startswith(x) for x in lower_headings):
                    content += ("\n" + lines[j]).strip()
                    j += 1
                out[h] = content.strip()
    return out

def extract_inline_sections(raw: str) -> Dict[str,str]:
    """
    Capture inline patterns like 'Must have: Python, SQL' that appear in the middle of a paragraph/line.
    Returns dict keys 'must_have', 'good_to_have', 'qualifications' if matched.
    """
    out = {}
    patterns = {
        "must_have": r'(?im)\b(?:must[-\s]*have|required|requirements?)\s*[:\-–]\s*(.+)',
        "good_to_have": r'(?im)\b(?:good[-\s]*to[-\s]*have|nice[-\s]*to[-\s]*have|preferred)\s*[:\-–]\s*(.+)',
        "qualifications": r'(?im)\b(?:qualification|qualifications|education)\s*[:\-–]\s*(.+)',
    }
    for k,p in patterns.items():
        matches = re.findall(p, raw)
        if matches:
            out[k] = " \n ".join(matches)
    return out

def parse_jd(raw_text: str) -> Dict:
    """
    Main improved parser. Returns:
      { title, must_have, good_to_have, qualifications, experience }
    - Works with inline "Must have: ..." and block-style headings.
    - Filters noise tokens and deduplicates.
    """
    jd = raw_text or ""
    res = {"title": "", "must_have": [], "good_to_have": [], "qualifications": [], "experience": ""}

    # title: first non-empty line
    for line in jd.splitlines():
        if line.strip():
            res["title"] = line.strip()[:200]
            break

    # experience detection (e.g. "1-3 years", "3+ years of experience")
    m = re.search(r'(\d+\+?\s+years?|\d+-\d+\s+years|\d+\s+years? of experience)', jd, flags=re.I)
    if m:
        res["experience"] = m.group(0).strip()

    # 1) inline patterns ("Must have: ...")
    inline = extract_inline_sections(jd)
    if "must_have" in inline:
        res["must_have"].extend(split_candidates(inline["must_have"]))
    if "good_to_have" in inline:
        res["good_to_have"].extend(split_candidates(inline["good_to_have"]))
    if "qualifications" in inline:
        res["qualifications"].extend(split_candidates(inline["qualifications"]))

    # 2) block headings (lines that are headings)
    headings = ["must have","must-have","required","requirements","skills","skillset","good to have","nice to have","preferred","qualification","qualifications","education","experience"]
    block = extract_section_lists_block(jd, headings)
    for h,content in block.items():
        key = None
        if h in ("must have","must-have","required","requirements","skills","skillset"):
            key = "must_have"
        elif h in ("good to have","nice to have","preferred"):
            key = "good_to_have"
        elif h in ("qualification","qualifications","education"):
            key = "qualifications"
        if key:
            tokens = split_candidates(content)
            res[key].extend(tokens)

    # 3) fallback: if no must_have found, look for likely list lines
    if not res["must_have"]:
        candidate_tokens = []
        for line in jd.splitlines():
            line = line.strip()
            if not line: continue
            # skip very long descriptive lines (likely responsibilities)
            if len(line.split()) > 20: continue
            if ',' in line or ';' in line or '/' in line or ' and ' in line:
                candidate_tokens.extend(split_candidates(line))
            else:
                caps = re.findall(r'\b[A-Z][A-Za-z0-9\+\#\.]{1,}\b', line)
                if len(caps) >= 2:
                    candidate_tokens.extend(split_candidates(line))
        res["must_have"].extend(candidate_tokens[:60])

    # Filter & dedupe
    res["must_have"] = filter_noise(res["must_have"])
    res["good_to_have"] = filter_noise(res["good_to_have"])
    res["qualifications"] = filter_noise(res["qualifications"])

    # remove duplicates between must and good
    must_set = set(m.lower() for m in res["must_have"])
    res["good_to_have"] = [g for g in res["good_to_have"] if g.lower() not in must_set]

    # final fallback: frequency-based pick if must still empty
    if not res["must_have"]:
        tokens = re.findall(r'\b[A-Za-z\+\#\.]{2,}\b', jd)
        freq = {}
        for t in tokens:
            tl = t.lower()
            if tl in NOISE_TOKENS: continue
            freq[tl] = freq.get(tl,0) + 1
        sorted_tokens = [t for t,_ in sorted(freq.items(), key=lambda x: x[1], reverse=True)]
        res["must_have"].extend(filter_noise(sorted_tokens[:20]))

    return res
