# # backend/services/jd_parser.py
# import re
# from typing import List, Dict, Optional, Tuple, Set

# # optional: fuzzy matching to canonical skill vocab
# try:
#     from rapidfuzz import process, fuzz
#     _HAS_RAPIDFUZZ = True
# except Exception:
#     _HAS_RAPIDFUZZ = False

# # optional: spaCy for NER (company removal)
# try:
#     import spacy
#     _HAS_SPACY = True
#     _SPACY_NLP = spacy.load("en_core_web_sm")
# except Exception:
#     _HAS_SPACY = False
#     _SPACY_NLP = None

# # tokens we consider noise (extend as you see patterns)
# NOISE_TOKENS = {
#     "job","role","description","overview","apply","apply now","detailed","duration",
#     "internship","interns","walk","bond","work","attendance","students","position",
#     "the","and","or","what","who","can","has","experience","understand","will","our","we"
# }

# # connectors to trim trailing context (e.g. "AI models that ...")
# CONNECTORS = [' including ', ' from ', ' that ', ' which ', ' such as ', ' for ', ' by ', ' to ', ' in ', ' with ']

# # helper: split lists into tokens
# _SPLIT_RE = re.compile(r'[,\n;\u2022•\t/|]| and | & ' , flags=re.I)

# def _split_candidates(text: str) -> List[str]:
#     parts = [p.strip() for p in _SPLIT_RE.split(text) if p and p.strip()]
#     # also split hyphen/paren groups if short
#     out = []
#     for p in parts:
#         if '-' in p and len(p.split()) <= 6:
#             out += [pp.strip() for pp in re.split(r'[-–—]', p) if pp.strip()]
#         else:
#             out.append(p)
#     return out

# def _trim_after_connectors(s: str) -> str:
#     s_low = s.lower()
#     for c in CONNECTORS:
#         if c.strip() and c in s_low:
#             pos = s_low.find(c)
#             s = s[:pos]
#             s_low = s.lower()
#     return s.strip(" ,;:-")

# def _clean_phrase(s: str) -> str:
#     if not s:
#         return ""
#     s = s.replace("’", "'").replace("“", '"').replace("”", '"')
#     s = re.sub(r'[\(\)\[\]\{\}]', '', s)  # drop brackets
#     s = s.strip()
#     s = _trim_after_connectors(s)
#     s = re.sub(r'\s+', ' ', s).strip()
#     # drop stray short tokens, punctuation-only
#     if len(re.sub(r'[^A-Za-z0-9\+\#]', '', s)) <= 1:
#         return ""
#     return s

# def _remove_companies(text: str) -> Set[str]:
#     """Return set of company/org names detected by spaCy (lowercased)."""
#     if not _HAS_SPACY or not _SPACY_NLP:
#         return set()
#     doc = _SPACY_NLP(text)
#     return {ent.text.lower() for ent in doc.ents if ent.label_ in ("ORG","PRODUCT")}

# def _extract_inline_sections(text: str) -> Dict[str,str]:
#     patterns = {
#         "must_have": r'(?im)(?:must[-\s]*have|required|requirements?)\s*[:\-–]\s*([^\n]+)',
#         "good_to_have": r'(?im)(?:nice[-\s]*to[-\s]*have|good[-\s]*to[-\s]*have|preferred)\s*[:\-–]\s*([^\n]+)',
#         "qualifications": r'(?im)(?:qualification|qualifications|education)\s*[:\-–]\s*([^\n]+)',
#     }
#     found = {}
#     for key, pat in patterns.items():
#         m = re.findall(pat, text)
#         if m:
#             found[key] = " ; ".join(m)
#     return found

# def _extract_block_sections(text: str, headings: List[str]) -> Dict[str,str]:
#     """
#     Collect content under lines that are headings like 'Skills:' or 'Must have:'.
#     Returns mapping heading_lower -> content string.
#     """
#     out = {}
#     lines = text.splitlines()
#     low_headings = [h.lower() for h in headings]
#     for i, line in enumerate(lines):
#         l = line.strip()
#         if not l:
#             continue
#         low = l.lower().rstrip(':').strip()
#         if any(low == hh or low.startswith(hh + ':') for hh in low_headings):
#             # collect until blank or next heading
#             content = ""
#             # inline portion after colon
#             if ':' in l:
#                 after = l.split(':', 1)[1].strip()
#                 if after:
#                     content = after
#             j = i + 1
#             while j < len(lines) and lines[j].strip() and not any(lines[j].strip().lower().rstrip(':').startswith(h) for h in low_headings):
#                 content += ("\n" + lines[j]).strip()
#                 j += 1
#             out[low] = content.strip()
#     return out

# def _fuzzy_map_token(token: str, vocab: List[str], score_cutoff: int = 78) -> Optional[str]:
#     """Return best match from vocab using rapidfuzz or None if no good match."""
#     if not _HAS_RAPIDFUZZ or not vocab:
#         return None
#     match = process.extractOne(token, vocab, scorer=fuzz.token_sort_ratio)
#     if match and match[1] >= score_cutoff:
#         return match[0]
#     return None

# def parse_jd(
#     raw_text: str,
#     title_hint: Optional[str] = None,
#     canonical_skill_vocab: Optional[List[str]] = None,
#     fuzz_threshold: int = 78
# ) -> Dict:
#     """
#     Returns dict:
#       { title, must_have: [canonical or cleaned tokens], good_to_have: [...], qualifications: [...], experience: "" }
#     - canonical_skill_vocab: optional list of canonical skills (best: 1k common skills). If provided, tokens will be mapped to canonicals.
#     """
#     if not raw_text:
#         return {"title": title_hint or "", "must_have": [], "good_to_have": [], "qualifications": [], "experience": ""}

#     text = raw_text.replace("\r", "\n")
#     companies = _remove_companies(text)

#     # title: prefer explicit or first non-empty line
#     title = title_hint or ""
#     if not title:
#         for line in text.splitlines():
#             if line.strip():
#                 title = line.strip()[:200]
#                 break

#     # experience
#     m = re.search(r'(\d+\+?\s+years?|\d+-\d+\s+years|\d+\s+years? of experience)', text, flags=re.I)
#     experience = m.group(0).strip() if m else ""

#     # block headings + inline
#     headings = ["must have","must-have","required","requirements","skills","skillset","good to have","nice to have","preferred","qualification","qualifications","education","experience"]
#     block = _extract_block_sections(text, headings)
#     inline = _extract_inline_sections(text)

#     candidates = []
#     # prefer explicit cited tokens
#     for k in ("must have","must-have","required","requirements","skills","skillset"):
#         if k in block:
#             candidates += _split_candidates(block[k])
#     if "must_have" in inline:
#         candidates += _split_candidates(inline["must_have"])

#     # nice-to-have
#     nice_candidates = []
#     for k in ("good to have","nice to have","preferred"):
#         if k in block:
#             nice_candidates += _split_candidates(block[k])
#     if "good_to_have" in inline:
#         nice_candidates += _split_candidates(inline["good_to_have"])

#     # qualifications
#     quals = []
#     if "qualification" in block or "qualifications" in block or "education" in block:
#         for key in ("qualification","qualifications","education"):
#             if key in block:
#                 quals += _split_candidates(block[key])
#     if "qualifications" in inline:
#         quals += _split_candidates(inline["qualifications"])

#     # If no explicit 'must' tokens found, scan whole text lines for likely lists
#     if not candidates:
#         for line in text.splitlines():
#             line = line.strip()
#             if not line: continue
#             if len(line.split()) > 20: continue  # skip long descriptive lines
#             if ',' in line or ';' in line or '/' in line or ' and ' in line:
#                 candidates += _split_candidates(line)

#     # normalize and clean candidates
#     def map_and_clean(lst: List[str]) -> List[str]:
#         out = []
#         seen = set()
#         for raw in lst:
#             raw = _clean_phrase(raw)
#             if not raw: continue
#             low = raw.lower()
#             # drop if a company detected by spaCy or obviously a company token
#             if any(comp in low for comp in companies): 
#                 continue
#             # drop if token is common noise
#             if any(nt in low for nt in NOISE_TOKENS):
#                 continue
#             # split combined tokens like "NLP/LLMs" -> entries
#             parts = _split_candidates(raw) if ('/' in raw or ' and ' in raw or ',' in raw) else [raw]
#             for p in parts:
#                 p = _clean_phrase(p)
#                 if not p: continue
#                 lowp = p.lower()
#                 if any(nt == lowp for nt in NOISE_TOKENS): continue
#                 # optional canonical mapping
#                 canon = None
#                 if canonical_skill_vocab and _HAS_RAPIDFUZZ:
#                     canon = _fuzzy_map_token(p, canonical_skill_vocab, fuzz_threshold)
#                 final_token = canon or p
#                 key = final_token.lower()
#                 if key in seen: continue
#                 seen.add(key)
#                 out.append(final_token)
#         return out

#     must_list = map_and_clean(candidates)
#     nice_list = map_and_clean(nice_candidates)
#     quals_list = map_and_clean(quals)

#     # final hygiene: if some nice items overlap must, remove them from nice
#     must_set = set([m.lower() for m in must_list])
#     nice_list = [n for n in nice_list if n.lower() not in must_set]

#     return {
#         "title": title,
#         "must_have": must_list,
#         "good_to_have": nice_list,
#         "qualifications": quals_list,
#         "experience": experience
#     }






# backend/services/jd_parser.py
import os
from typing import Dict
from . import jd_heuristic  # your existing heuristic parser module (rename to match your repo)
from backend.services import groq_client

# Strict JSON extraction prompt template
JD_PROMPT = """
You are an automated parser. Given the job description text below, extract a JSON object with exactly these keys:
- title (string)
- must_have (array of short strings)  -- skills or technologies explicitly required
- good_to_have (array of short strings) -- optional skills
- qualifications (array of short strings)
- experience (string) e.g. "1-3 years" or empty string if not present

Job description:
\"\"\"{jd}\"\"\"

Return only valid JSON (no explanation). Keep skill names short (1-4 words), prefer canonical forms (e.g., "Python", "NLP", "PyTorch", "TensorFlow", "Spark").
"""

def parse_jd_with_groq(jd_text: str, model_url: str = None) -> Dict:
    # try LLM-based extraction if key/url provided
    try:
        parsed = groq_client.groq_generate_json(JD_PROMPT.format(jd=jd_text), max_tokens=800, temperature=0.0, model_url=model_url)
        # ensure keys exist and are lists/strings
        r = {
            "title": parsed.get("title","") if isinstance(parsed.get("title",""), str) else "",
            "must_have": parsed.get("must_have", []) if isinstance(parsed.get("must_have", []), list) else [],
            "good_to_have": parsed.get("good_to_have", []) if isinstance(parsed.get("good_to_have", []), list) else [],
            "qualifications": parsed.get("qualifications", []) if isinstance(parsed.get("qualifications", []), list) else [],
            "experience": parsed.get("experience","") if isinstance(parsed.get("experience",""), str) else "",
            "raw_text": jd_text
        }
        return r
    except Exception as e:
        # fallback to heuristic parser you already have
        try:
            return jd_heuristic.extract_skills_from_jd_improved(jd_text)
        except Exception:
            # last-resort simple structure
            return {"title": jd_text.splitlines()[0] if jd_text else "", "must_have": [], "good_to_have": [], "qualifications": [], "experience": "", "raw_text": jd_text}
