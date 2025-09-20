# backend/services/evaluator.py
from typing import Dict, List
import numpy as np
from rapidfuzz import fuzz
from backend.services.embeddings_manager import SBERTEmbeddingsProvider, OpenAIEmbeddingsProvider, EmbeddingsProvider

# instantiate embeddings provider (prefer OpenAI if configured)
def get_default_embeddings_provider():
    try:
        # try OpenAI provider if api key present
        import os
        if os.getenv("OPENAI_API_KEY"):
            return OpenAIEmbeddingsProvider(os.getenv("OPENAI_API_KEY"))
    except Exception:
        pass
    # fallback
    return SBERTEmbeddingsProvider()

EMB_PROVIDER = get_default_embeddings_provider()

def cosine_sim(a, b):
    a = np.array(a); b = np.array(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom != 0 else 0.0

def hard_match_score(resume_text: str, must_have: List[str], good_to_have: List[str], threshold: int = 70):
    found_must = []
    for s in must_have:
        if fuzz.partial_ratio(s.lower(), resume_text.lower()) >= threshold:
            found_must.append(s)
    found_good = []
    for s in good_to_have:
        if fuzz.partial_ratio(s.lower(), resume_text.lower()) >= threshold:
            found_good.append(s)

    must_cov = (len(found_must) / len(must_have)) if must_have else 1.0
    good_cov = (len(found_good) / len(good_to_have)) if good_to_have else 1.0
    hard_score = (0.75 * must_cov + 0.25 * good_cov) * 100
    return round(hard_score, 2), [m for m in must_have if m not in found_must], [g for g in good_to_have if g not in found_good]

def soft_match_score(jd_text: str, resume_text: str):
    jd_vec = EMB_PROVIDER.embed([jd_text])[0]
    res_vec = EMB_PROVIDER.embed([resume_text])[0]
    return round(cosine_sim(jd_vec, res_vec) * 100, 2)

def final_score(resume_text: str, jd_parsed: Dict, weights: Dict = None):
    # weights can be provided per-job or default
    if weights is None:
        weights = {"hard": 0.6, "soft": 0.4}
    hard, missing_must, missing_good = hard_match_score(resume_text, jd_parsed.get("must_have", []), jd_parsed.get("good_to_have", []))
    soft = soft_match_score(jd_parsed.get("raw_text", ""), resume_text)
    final = round(weights["hard"] * hard + weights["soft"] * soft, 2)
    verdict = "High" if final >= 75 else "Medium" if final >= 50 else "Low"
    return {
        "final": final,
        "verdict": verdict,
        "hard": hard,
        "soft": soft,
        "missing": {"must": missing_must, "good": missing_good},
        "weights": weights
    }
