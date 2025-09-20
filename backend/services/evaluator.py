from typing import Dict, List, Tuple
import math
from rapidfuzz import fuzz
from backend.services.embeddings_manager import EmbeddingsProvider, SBERTEmbeddingsProvider, OpenAIEmbeddingsProvider

from backend.services import nlp_utils  # for skill extraction if needed

# Instantiate an embeddings provider (single global instance okay)
EMB_PROVIDER = EmbeddingsProvider()

def cosine_sim(a, b):
    import numpy as np
    a = np.array(a)
    b = np.array(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def hard_match_score(resume_text: str, must_have: List[str], good_to_have: List[str], threshold=75):
    """
    Hard match: calculates coverage percentage of must_have and good_to_have.
    Returns numeric hard_score (0-100) and missing lists.
    """
    # use fuzzy matching for skill presence
    found_must = []
    for skill in must_have:
        score = fuzz.partial_ratio(skill.lower(), resume_text.lower())
        if score >= threshold:
            found_must.append(skill)

    found_good = []
    for skill in good_to_have:
        score = fuzz.partial_ratio(skill.lower(), resume_text.lower())
        if score >= threshold:
            found_good.append(skill)

    must_cov = (len(found_must) / len(must_have)) if must_have else 1.0
    good_cov = (len(found_good) / len(good_to_have)) if good_to_have else 1.0

    # Compose hard score: 70% must, 30% good
    hard_score = (0.7 * must_cov + 0.3 * good_cov) * 100

    missing_must = [m for m in must_have if m not in found_must]
    missing_good = [g for g in good_to_have if g not in found_good]

    return round(hard_score, 2), missing_must, missing_good

def soft_match_score(jd_text: str, resume_text: str):
    """
    Compute embedding-based cosine similarity (0-100)
    """
    # embed
    jd_vec = EMB_PROVIDER.embed(jd_text)[0]
    res_vec = EMB_PROVIDER.embed(resume_text)[0]
    sim = cosine_sim(jd_vec, res_vec)
    return round(sim * 100, 2)

def final_score(resume_text: str, jd_parsed: Dict, weights: Dict = None):
    """
    Combine hard and soft into a final score.
    jd_parsed: contains keys must_have, good_to_have, raw_text
    weights: {'hard': 0.6, 'soft': 0.4}
    """
    if weights is None:
        weights = {'hard': 0.6, 'soft': 0.4}

    hard, missing_must, missing_good = hard_match_score(resume_text, jd_parsed.get("must_have", []), jd_parsed.get("good_to_have", []))
    soft = soft_match_score(jd_parsed.get("raw_text", ""), resume_text)

    combined = round(weights['hard'] * hard + weights['soft'] * soft, 2)
    # verdict
    if combined >= 75:
        verdict = "High"
    elif combined >= 50:
        verdict = "Medium"
    else:
        verdict = "Low"

    missing = {"must": missing_must, "good": missing_good}
    breakdown = {"hard": hard, "soft": soft, "final": combined}

    return {"score": combined, "verdict": verdict, "missing": missing, "breakdown": breakdown}
