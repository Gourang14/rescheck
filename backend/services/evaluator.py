# backend/services/evaluator.py
from typing import Dict, List
import os
from backend.services import groq_client
from backend.services import nlp_utils  # your existing local scorer (fallback)
import json

EVAL_PROMPT = """
You are a hiring assistant that MUST return a strict JSON object (no commentary) with the following keys:
- hard (number 0-100): hard-match score based on exact/fuzzy keyword coverage
- soft (number 0-100): semantic fit score (0-100) representing how well resume content matches the JD overall
- final (number 0-100): weighted final score (use weights provided below)
- verdict (string): one of "High", "Medium", "Low"
- missing (object): { "must": [ ... ], "good": [ ... ] } - lists of missing skills from must_have and good_to_have
- feedback (array of short strings): 3 to 6 concise actionable suggestions to improve the resume for this JD

Inputs (below) include job_parsed JSON and the full resume text. Use weights: hard_weight and soft_weight (sum to 1). If a weight is missing, default to hard=0.6 soft=0.4. Output only JSON.

JD_PARSED:
{jd_parsed_json}

RESUME_TEXT (first 7000 characters):
\"\"\"{resume_excerpt}\"\"\"

WEIGHTS:
hard_weight: {hard_weight}
soft_weight: {soft_weight}

Constraints:
- Numbers must be between 0 and 100.
- Missing skill lists should include canonical short strings that appear in JD_parsed must_have/good_to_have if they are absent in the resume (do not invent skills).
- soft score should reflect semantic similarity (use your reasoning).
- Keep JSON valid and compact.
"""

def evaluate_with_groq(jd_parsed: Dict, resume_text: str, hard_weight: float = 0.6, soft_weight: float = 0.4, model_url: str = None) -> Dict:
    # prepare prompt
    jd_json = json.dumps(jd_parsed, ensure_ascii=False)
    resume_excerpt = resume_text[:7000]
    prompt = EVAL_PROMPT.format(jd_parsed_json=jd_json, resume_excerpt=resume_excerpt, hard_weight=hard_weight, soft_weight=soft_weight)
    try:
        parsed = groq_client.groq_generate_json(prompt, max_tokens=800, temperature=0.0, model_url=model_url)
        # validate keys
        for k in ("hard","soft","final","verdict","missing","feedback"):
            if k not in parsed:
                raise RuntimeError(f"Missing key {k} in Groq output")
        # ensure numeric ranges
        parsed["hard"] = float(parsed["hard"])
        parsed["soft"] = float(parsed["soft"])
        parsed["final"] = float(parsed["final"])
        return parsed
    except Exception as e:
        # fallback to local approach
        try:
            # nlp_utils.final_score or similar function you already have
            return nlp_utils.fallback_final_score(resume_text, jd_parsed, hard_weight=hard_weight, soft_weight=soft_weight)
        except Exception:
            # minimal fallback
            return {
                "hard": 0.0,
                "soft": 0.0,
                "final": 0.0,
                "verdict": "Low",
                "missing": {"must": jd_parsed.get("must_have", []), "good": jd_parsed.get("good_to_have", [])},
                "feedback": ["No feedback available (fallback)."]
            }
