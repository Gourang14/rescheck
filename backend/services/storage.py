# backend/services/storage.py
from backend.db.models import Evaluation, SessionLocal, init_db

# ensure tables exist
init_db()

def save_evaluation(resume_name: str, eval_data: dict):
    """
    Save evaluation result into DB.
    """
    db = SessionLocal()
    try:
        result = Evaluation(
            job_id=None,  # optional unless you map jobs
            resume_id=None,  # optional unless you map resumes
            result={
                "resume_name": resume_name,
                "jd_title": eval_data.get("jd_title", "Unknown Role"),
                "relevance_score": eval_data.get("relevance_score", 0),
                "semantic_score": eval_data.get("semantic_score", 0),
                "hard_score": eval_data.get("hard_score", 0),
                "verdict": eval_data.get("verdict", "Low"),
                "missing_skills": eval_data.get("missing_skills", []),
            },
        )
        db.add(result)
        db.commit()
        db.refresh(result)
        return result
    finally:
        db.close()


def get_all_evaluations():
    """
    Fetch all stored evaluations.
    """
    db = SessionLocal()
    try:
        return db.query(Evaluation).all()
    finally:
        db.close()
