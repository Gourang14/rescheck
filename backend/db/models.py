# backend/db/models.py
from sqlalchemy import Column, Integer, String, Float, Text
from backend.db.database import Base


class EvaluationResult(Base):
    __tablename__ = "evaluation_results"

    id = Column(Integer, primary_key=True, index=True)
    resume_name = Column(String, nullable=False)
    jd_title = Column(String, nullable=False)
    relevance_score = Column(Float, nullable=False)
    semantic_score = Column(Float, nullable=False)
    hard_score = Column(Float, nullable=False)
    verdict = Column(String, nullable=False)
    missing_skills = Column(Text)  # store as comma-separated string