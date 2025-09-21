# backend/db/models.py
from sqlalchemy import Column, Integer, String, Text, JSON, create_engine, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import datetime
import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///rescheck.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class Job(Base):
    __tablename__ = "jobs"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    raw_text = Column(Text)
    parsed = Column(JSON)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class Resume(Base):
    __tablename__ = "resumes"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    raw_text = Column(Text)
    sections = Column(JSON)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class Evaluation(Base):
    __tablename__ = "evaluations"
    id = Column(Integer, primary_key=True, index=True)
    resume_name = Column(String, index=True)
    jd_title = Column(String, default="Unknown Role")
    relevance_score = Column(Integer, default=0)
    semantic_score = Column(Integer, default=0)
    hard_score = Column(Integer, default=0)
    verdict = Column(String, default="Low")
    missing_skills = Column(Text)  # store as comma-separated
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


def init_db():
    Base.metadata.create_all(bind=engine)
