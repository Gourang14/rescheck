# app/main.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import os, tempfile

from app.models import init_db, SessionLocal, Job, Resume, Evaluation
from backend.services.parser import extract_text_from_pdf, extract_text_from_docx
from backend.services.jd_parser import parse_jd_with_llm
from backend.services.evaluator import final_score
from backend.services.nlp_utils import section_split
from backend.services.feedback import generate_feedback

init_db()
app = FastAPI(title="Resume Relevance API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def save_temp(upload: UploadFile):
    suffix = os.path.splitext(upload.filename)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(upload.file.read())
    tmp.close()
    return tmp.name

@app.post("/jd/upload")
async def upload_jd(file: UploadFile = File(...), title: str = Form("")):
    path = save_temp(file)
    try:
        if file.filename.lower().endswith(".pdf"):
            raw = extract_text_from_pdf(path)
        elif file.filename.lower().endswith(".docx"):
            raw = extract_text_from_docx(path)
        else:
            raw = open(path, "r", encoding="utf-8", errors="ignore").read()
    finally:
        os.unlink(path)
    parsed = parse_jd_with_llm(raw, title)
    db: Session = SessionLocal()
    job = Job(title=title or parsed.get("title",""), raw_text=raw, parsed=parsed)
    db.add(job); db.commit(); db.refresh(job); db.close()
    return {"job_id": job.id, "parsed": parsed}

@app.post("/resume/upload")
async def upload_resume(file: UploadFile = File(...)):
    path = save_temp(file)
    try:
        if file.filename.lower().endswith(".pdf"):
            raw = extract_text_from_pdf(path)
        elif file.filename.lower().endswith(".docx"):
            raw = extract_text_from_docx(path)
        else:
            raw = open(path, "r", encoding="utf-8", errors="ignore").read()
    finally:
        os.unlink(path)
    sections = section_split(raw)
    db = SessionLocal()
    resume = Resume(filename=file.filename, raw_text=raw, sections=sections)
    db.add(resume); db.commit(); db.refresh(resume); db.close()
    return {"resume_id": resume.id, "sections": sections}

@app.post("/evaluate")
async def evaluate(job_id: int = Form(...), resume_id: int = Form(...), hard_weight: float = Form(None), soft_weight: float = Form(None)):
    db = SessionLocal()
    job = db.query(Job).filter(Job.id == job_id).first()
    resume = db.query(Resume).filter(Resume.id == resume_id).first()
    if not job or not resume:
        raise HTTPException(status_code=404, detail="Job or Resume not found")
    weights = None
    if hard_weight is not None and soft_weight is not None:
        weights = {"hard": float(hard_weight), "soft": float(soft_weight)}
    result = final_score(resume.raw_text, job.parsed, weights)
    missing = result["missing"]["must"] + result["missing"]["good"]
    feedback = generate_feedback(resume.raw_text, job.raw_text, missing, jd_title=job.title)
    result["feedback"] = feedback
    evaluation = Evaluation(job_id=job.id, resume_id=resume.id, result=result)
    db.add(evaluation); db.commit(); db.refresh(evaluation); db.close()
    return {"evaluation_id": evaluation.id, "result": result}

# fetch evaluations for a job with filters
@app.get("/results/{job_id}")
async def get_results(job_id: int, min_score: float = 0.0, verdict: str = None, limit: int = 50, offset: int = 0):
    db = SessionLocal()
    query = db.query(Evaluation).filter(Evaluation.job_id == job_id)
    # naive JSON filter by loading result JSON in Python (SQLite JSON indexing not used here)
    items = []
    for e in query.offset(offset).limit(limit).all():
        res = e.result
        if res is None:
            continue
        if res.get("final", 0) < float(min_score):
            continue
        if verdict and res.get("verdict") != verdict:
            continue
        items.append({"id": e.id, "resume_id": e.resume_id, "result": res})
    db.close()
    return {"job_id": job_id, "count": len(items), "evaluations": items}

# search resumes semantically for a job (top-N)
@app.get("/search/{job_id}")
async def semantic_search(job_id: int, top_k: int = 10):
    from backend.services.embeddings_manager import EmbeddingsProvider, SBERTEmbeddingsProvider
    db = SessionLocal()
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    # load resumes and compute similarity (simple linear scan)
    from backend.services.evaluator import cosine_sim
    from backend.services.embeddings_manager import SBERTEmbeddingsProvider
    emb = SBERTEmbeddingsProvider()
    job_vec = emb.embed([job.raw_text])[0]
    rows = []
    for r in db.query(Resume).all():
        res_vec = emb.embed([r.raw_text])[0]
        score = cosine_sim(job_vec, res_vec) * 100
        rows.append({"resume_id": r.id, "filename": r.filename, "score": score})
    rows = sorted(rows, key=lambda x: x["score"], reverse=True)[:top_k]
    db.close()
    return {"job_id": job_id, "top_k": top_k, "results": rows}
