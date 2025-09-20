# from fastapi import FastAPI, UploadFile, File, Form, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pathlib import Path
# import uvicorn
# import json
# import shutil
# from app import models, evaluator

# BASE_DIR = Path(__file__).resolve().parent
# UPLOAD_DIR = BASE_DIR.parent / "uploads"
# UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# app = FastAPI(title="Resume Relevance Check — API")
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Dependency: DB session helpers
# SessionLocal = models.Session

# @app.post("/upload_jd/")
# async def upload_jd(file: UploadFile = File(...), title: str = Form(...)):
#     if not file.filename:
#         raise HTTPException(status_code=400, detail="Missing file")
#     dest = UPLOAD_DIR / f"jd_{file.filename}"
#     with open(dest, "wb") as f:
#         shutil.copyfileobj(file.file, f)

#     ext = file.filename.lower()
#     if ext.endswith(".pdf"):
#         raw = evaluator.extract_text_from_pdf(str(dest))
#     else:
#         raw = evaluator.extract_text_from_docx(str(dest))
#     raw = evaluator.normalize_text(raw)
#     parsed = evaluator.parse_jd_with_llm(raw, title)


#     db = SessionLocal()
#     job = models.Job(title=title, raw_text=raw, parsed=parsed)
#     db.add(job)
#     db.commit()
#     db.refresh(job)
#     db.close()
#     return {"job_id": job.id, "title": job.title}

# @app.post("/upload_resume/")
# async def upload_resume(file: UploadFile = File(...), student_name: str = Form(...)):
#     if not file.filename:
#         raise HTTPException(status_code=400, detail="Missing file")
#     dest = UPLOAD_DIR / f"res_{file.filename}"
#     with open(dest, "wb") as f:
#         shutil.copyfileobj(file.file, f)

#     ext = file.filename.lower()
#     if ext.endswith(".pdf"):
#         raw = evaluator.extract_text_from_pdf(str(dest))
#     else:
#         raw = evaluator.extract_text_from_docx(str(dest))
#     raw = evaluator.normalize_text(raw)
#     sections = evaluator.split_sections(raw)

#     db = SessionLocal()
#     r = models.Resume(name=student_name, raw_text=raw, sections=sections)
#     db.add(r)
#     db.commit()
#     db.refresh(r)
#     db.close()
#     return {"resume_id": r.id, "name": r.name}

# @app.post("/evaluate/")
# async def evaluate_resume(job_id: int = Form(...), resume_id: int = Form(...)):
#     db = SessionLocal()
#     job = db.query(models.Job).filter(models.Job.id == job_id).first()
#     resume = db.query(models.Resume).filter(models.Resume.id == resume_id).first()
#     if not job or not resume:
#         db.close()
#         raise HTTPException(status_code=404, detail="Job or Resume not found")

#     result = evaluator.score_resume_against_jd(resume.sections or {}, job.parsed or {})
#     # persist evaluation
#     eval_row = models.Evaluation(job_id=job.id, resume_id=resume.id, result=result)
#     db.add(eval_row)
#     db.commit()
#     db.refresh(eval_row)
#     db.close()
#     return result

# @app.get("/resume/{resume_id}")
# async def get_resume(resume_id: int):
#     db = SessionLocal()
#     res = db.query(models.Resume).filter(models.Resume.id == resume_id).first()
#     db.close()
#     if not res:
#         raise HTTPException(status_code=404, detail="Resume not found")
#     return {"id": res.id, "name": res.name, "sections": res.sections}

# @app.get("/job/{job_id}")
# async def get_job(job_id: int):
#     db = SessionLocal()
#     job = db.query(models.Job).filter(models.Job.id == job_id).first()
#     db.close()
#     if not job:
#         raise HTTPException(status_code=404, detail="Job not found")
#     return {"id": job.id, "title": job.title, "parsed": job.parsed}

# if __name__ == "__main__":
#     uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil, tempfile, os
from app.models import SessionLocal, init_db, Job, Resume, Evaluation
from backend.services.parser import extract_text_from_pdf, extract_text_from_docx
from backend.services.jd_parser import parse_jd_with_llm
from backend.services.evaluator import final_score
from backend.services.nlp_utils import section_split
from backend.services.feedback import generate_feedback

init_db()

app = FastAPI(title="Resume Relevance API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

def save_temp_file(upload: UploadFile):
    suffix = os.path.splitext(upload.filename)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(upload.file.read())
    tmp.close()
    return tmp.name

@app.post("/jd/upload")
async def upload_jd(file: UploadFile = File(...), title: str = Form("")):
    path = save_temp_file(file)
    try:
        if file.filename.lower().endswith(".pdf"):
            raw = extract_text_from_pdf(path)
        else:
            raw = extract_text_from_docx(path) if file.filename.lower().endswith(".docx") else open(path, "r", encoding="utf-8", errors="ignore").read()
    finally:
        os.unlink(path)
    parsed = parse_jd_with_llm(raw, title)
    db = SessionLocal()
    job = Job(title=title, raw_text=raw, parsed=parsed)
    db.add(job)
    db.commit()
    db.refresh(job)
    db.close()
    return {"job_id": job.id, "parsed": parsed}

@app.post("/resume/upload")
async def upload_resume(file: UploadFile = File(...)):
    path = save_temp_file(file)
    try:
        if file.filename.lower().endswith(".pdf"):
            raw = extract_text_from_pdf(path)
        else:
            raw = extract_text_from_docx(path) if file.filename.lower().endswith(".docx") else open(path, "r", encoding="utf-8", errors="ignore").read()
    finally:
        os.unlink(path)
    sections = section_split(raw)
    db = SessionLocal()
    resume = Resume(filename=file.filename, raw_text=raw, sections=sections)
    db.add(resume)
    db.commit()
    db.refresh(resume)
    db.close()
    return {"resume_id": resume.id, "sections": sections}

@app.post("/evaluate")
async def evaluate(job_id: int = Form(...), resume_id: int = Form(...)):
    db = SessionLocal()
    job = db.query(Job).filter(Job.id == job_id).first()
    resume = db.query(Resume).filter(Resume.id == resume_id).first()
    if not job or not resume:
        raise HTTPException(status_code=404, detail="Job or Resume not found")
    # run scoring
    res = final_score(resume.raw_text, job.parsed)
    # generate feedback text
    missing_list = res["missing"].get("must", []) + res["missing"].get("good", [])
    feedback_text = generate_feedback(resume.raw_text, job.raw_text, missing_list, jd_title=job.title)
    res["feedback"] = feedback_text
    # store evaluation
    evaluation = Evaluation(job_id=job.id, resume_id=resume.id, result=res)
    db.add(evaluation)
    db.commit()
    db.refresh(evaluation)
    db.close()
    return {"evaluation_id": evaluation.id, "result": res}

@app.get("/results/{job_id}")
async def get_results(job_id: int):
    db = SessionLocal()
    evs = db.query(Evaluation).filter(Evaluation.job_id == job_id).all()
    out = [{"id": e.id, "resume_id": e.resume_id, "result": e.result} for e in evs]
    db.close()
    return {"job_id": job_id, "evaluations": out}