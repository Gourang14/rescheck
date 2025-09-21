# app/main.py  (only JD upload endpoint shown; keep other endpoints unchanged)
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import tempfile, os
from app.models import init_db, SessionLocal, Job
from backend.services.parser import extract_text_from_pdf, extract_text_from_docx
from backend.services.jd_parser import parse_jd

init_db()
app = FastAPI()

def save_temp(upload: UploadFile):
    suffix = os.path.splitext(upload.filename)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(upload.file.read())
    tmp.close()
    return tmp.name

@app.post("/jd/upload")
async def upload_jd(file: UploadFile = File(...), title: str = Form(""), use_groq: bool = Form(True), model_url: str = Form(None), api_key: str = Form(None)):
    path = save_temp(file)
    try:
        if file.filename.lower().endswith(".pdf"):
            raw = extract_text_from_pdf(path)
        elif file.filename.lower().endswith(".docx"):
            raw = extract_text_from_docx(path)
        else:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                raw = f.read()
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass

    parsed = parse_jd(raw, use_groq=use_groq, model_url=model_url, api_key=api_key)
    db = SessionLocal()
    job = Job(title=title or parsed.get("title",""), raw_text=raw, parsed=parsed)
    db.add(job); db.commit(); db.refresh(job); db.close()
    return {"job_id": job.id, "parsed": parsed}
