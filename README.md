# Resume Relevance Check — MVP 🦇

An **AI-powered system** that evaluates resumes against job descriptions (JDs).  
It produces a **Relevance Score (0–100)**, highlights missing skills, and generates actionable feedback — helping recruiters quickly shortlist candidates.

This project is part of **Theme 2 — Automated Resume Relevance Check System (Innomatics Research Labs)**.

---

## ✨ Features
- 📄 Upload **Job Descriptions** (PDF/DOCX/TXT).
- 📑 Upload **Resumes** (PDF/DOCX, multiple at once).
- ⚖️ **Hybrid Scoring**:
  - **Hard Match**: keyword & fuzzy skill matching.
  - **Soft Match**: semantic similarity via embeddings.
- 🎯 Outputs:
  - Final **Relevance Score** (0–100).
  - Verdict: ✅ High / ⚠️ Medium / ❌ Low.
  - Missing Skills.
  - Actionable Feedback.
- 📊 Dashboard for placement team:
  - Upload JD & resumes.
  - Ranked shortlist + filtering.
  - Stored evaluations history (SQLite).
- 🧩 Modular backend (LangChain, embeddings, vector store ready).

---

## 🛠️ Tech Stack
**Backend**
- FastAPI (resume/JD processing APIs)
- SQLite (MVP DB) → can scale to PostgreSQL
- LangChain, SBERT/OpenAI embeddings
- RapidFuzz (fuzzy matching), PyMuPDF / python-docx

**Frontend**
- Streamlit dashboard (upload, review, filtering)

---

## 🚀 Quick Start (Local)

1. **Clone repo & setup environment**
```bash
git clone https://github.com/your-org/rescheck.git
cd rescheck
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate
pip install -r requirements.txt
