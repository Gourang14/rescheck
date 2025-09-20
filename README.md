# Resume Relevance Check — MVP

## Quick start (local)

1. Create venv and install

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Set OpenAI key (optional but recommended for embeddings)

```bash
export OPENAI_API_KEY="sk-..."
```

3. Run API

```bash
uvicorn app.main:app --reload --port 8000
```

4. Open Streamlit dashboard (in another terminal)

```bash
streamlit run dashboard/streamlit_app.py
```

5. Use `demo.sh` to test (requires `jq` & sample files in `sample_data/`)


## Notes
- This is an MVP. For production: use S3 for file storage, persistent vector DB (Chroma/Pinecone), job queue for async batch processing, and add authentication.
```

---