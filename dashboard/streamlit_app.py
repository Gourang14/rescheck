import streamlit as st
import requests
import json

# Constants
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3-8b-instruct"   # Replace with llama-3-70b if needed

# ======================
# Helper: Call Groq API
# ======================
def call_groq(api_key, prompt):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2
    }
    try:
        res = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=60)
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"]
    except Exception as e:
        st.error(f"❌ Groq API error: {e}")
        return None

# ======================
# JD Parsing Function
# ======================
def parse_jd(api_key, jd_text):
    prompt = f"""
    You are a job description parser. Extract the following fields from this JD:
    1. Job Title
    2. Must-have Skills (hard requirements)
    3. Good-to-have Skills (bonus skills)
    4. Qualifications (degrees, years of experience, certifications)

    Return a JSON with keys: title, must_have_skills, nice_to_have_skills, qualifications.

    JD:
    {jd_text}
    """
    output = call_groq(api_key, prompt)
    try:
        return json.loads(output)
    except:
        st.warning("⚠️ Could not parse Groq output into JSON. Showing raw output.")
        return {"raw": output}

# ======================
# Resume Scoring
# ======================
def score_resume(api_key, resume_text, jd_parsed):
    prompt = f"""
    You are a resume evaluator. Given a job description schema and a candidate resume,
    score the resume and provide feedback.

    Job Schema:
    {json.dumps(jd_parsed, indent=2)}

    Resume:
    {resume_text}

    Return JSON with keys:
    - score (0-100)
    - verdict (Shortlist / Maybe / Reject)
    - feedback (list of suggestions)
    """
    output = call_groq(api_key, prompt)
    try:
        return json.loads(output)
    except:
        st.warning("⚠️ Could not parse Groq output into JSON. Showing raw output.")
        return {"raw": output}

# ======================
# Streamlit UI
# ======================
st.title("🚀 AI Resume Screener (Groq-Powered)")

api_key = st.text_input("Enter your Groq API Key", type="password")

if api_key:
    jd_text = st.text_area("📄 Paste Job Description", height=200)

    if st.button("Parse JD with Groq"):
        if jd_text.strip():
            jd_parsed = parse_jd(api_key, jd_text)
            st.json(jd_parsed)

            st.subheader("📌 Upload Candidate Resumes")
            resumes = st.file_uploader("Upload Resumes (.txt)", type="txt", accept_multiple_files=True)

            if resumes:
                results = []
                for file in resumes:
                    resume_text = file.read().decode("utf-8")
                    result = score_resume(api_key, resume_text, jd_parsed)
                    results.append({"resume": file.name, "result": result})

                st.subheader("📊 Results")
                for r in results:
                    st.markdown(f"### {r['resume']}")
                    st.json(r["result"])
        else:
            st.warning("Please paste a JD first.")
else:
    st.info("🔑 Enter your Groq API Key to start.")
