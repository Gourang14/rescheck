# streamlit_app.py

import streamlit as st
from google import genai
from typing import List
import PyPDF2
import docx
import os

# --- Gemini API Client ---
st.set_page_config(page_title="Automated Resume Relevance Check", layout="wide")
st.title("Resume Relevance Checker powered by Gemini API")

# Ask user for Gemini API key
gemini_api_key = st.text_input("Enter your Gemini API Key:", type="password")

if gemini_api_key:
    client = genai.Client(api_key=gemini_api_key)

    # --- Upload Job Description ---
    st.header("Upload Job Description")
    jd_file = st.file_uploader("Upload JD (PDF or DOCX)", type=["pdf", "docx"])

    # --- Upload Resumes ---
    st.header("Upload Resumes")
    resumes = st.file_uploader("Upload Resumes (PDF or DOCX, multiple allowed)", type=["pdf", "docx"], accept_multiple_files=True)

    def extract_text(file) -> str:
        """Extract text from PDF or DOCX"""
        if file.name.endswith(".pdf"):
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        elif file.name.endswith(".docx"):
            doc = docx.Document(file)
            text = "\n".join([p.text for p in doc.paragraphs])
            return text
        else:
            return ""

    def parse_jd_gemini(jd_text: str) -> dict:
        """Use Gemini to parse JD into structured fields"""
        prompt = f"""
        Extract the following from this Job Description:
        - Title
        - Must-have skills
        - Good-to-have skills
        - Qualifications

        Provide output as JSON only.
        JD: '''{jd_text}'''
        """
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        try:
            import json
            return json.loads(response.text)
        except:
            st.warning("Failed to parse JD via Gemini API.")
            return {}

    def score_resume_gemini(resume_text: str, jd_json: dict) -> dict:
        """Use Gemini to generate relevance score, verdict, and feedback"""
        prompt = f"""
        You are an AI resume evaluator.
        JD JSON: {jd_json}
        Resume Text: '''{resume_text}'''

        Tasks:
        1. Compute a relevance score (0-100)
        2. Provide High / Medium / Low suitability
        3. Highlight missing skills/projects/certifications
        4. Give brief improvement feedback

        Provide output as JSON:
        {{
            "relevance_score": int,
            "verdict": str,
            "missing_elements": list,
            "feedback": str
        }}
        """
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        try:
            import json
            return json.loads(response.text)
        except:
            st.warning("Failed to score resume via Gemini API.")
            return {}

    if jd_file and resumes:
        jd_text = extract_text(jd_file)
        st.subheader("Parsing Job Description...")
        jd_parsed = parse_jd_gemini(jd_text)
        st.json(jd_parsed)

        results = []
        for r in resumes:
            resume_text = extract_text(r)
            st.subheader(f"Evaluating {r.name} ...")
            score = score_resume_gemini(resume_text, jd_parsed)
            st.json({r.name: score})
            results.append({"resume": r.name, **score})

        # Display leaderboard
        st.header("Leaderboard")
        import pandas as pd
        df = pd.DataFrame(results)
        if "relevance_score" in df.columns:
            df = df.sort_values(by="relevance_score", ascending=False)
        st.dataframe(df)

else:
    st.info("Please enter your Gemini API Key to start evaluating JDs and resumes.")
