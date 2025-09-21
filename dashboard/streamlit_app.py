import streamlit as st
import requests
import pandas as pd

API = "http://localhost:8000"

st.title("Placement — Resume Relevance Dashboard (MVP)")

st.sidebar.header("Upload JD")
jd_file = st.sidebar.file_uploader("Upload JD (pdf/docx)", type=['pdf','docx'])
jd_title = st.sidebar.text_input("Job title")
if st.sidebar.button("Upload JD") and jd_file and jd_title:
    files = {'file': (jd_file.name, jd_file.getvalue(), jd_file.type)}
    data = {'title': jd_title}
    resp = requests.post(f"{API}/upload_jd/", files=files, data=data)
    st.sidebar.success(f"Uploaded JD id {resp.json()['job_id']}")

st.sidebar.header("Upload Resume")
res_file = st.sidebar.file_uploader("Upload Resume", type=['pdf','docx'], key='resfile')
student_name = st.sidebar.text_input("Student name", key='student')
if st.sidebar.button("Upload Resume"):
    files = {'file': (res_file.name, res_file.getvalue(), res_file.type)}
    data = {'student_name': student_name}
    resp = requests.post(f"{API}/upload_resume/", files=files, data=data)
    st.sidebar.success(f"Uploaded Resume id {resp.json()['resume_id']}")

st.header("Evaluate")
job_id = st.number_input("Job id", step=1, value=1)
resume_id = st.number_input("Resume id", step=1, value=1)
if st.button("Evaluate"):
    data = {'job_id': job_id, 'resume_id': resume_id}
    resp = requests.post(f"{API}/evaluate/", data=data)
    st.json(resp.json())
