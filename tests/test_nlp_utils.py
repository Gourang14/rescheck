from backend.services import nlp_utils
import sys, os
import pytest

# Add project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.services import nlp_utils

def test_section_split_returns_dict():
    text = """
    EXPERIENCE
    Worked as Data Analyst at XYZ Corp.
    EDUCATION
    B.Tech in Computer Science
    SKILLS
    Python, SQL, AWS
    PROJECTS
    Resume Screening Automation
    """
    sections = nlp_utils.section_split(text)
    assert isinstance(sections, dict)
    assert "Experience" in sections
    assert "Education" in sections
    assert "Skills" in sections
    assert "Projects" in sections

def test_extract_skills_exact_and_fuzzy():
    resume_text = "Experienced in Python, SQL, and AWS Cloud. Certified in Azure."
    skills = ["Python", "Java", "AWS", "Azure", "Machine Learning"]

    results = nlp_utils.extract_skills(resume_text, skills)

    # Python should be exact match
    assert results["Python"] == 100

    # AWS should be fuzzy but >80
    assert results["AWS"] >= 80

    # Java should not be found
    assert results["Java"] == 0

    # Azure should be found
    assert results["Azure"] == 100

    # Machine Learning should not be found
    assert results["Machine Learning"] == 0