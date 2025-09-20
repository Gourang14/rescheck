import re

# Try to import rapidfuzz, else fallback to difflib
try:
    from rapidfuzz import fuzz, process
    _has_rapidfuzz = True
except ImportError:
    import difflib
    _has_rapidfuzz = False


def section_split(text: str):
    """
    Split resume text into sections by common headings.
    Returns dict {section_name: section_text}.
    """
    headings = ["EXPERIENCE", "EDUCATION", "SKILLS", "PROJECTS", "CERTIFICATIONS"]
    pattern = r"\n\s*(" + "|".join(headings) + r")\s*\n"
    parts = re.split(pattern, text, flags=re.I)

    sections = {}
    current_heading = "General"
    buffer = []
    for part in parts:
        if part.strip().upper() in headings:
            if buffer:
                sections[current_heading] = "\n".join(buffer).strip()
                buffer = []
            current_heading = part.strip().title()
        else:
            buffer.append(part)
    if buffer:
        sections[current_heading] = "\n".join(buffer).strip()

    return sections


def extract_skills(resume_text: str, skills_list: list, threshold: int = 80):
    """
    Extract skills from resume_text by checking presence against skills_list.
    Uses exact and fuzzy matching (RapidFuzz if available, else difflib).
    """
    resume_lower = resume_text.lower()
    found = {}

    for skill in skills_list:
        skill_lower = skill.lower()

        # Exact match
        if skill_lower in resume_lower:
            found[skill] = 100
            continue

        # Fuzzy match
        if _has_rapidfuzz:
            best_match, score, _ = process.extractOne(
                skill_lower, resume_lower.split(), scorer=fuzz.ratio
            )
            found[skill] = min(100, score) if score >= threshold else 0
        else:
            best_match = difflib.get_close_matches(
                skill_lower, resume_lower.split(), n=1, cutoff=threshold / 100
            )
            found[skill] = 90 if best_match else 0  # simplified fallback

    return found