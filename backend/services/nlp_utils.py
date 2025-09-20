import re
from rapidfuzz import fuzz, process

def section_split(text: str):
    """
    Naive section split by common resume headings.
    Returns a dict {section_name: section_text}.
    """
    headings = ["EXPERIENCE", "EDUCATION", "SKILLS", "PROJECTS", "CERTIFICATIONS"]
    # Regex pattern to split sections
    pattern = r"\n\s*(" + "|".join(headings) + r")\s*\n"
    parts = re.split(pattern, text, flags=re.I)

    sections = {}
    current_heading = "General"
    buffer = []
    for part in parts:
        if part.strip().upper() in headings:
            # save old buffer
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
    Uses both exact and fuzzy matching.
    
    Args:
        resume_text (str): full resume text
        skills_list (list[str]): list of skills to check
        threshold (int): fuzzy match cutoff (0–100)

    Returns:
        dict: {skill: score}
    """
    resume_lower = resume_text.lower()
    found = {}

    for skill in skills_list:
        skill_lower = skill.lower()

        # Exact match check
        if skill_lower in resume_lower:
            found[skill] = 100
            continue

        # Fuzzy match check
        best_match, score, _ = process.extractOne(
            skill_lower, resume_lower.split(), scorer=fuzz.ratio
        )
        if score >= threshold:
            found[skill] = min(100, score)
        else:
            found[skill] = 0

    return found