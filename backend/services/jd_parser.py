import os
import json
from typing import Dict, Any
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

SCHEMA = {
    "title": "string",
    "must_have_skills": ["string"],
    "nice_to_have_skills": ["string"],
    "qualifications": ["string"],
    "other_details": "string"
}

def parse_jd_with_llm(jd_text: str, api_key: str = None) -> Dict[str, Any]:
    """
    Use LLM (OpenAI or Groq) to parse JD into structured fields.
    Falls back to naive regex if no API key is available.
    """
    if not api_key:
        # Fallback: crude extraction
        return {
            "title": jd_text.splitlines()[0][:100],
            "must_have_skills": [],
            "nice_to_have_skills": [],
            "qualifications": [],
            "other_details": jd_text[:500]
        }

    os.environ["OPENAI_API_KEY"] = api_key
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = PromptTemplate(
        input_variables=["jd_text"],
        template="""
You are an assistant that extracts structured fields from job descriptions.
Extract the following strictly as JSON:
{{
  "title": "...",
  "must_have_skills": ["..."],
  "nice_to_have_skills": ["..."],
  "qualifications": ["..."],
  "other_details": "..."
}}
Text:
{jd_text}
"""
    )

    parser = JsonOutputParser()
    chain = prompt | model | parser

    try:
        result = chain.invoke({"jd_text": jd_text})
        return result
    except Exception as e:
        return {
            "title": jd_text.splitlines()[0][:100],
            "must_have_skills": [],
            "nice_to_have_skills": [],
            "qualifications": [],
            "other_details": jd_text[:500],
            "error": str(e)
        }
