# backend/services/groq_client.py
import os
import requests
import json
import re
from typing import Optional

GROQ_API_URL = os.getenv("GROQ_API_URL", "").strip()  # e.g. "https://api.groq.ai/v1/models/llama-38b/generate"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

DEFAULT_TIMEOUT = 30  # seconds

def _extract_json_from_text(text: str) -> Optional[dict]:
    """
    Find the first JSON object inside a model response and return parsed dict.
    """
    try:
        # crude but practical: capture {...}
        m = re.search(r'(\{(?:[^{}]|\n|(?R))*\})', text, flags=re.S)
        if m:
            payload = m.group(1)
            return json.loads(payload)
    except Exception:
        pass
    try:
        return json.loads(text)
    except Exception:
        return None

def call_groq(prompt: str, max_tokens: int = 1024, temperature: float = 0.0, model_url: Optional[str] = None, timeout: int = DEFAULT_TIMEOUT):
    """
    Generic POST call to a Groq-compatible endpoint.
    model_url: provide full endpoint (if None, uses GROQ_API_URL env var).
    The payload format below is generic; if your provider expects a different JSON schema,
    adapt the body accordingly.
    """
    url = model_url or GROQ_API_URL
    if not url or not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_URL and GROQ_API_KEY environment variables must be set for Groq LLM use.")

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    # Common payload shape many Groq endpoints accept; adjust if your provider uses a different field name
    body = {
        "input": prompt,
        "max_output_tokens": max_tokens,
        "temperature": temperature
    }

    # POST and parse
    resp = requests.post(url, headers=headers, json=body, timeout=timeout)
    resp.raise_for_status()
    resp_text = resp.text

    return resp_text

def groq_generate_json(prompt: str, max_tokens: int = 1024, temperature: float = 0.0, model_url: Optional[str] = None):
    """
    Send prompt, attempt to parse JSON out of the LLM response.
    Returns dict on success, or raises RuntimeError on failure.
    """
    text = call_groq(prompt, max_tokens=max_tokens, temperature=temperature, model_url=model_url)
    parsed = _extract_json_from_text(text)
    if parsed is None:
        raise RuntimeError(f"Groq did not return parseable JSON. Raw response: {text[:2000]}")
    return parsed
