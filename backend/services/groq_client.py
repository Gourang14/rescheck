# backend/services/groq_client.py
import os, re, json, requests
from typing import Optional

# prefer env vars but allow passing explicitly
DEFAULT_TIMEOUT = 30

def _try_parse_json(resp_text: str):
    # 1) direct json
    try:
        return json.loads(resp_text)
    except Exception:
        pass
    # 2) common wrapper keys e.g. {"output": [{"content":"..."}]} or {"results":[{"text":"..."}]}
    try:
        short = resp_text.strip()
        # balanced-brace extraction: find first {...} block
        start = short.find('{')
        if start != -1:
            depth = 0
            for i in range(start, len(short)):
                ch = short[i]
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        cand = short[start:i+1]
                        return json.loads(cand)
    except Exception:
        pass
    # 3) try to find a JSON object substring using regex (last resort)
    try:
        m = re.search(r'(\{(?:[^{}]|(?R))*\})', resp_text, flags=re.S)
        if m:
            return json.loads(m.group(1))
    except Exception:
        pass
    return None

def call_groq_raw(prompt: str,
                  model_url: Optional[str] = None,
                  api_key: Optional[str] = None,
                  max_output_tokens: int = 1024,
                  temperature: float = 0.0,
                  timeout: int = DEFAULT_TIMEOUT) -> str:
    url = model_url or os.getenv("GROQ_API_URL", "").strip()
    key = api_key or os.getenv("GROQ_API_KEY", "").strip()
    if not (url and key):
        raise RuntimeError("GROQ_API_URL and GROQ_API_KEY must be provided (env or params).")

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    # Try two common payload shapes: {"input": "..."} and {"prompt": "..."}
    bodies = [
        {"input": prompt, "max_output_tokens": max_output_tokens, "temperature": temperature},
        {"prompt": prompt, "max_tokens": max_output_tokens, "temperature": temperature},
        {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_output_tokens, "temperature": temperature}
    ]

    last_err = None
    for body in bodies:
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=timeout)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            last_err = e
            # try next shape
            continue
    # if we reach here none succeeded
    raise RuntimeError(f"Groq request failed with last error: {last_err}") from last_err

def groq_generate_json(prompt: str,
                       model_url: Optional[str] = None,
                       api_key: Optional[str] = None,
                       max_output_tokens: int = 1024,
                       temperature: float = 0.0,
                       timeout: int = DEFAULT_TIMEOUT):
    raw = call_groq_raw(prompt, model_url=model_url, api_key=api_key,
                        max_output_tokens=max_output_tokens, temperature=temperature, timeout=timeout)
    parsed = _try_parse_json(raw)
    if parsed is not None:
        return parsed
    # If no JSON object found, attempt to extract the textual output field if provider wraps results
    # Try to heuristically extract "output_text" or similar fields by regex
    # Return a dict with 'raw_response' key so caller can inspect
    return {"raw_response": raw}
