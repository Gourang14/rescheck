import re

def safe_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-\.]", "_", name)