# src/contact_dedupe/pipeline/llm_validation.py
import os
from typing import Optional

# Only hit OpenAI if USE_LLM=1 and an API key is present
USE_LLM = os.getenv("USE_LLM", "0") == "1"

try:
    from openai import OpenAI  # type: ignore

    _CLIENT: Optional[OpenAI] = OpenAI() if os.getenv("OPENAI_API_KEY") else None
except Exception:
    _CLIENT = None


def ask_names_same(name_a: str, name_b: str, *, lenient: bool = True) -> bool:
    """
    Returns True if the LLM thinks the two names are the same person.
    Runs only when USE_LLM=1 and _CLIENT is available; otherwise always False.
    """
    if not USE_LLM or _CLIENT is None:
        return False

    system = (
        "You are a contact dedupe assistant. "
        "Be very lenient: treat nicknames, misspellings, and phonetics as the same person. "
        "Answer strictly YES or NO."
    )
    if not lenient:
        system = (
            "You are a cautious contact dedupe assistant. Answer strictly YES or NO."
        )

    try:
        resp = _CLIENT.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": f"1) {name_a}\n2) {name_b}\nAre these the same person? Answer YES or NO.",
                },
            ],
        )
        return resp.choices[0].message.content.strip().upper().startswith("Y")
    except Exception:
        return False
