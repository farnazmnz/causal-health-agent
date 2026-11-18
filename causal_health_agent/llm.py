"""Thin OpenRouter client. Keys are read ONLY from environment variables (never stored in the repo):
    OPENROUTER_API_KEY   TAVILY_API_KEY
"""
import json
import os
import re

from . import config as C


def get_client():
    from openai import OpenAI
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("Set the OPENROUTER_API_KEY environment variable (see .env.example).")
    return OpenAI(api_key=key, base_url=C.OPENROUTER_BASE_URL)


def chat(client, model, messages, temperature=0.0):
    r = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
    return r.choices[0].message.content


def parse_json(raw):
    """Parse a JSON object from an LLM reply (tolerates ```json fences and surrounding text)."""
    t = re.sub(r"^```(?:json)?|```$", "", str(raw).strip(), flags=re.M).strip()
    m = re.search(r"\{.*\}", t, re.S)
    for cand in (t, m.group(0) if m else None):
        if cand:
            try:
                return json.loads(cand)
            except Exception:
                pass
    return {}
