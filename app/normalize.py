import hashlib
import re


def raw_hash(messages) -> str:
    """Exact-match key. Over the RAW text — zero false-hit risk by construction."""
    blob = "\x00".join(f"{m['role']}:{m['content']}" for m in messages)
    return hashlib.sha256(blob.encode()).hexdigest()


_WS = re.compile(r"\s+")


def _user_text(messages) -> str:
    return " ".join(m["content"] for m in messages if m["role"] == "user")


def normalize(messages) -> str:
    """Embedding input ONLY. Lossy on purpose; never used as an exact key."""
    return _WS.sub(" ", _user_text(messages).lower()).strip()


def constraint_input(messages) -> str:
    """Constraint-extraction input — case preserved, unlike normalize().
    app/constraints.py's entities dimension depends on capitalization to
    find proper nouns; feeding it normalize()'s lowercased output would
    silently zero out entity detection entirely, not just degrade it.
    """
    return _user_text(messages)


def params_hash(messages, temperature: float, max_tokens: int, model: str) -> str:
    """Hashes what makes two requests different besides the user text:
    system prompt, temperature, max_tokens, upstream model. Miss one of
    these and a cached answer generated under different params gets
    served as if it were the same question — e.g. a temperature-0 answer
    returned to a temperature-1 request.
    """
    system_prompt = next((m["content"] for m in messages if m["role"] == "system"), "")
    blob = "\x00".join([system_prompt, str(temperature), str(max_tokens), model])
    return hashlib.sha256(blob.encode()).hexdigest()


def hash_api_key(api_key: str) -> str:
    """Not cache-key hashing — shared here only because it's the same
    primitive. tenants.api_key_hash stores this, never the raw key; the
    raw key is shown once at seed time and never persisted anywhere.
    """
    return hashlib.sha256(api_key.encode()).hexdigest()
