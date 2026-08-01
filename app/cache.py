"""§4 step 7 (similarity search), §5 (admission decision), §7 (output-
constraint audit). Imported by both the live service (app/main.py) and
eval/ scripts — same code path, so eval measures what production runs.
"""

import json
import re

from app.constraints import compatible


def constraints_to_json(c: dict) -> dict:
    """extract()'s entities is a set — not directly JSON-serializable
    through the jsonb codec (json.dumps chokes on a raw set)."""
    c = dict(c)
    if c.get("entities") is not None:
        c["entities"] = sorted(c["entities"])
    return c


def constraints_from_json(c: dict) -> dict:
    c = dict(c)
    if c.get("entities") is not None:
        c["entities"] = set(c["entities"])
    return c


async def search_candidates(conn, tenant_id, upstream_model, params_hash, embedding, k: int):
    """§4 step 7: exact cosine scan, ORDER BY <=> ascending — that operator
    is cosine *distance*, so ascending distance is descending similarity,
    nearest/most-similar first. Scoped by tenant+model+params: cross-
    tenant reuse is a data leak, and a different system prompt or
    temperature is a different question (§3)."""
    return await conn.fetch(
        """SELECT id, response_text, response_tokens, constraints,
                  1 - (embedding <=> $1) AS similarity
           FROM cache_entries
           WHERE tenant_id = $2 AND upstream_model = $3 AND params_hash = $4
           ORDER BY embedding <=> $1
           LIMIT $5""",
        embedding, tenant_id, upstream_model, params_hash, k,
    )


def admit(candidates, request_constraints: dict, threshold: float) -> dict:
    """§5's admission loop, with fall-through: a candidate that fails the
    constraint gate doesn't end the search — the next-nearest candidate
    still gets a chance, rather than giving up on the first rejection.

    Returns a dict: decision in {HIT_SEMANTIC, MISS_LOW_SIM, MISS_GATE,
    MISS_NO_CANDIDATE}, candidate (the winning row or None), similarity
    (of the decisive candidate, for logging even on a miss), and diff —
    for MISS_GATE, a list of every candidate checked and why each was
    rejected (richer than just the first, since Phase 8's false-hits-by-
    dimension figure wants to know which dimension differed, not just
    that one did).
    """
    if not candidates:
        return {"decision": "MISS_NO_CANDIDATE", "candidate": None, "similarity": None, "diff": None}

    checked = []
    for c in candidates:
        sim = c["similarity"]
        if sim < threshold:
            # DECISION-V2: candidates arrive sorted by similarity
            # descending (search_candidates' ORDER BY), so the first one
            # below tau means every remaining candidate is too — stop
            # instead of scanning the rest. Matches TOLLGATE.md §5's
            # pseudocode exactly (return MISS_LOW_SIM on the first
            # c.sim < tau, not "keep looking for a compatible low-sim one").
            return {"decision": "MISS_LOW_SIM", "candidate": None, "similarity": sim, "diff": None}
        ok, diff = compatible(request_constraints, constraints_from_json(c["constraints"]))
        checked.append({"candidate_id": str(c["id"]), "similarity": sim, "diff": diff})
        if ok:
            return {"decision": "HIT_SEMANTIC", "candidate": c, "similarity": sim, "diff": None}
    return {
        "decision": "MISS_GATE",
        "candidate": None,
        "similarity": checked[0]["similarity"],
        "diff": checked,
    }


# ---------------------------------------------------------------------
# §7 — output-constraint audit. On every HIT, before returning: compare
# the INCOMING request's constraints against the CACHED RESPONSE's own
# observable properties. This is not re-checking the gate (that compares
# two prompts) — it catches the upstream model's own imperfect
# instruction-following, e.g. asked for 3 bullets, produced 2, and that
# stale-but-imperfect response keeps getting served on every future hit.
# No upstream call, no judge, no sampling — a continuously measured lower
# bound on production false-hit rate.
# ---------------------------------------------------------------------

_BULLET_LINE_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+", re.MULTILINE)
_TABLE_ROW_RE = re.compile(r"^\s*\|.+\|\s*$", re.MULTILINE)

# DECISION-V2: language detection here is a coarse heuristic (script
# unicode ranges + a handful of characteristic Latin-script markers), not
# a real language-ID model. A proper one (e.g. langdetect) would be more
# accurate but adds a dependency for a lower-bound signal that only needs
# to be roughly right — "no judge" in the spec reads as "don't spend a
# model call or real precision on this," and this stays well inside that.
_SCRIPT_RANGES = [
    ("russian", range(0x0400, 0x0500)),
    ("japanese", range(0x3040, 0x3100)),  # hiragana/katakana - kana implies Japanese even mixed with kanji
    ("chinese", range(0x4E00, 0xA000)),  # CJK ideographs with no kana seen
    ("arabic", range(0x0600, 0x0700)),
    ("korean", range(0xAC00, 0xD7A4)),
    ("hindi", range(0x0900, 0x0980)),
]
_LATIN_MARKERS = {
    "german": set("äöüß"),
    "french": set("éèêëàâùûçœ"),
    "spanish": set("ñ¿¡"),
    "portuguese": set("ãõ"),
    "italian": set("ìò"),
}


def _detect_language(text: str) -> str:
    for lang, code_range in _SCRIPT_RANGES:
        if any(ord(c) in code_range for c in text):
            return lang
    lower = text.lower()
    for lang, markers in _LATIN_MARKERS.items():
        if any(c in markers for c in lower):
            return lang
    return "english"


def _length_band(n_chars: int) -> str:
    if n_chars < 200:
        return "short"
    if n_chars < 800:
        return "medium"
    return "long"


def _extract_response_properties(text: str) -> dict:
    bullet_lines = _BULLET_LINE_RE.findall(text)
    bullet_count = len(bullet_lines) if bullet_lines else None

    table_present = bool(_TABLE_ROW_RE.search(text))

    json_valid = None
    stripped = text.strip()
    if stripped[:1] in "{[":
        try:
            json.loads(stripped)
            json_valid = True
        except (json.JSONDecodeError, ValueError):
            json_valid = False

    if json_valid:
        fmt = "json"
    elif table_present:
        fmt = "table"
    elif bullet_count:
        fmt = "bullets"
    else:
        fmt = "prose"

    return {
        "bullet_count": bullet_count,
        "table_present": table_present,
        "json_valid": json_valid,
        "language": _detect_language(text),
        "length_band": _length_band(len(text)),
        "format": fmt,
    }


_LIST_TYPE_NOUNS = {"bullet", "bullets", "point", "points", "item", "items", "step", "steps"}
_COUNT_NOUN_RE = re.compile(
    r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+\w*\s*"
    r"(bullets?|points?|words?|sentences?|examples?|steps?|items?|paragraphs?|ways?|reasons?)\b",
    re.I,
)


def _count_is_list_type(request_text: str) -> bool:
    """DECISION-V2: the audit's count-check only ever counts bullet/list
    lines (props["bullet_count"]) — it has no word or sentence counter.
    Without this guard, ANY "N sentences"/"N words" request would register
    a false audit failure (bullet_count is always None for prose),
    inflating llm_compliance_failure_rate for a whole class of requests
    the audit was never able to verify. Cheaper to skip the check than to
    grow a second constraint parser just for the audit path; the request
    noun is re-derived here rather than carried on request_constraints
    (which only keeps the count integer, not which noun it was attached
    to) — a narrow, audit-only regex, not a change to extract()'s schema.
    """
    m = _COUNT_NOUN_RE.search(request_text)
    return bool(m) and m.group(1).lower() in _LIST_TYPE_NOUNS


def audit_output(request_constraints: dict, request_text: str, response_text: str) -> tuple[bool, dict]:
    """Only checks dimensions the request actually constrained — a
    request with no format/count/language opinion can't fail on one.
    `count` is checked against list-item count specifically (bullets or
    numbered lines) and only when the request actually asked for a list-
    type count (see _count_is_list_type) — "N sentences"/"N words" is
    skipped rather than compared against a bullet count it was never
    about.
    """
    props = _extract_response_properties(response_text)
    diff = {}

    if request_constraints.get("count") is not None and _count_is_list_type(request_text):
        if props["bullet_count"] != request_constraints["count"]:
            diff["count"] = [request_constraints["count"], props["bullet_count"]]

    if request_constraints.get("format") is not None:
        if props["format"] != request_constraints["format"]:
            diff["format"] = [request_constraints["format"], props["format"]]

    if request_constraints.get("language") is not None:
        if props["language"] != request_constraints["language"]:
            diff["language"] = [request_constraints["language"], props["language"]]

    return len(diff) == 0, diff
