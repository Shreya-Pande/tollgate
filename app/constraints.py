"""§6: pure Python, regex + lexicons, no model on the hot path. Every
pattern is compiled at import time — extract() must run under 1ms/prompt
(tests/test_constraints.py enforces this), and re-compiling per call would
blow that budget for no reason.
"""

import re

NUM_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
OUTPUT_NOUNS = r"(?:bullets?|points?|words?|sentences?|examples?|steps?|items?|paragraphs?|ways?|reasons?)"
_NUM_ALT = r"(?:\d+|" + "|".join(NUM_WORDS) + r")"
_COUNT_RE = re.compile(rf"\b({_NUM_ALT})\s+\w*\s*{OUTPUT_NOUNS}\b", re.I)

# Longer cues first: "do not" must be tried as a whole before "not" alone
# would otherwise be found starting mid-phrase in some inputs.
NEG_CUES = ["do not", "don't", "without", "excluding", "other than", "except", "avoid", "not"]
_NEG_RE = re.compile(r"\b(" + "|".join(re.escape(c) for c in NEG_CUES) + r")\b", re.I)
_SENT_END_RE = re.compile(r"[.!?]")

LANGUAGES = [
    "french", "german", "spanish", "hindi", "japanese", "italian",
    "portuguese", "chinese", "russian", "arabic", "korean",
]
_LANG_RE = re.compile(rf"\b(?:in|into|to)\s+({'|'.join(LANGUAGES)})\b", re.I)

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_CAP_TOKEN_RE = re.compile(r"\b[A-Z][a-zA-Z]*\b")
_NUMERAL_RE = re.compile(r"\b\d+\b")

# Longest phrase first per label, so e.g. "prose paragraph" is preferred
# over the shorter "paragraph" when both could match.
_FORMAT_PHRASES = [
    ("bullet points", "bullets"), ("bulleted list", "bullets"),
    ("bullet list", "bullets"), ("bullets", "bullets"),
    ("markdown table", "table"), ("table", "table"),
    ("json object", "json"), ("json", "json"),
    ("code block", "code"), ("code snippet", "code"), ("code", "code"),
    ("single prose paragraph", "prose"), ("prose paragraph", "prose"),
    ("prose", "prose"), ("paragraph", "prose"),
]
_FORMAT_PHRASES.sort(key=lambda p: -len(p[0]))
_FORMAT_LOOKUP = {phrase.lower(): label for phrase, label in _FORMAT_PHRASES}
_FORMAT_RE = re.compile(
    r"\b(" + "|".join(re.escape(p) for p, _ in _FORMAT_PHRASES) + r")\b", re.I
)


def _extract_count(text: str) -> int | None:
    m = _COUNT_RE.search(text)
    if not m:
        return None
    token = m.group(1).lower()
    return int(token) if token.isdigit() else NUM_WORDS[token]


def _extract_negation(text: str) -> str | None:
    """Cue plus the span it scopes over — the text from just after the cue
    to the end of that clause, normalized. Two prompts are compatible on
    this dimension only if their negation scopes match exactly; a bare
    True/False would let "not X" and "not Y" pass as the same constraint.
    """
    m = _NEG_RE.search(text)
    if not m:
        return None
    rest = text[m.end():]
    end = _SENT_END_RE.search(rest)
    scope = rest[: end.start()] if end else rest
    scope = re.sub(r"\s+", " ", scope).strip().lower()
    return scope or None


def _extract_language(text: str) -> str | None:
    m = _LANG_RE.search(text)
    return m.group(1).lower() if m else None


def _extract_entities(text: str) -> set[str] | None:
    entities: set[str] = set()
    for sentence in _SENT_SPLIT_RE.split(text):
        tokens = _CAP_TOKEN_RE.findall(sentence)
        # DECISION-V2: skip the sentence-initial token. Every sentence
        # starts capitalized for grammar reasons alone, not because the
        # word is an entity ("Focus on Python" vs "Do not include
        # Python" both start capitalized). Without this, negation's own
        # pair ("Include..." vs "Do not include...") would register a
        # spurious entities diff on top of the real negation diff,
        # diluting the false-hits-by-dimension breakdown for a dimension
        # that isn't actually what differs. Alternative: keep
        # sentence-initial tokens and accept the noise — rejected
        # because it specifically corrupts one family's diff attribution
        # rather than being a general, evenly-distributed cost.
        entities.update(tokens[1:])
    entities.update(_NUMERAL_RE.findall(text))
    return entities or None


def _extract_format(text: str) -> str | None:
    m = _FORMAT_RE.search(text)
    return _FORMAT_LOOKUP[m.group(1).lower()] if m else None


def extract(text: str) -> dict:
    return {
        "count": _extract_count(text),
        "negation": _extract_negation(text),
        "language": _extract_language(text),
        "entities": _extract_entities(text),
        "format": _extract_format(text),
    }


def compatible(a: dict, b: dict) -> tuple[bool, dict]:
    """Strict: for every dimension, both None or equal (set equality for
    entities). Returns the diff, not just a bool — {"count": [3, 10]} is
    what powers the false-hits-by-dimension figure; gate_passed=False
    alone is undebuggable.
    """
    diff: dict = {}
    for dim in ("count", "negation", "language", "entities", "format"):
        va, vb = a.get(dim), b.get(dim)
        if va != vb:
            if dim == "entities":
                diff[dim] = [sorted(va) if va else None, sorted(vb) if vb else None]
            else:
                diff[dim] = [va, vb]
    return len(diff) == 0, diff
