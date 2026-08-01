import time

from fastembed import TextEmbedding

from app.config import settings

# Loaded at import time, not inside the request handler — loading the model
# per-request costs ~2s.
_model = TextEmbedding(
    settings.embedding_model,
    cache_dir=settings.fastembed_cache_path,
)


def encode(text: str) -> tuple[list[float], float]:
    """Single-text encode for the serving path — one request at a time.

    eval/ scripts embedding thousands of texts (Phase 7's build_dataset.py,
    Phase 8's two-model sweep) MUST batch via model.embed(list_of_texts)
    instead of looping this function per text — fastembed batches
    internally, and at ~62ms/single-text (measured, see numbers.md), a
    loop over ~4800+ texts costs 5+ minutes per sweep run instead of
    seconds. The serving path stays single-text because a real request
    has exactly one prompt to embed.
    """
    t0 = time.perf_counter()
    vector = list(_model.embed([text]))[0]
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return vector, elapsed_ms
