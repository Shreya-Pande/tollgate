"""Unrelated-topic control for the GPTCache defaults comparison
(numbers.md, "GPTCache default false-hit rate"). Confirms GPTCache's
default threshold isn't simply broken/always-hit: two obviously-unrelated
prompts correctly miss. The main measurement (2300 near-duplicate pairs,
100% hit) probes a harder regime this doesn't touch — see numbers.md for
why that regime, not this one, is the one a semantic cache exists to
handle.

Requires the separate `.venv-gptcache` environment (GPTCache 0.1.44 needs
`transformers<5`; see numbers.md's standing note on why it can't share
this project's main venv). Not run by CI - one-off manual verification.
"""
import shutil
import time
from pathlib import Path

from gptcache import Cache
from gptcache.adapter.api import get, init_similar_cache, put
from gptcache.config import Config

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "gptcache_smoke_data"
shutil.rmtree(DATA_DIR, ignore_errors=True)

t0 = time.perf_counter()
cache_obj = Cache()
init_similar_cache(data_dir=str(DATA_DIR), cache_obj=cache_obj, config=Config())  # pure defaults
print(f"init took {time.perf_counter() - t0:.1f}s (model download/load, one-time)")

put("What is a hash map?", "answer A", cache_obj=cache_obj)

result = get("What is a hash map, explained briefly?", cache_obj=cache_obj)
print("result (paraphrase, expect a hit):", result)

result2 = get("What is a completely unrelated topic about gardening?", cache_obj=cache_obj)
print("result (unrelated, expect None):", result2)

shutil.rmtree(DATA_DIR, ignore_errors=True)
