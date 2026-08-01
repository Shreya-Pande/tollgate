"""Generates eval/ci_frozen_eval.npz — a ~200-pair frozen subset of ROC B
(the headline metric) for the CI eval-gate (eval/ci_eval_gate.py). Not
run by CI itself; run locally/manually whenever eval_pairs changes enough
to warrant a new baseline, and commit the resulting .npz.

sim is frozen (the expensive, model-dependent part); compat is NOT — see
eval/ci_eval_gate.py for why (a regression in app/constraints.py must
still be catchable, not baked into a frozen "baseline" that can never
disagree with itself).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from app.config import settings  # noqa: E402
from app.db import init_pool, pool  # noqa: E402
from eval.run_sweep import (  # noqa: E402
    ROC_GROUPS,
    _embedding_input,
    compute_pair_compatibility,
    cosine_similarity_rows,
    embed_texts_cached,
    fetch_eval_pairs,
    scores,
)
from fastembed import TextEmbedding  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

OUT_PATH = Path(__file__).resolve().parent.parent / "eval" / "ci_frozen_eval.npz"


async def main():
    await init_pool()
    model_name = settings.embedding_model
    model = TextEmbedding(model_name, cache_dir=settings.fastembed_cache_path)

    group = ROC_GROUPS["roc_b"]
    df = await fetch_eval_pairs([group["positives"], group["negatives"]])

    # Stratified sample: proportional draw from P3_control (family=None,
    # treated as one bucket) and each P3 family, targeting ~200 total so
    # no single family dominates a small subset.
    parts = []
    control = df[df["population"] == "P3_control"]
    parts.append(control.sample(n=min(50, len(control)), random_state=0))
    p3 = df[df["population"] == "P3"]
    per_family = 150 // p3["family"].nunique()
    for fam in sorted(p3["family"].unique()):
        pool_fam = p3[p3["family"] == fam]
        parts.append(pool_fam.sample(n=min(per_family, len(pool_fam)), random_state=0))
    sample = pd.concat(parts, ignore_index=True)
    print(f"frozen subset: {len(sample)} rows")
    print(sample["population"].value_counts())

    y = sample["safe_to_reuse"].to_numpy(dtype=bool)
    texts_a = [_embedding_input(t) for t in sample["prompt_a"]]
    texts_b = [_embedding_input(t) for t in sample["prompt_b"]]
    all_vecs = embed_texts_cached(texts_a + texts_b, model, model_name)
    vec_a, vec_b = all_vecs[: len(texts_a)], all_vecs[len(texts_a):]
    sim = cosine_similarity_rows(vec_a, vec_b)
    compat = compute_pair_compatibility(sample)  # for the printed baseline only, not saved

    score_variants = scores(sim, compat)
    baseline_auc = roc_auc_score(y, score_variants["gated"])
    print(f"baseline gated AUC on frozen subset (current app/constraints.py): {baseline_auc:.4f}")

    np.savez(
        OUT_PATH,
        y=y,
        sim=sim,
        prompt_a=sample["prompt_a"].to_numpy(dtype=object),
        prompt_b=sample["prompt_b"].to_numpy(dtype=object),
        baseline_auc=np.array([baseline_auc]),
        model_name=np.array([model_name]),
    )
    print(f"wrote {OUT_PATH}")

    await pool().close()


if __name__ == "__main__":
    asyncio.run(main())
