"""§9: nobody chooses a similarity threshold — you choose a cost ratio and
the threshold falls out. Reuses eval/run_sweep.py's fetch/embed/gate
machinery so this measures the same admission rule the service runs, not
a separate reimplementation.

  P(hit)       = rho*TPR(tau) + (1-rho)*FPR(tau)
  P(false_hit) = (1-rho)*FPR(tau)
  E[cost](tau) = (1 - P(hit))*c_miss + P(false_hit)*C
  tau*         = argmin E[cost](tau)

rho is a property of the WORKLOAD (what fraction of real traffic has a
genuine near-duplicate in cache), not of this eval set — P3/P3_control are
deliberately adversarial-heavy, so plugging the eval set's own base rate
in for rho would be exactly the prevalence error this section warns
against. rho is supplied here, not measured.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio

import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

from app.config import settings  # noqa: E402
from app.db import init_pool, pool  # noqa: E402
from eval.run_sweep import (  # noqa: E402
    FIGURES_DIR,
    ROC_GROUPS,
    compute_pair_compatibility,
    cosine_similarity_rows,
    embed_texts_cached,
    fetch_eval_pairs,
    scores,
    sweep_thresholds,
)
from fastembed import TextEmbedding  # noqa: E402

RHOS = (0.1, 0.3, 0.5)
# DECISION-V2: log range is -3 to 2 (0.001x to 100x), not the more
# "obvious" -1 to 3. c_miss is real-measured and tiny (~0.005 cents —
# free-tier Gemini), so tau* saturates at 1.0 for any C/c_miss above ~0.3;
# a -1-to-3 range would render as a flat line at 1.0 for 95% of its width.
# Widening toward smaller ratios is what actually shows the transition.
COST_RATIOS = np.logspace(-3, 2, 80)


def expected_cost(tau_idx: int, tpr: np.ndarray, fpr: np.ndarray, rho: float, c_miss: float, C: float) -> float:
    p_hit = rho * tpr[tau_idx] + (1 - rho) * fpr[tau_idx]
    p_false_hit = (1 - rho) * fpr[tau_idx]
    return (1 - p_hit) * c_miss + p_false_hit * C


def find_tau_star(thresholds: np.ndarray, tpr: np.ndarray, fpr: np.ndarray, rho: float, c_miss: float, C: float) -> float:
    costs = [expected_cost(i, tpr, fpr, rho, c_miss, C) for i in range(len(thresholds))]
    return float(thresholds[int(np.argmin(costs))])


async def get_gated_tpr_fpr(group_key: str = "roc_b", model_key: str = "minilm") -> dict:
    """The gated TPR/FPR curve the cost model optimizes over. Defaults to
    ROC B (what the gate targets) on MiniLM (the production model,
    settings.embedding_model) — the "real" admission behavior, not a
    synthetic one.
    """
    await init_pool()
    from eval.run_sweep import MODELS

    model_name = MODELS[model_key]
    model = TextEmbedding(model_name, cache_dir=settings.fastembed_cache_path)

    group = ROC_GROUPS[group_key]
    df = await fetch_eval_pairs([group["positives"], group["negatives"]])
    y = df["safe_to_reuse"].to_numpy(dtype=bool)

    from eval.run_sweep import _embedding_input

    texts_a = [_embedding_input(t) for t in df["prompt_a"]]
    texts_b = [_embedding_input(t) for t in df["prompt_b"]]
    all_vecs = embed_texts_cached(texts_a + texts_b, model, model_name)
    vec_a, vec_b = all_vecs[: len(texts_a)], all_vecs[len(texts_a):]
    sim = cosine_similarity_rows(vec_a, vec_b)
    compat = compute_pair_compatibility(df)

    score_variants = scores(sim, compat)
    thresholds = np.arange(0.70, 1.001, 0.005)
    sweep_results = sweep_thresholds(y, score_variants, thresholds)
    return {"thresholds": thresholds, "tpr": sweep_results["gated"]["tpr"], "fpr": sweep_results["gated"]["fpr"]}


async def get_real_c_miss() -> float:
    """Mean of actually-measured miss costs from cache_decisions — not a
    theoretical figure. Uses the pool already opened by get_gated_tpr_fpr.
    """
    row = await pool().fetchrow(
        "SELECT avg(cost_cents) AS c, count(*) AS n FROM cache_decisions "
        "WHERE decision LIKE 'MISS%' AND cost_cents > 0"
    )
    if row["n"] == 0:
        raise ValueError("no measured miss costs in cache_decisions yet — can't compute a real c_miss")
    return float(row["c"]), int(row["n"])


def plot_tau_star(thresholds, tpr, fpr, c_miss: float, out_path: Path = None) -> Path:
    out_path = out_path or FIGURES_DIR / "tau_star.png"
    fig, ax = plt.subplots(figsize=(7, 6))
    for rho in RHOS:
        tau_stars = [find_tau_star(thresholds, tpr, fpr, rho, c_miss, C) for C in COST_RATIOS]
        ax.plot(COST_RATIOS, tau_stars, label=f"rho={rho}")
    ax.set_xscale("log")
    ax.set_xlabel("C / c_miss  (cost of one false hit / cost of one miss)")
    ax.set_ylabel("tau* (optimal similarity threshold)")
    ax.set_title(f"Optimal threshold vs cost ratio (c_miss={c_miss:.4f}c, ROC B gated, minilm)")
    ax.legend(title="rho (workload near-dup rate)", fontsize=8)
    ax.set_ylim(0.68, 1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


async def main():
    curve = await get_gated_tpr_fpr()
    c_miss, n = await get_real_c_miss()
    print(f"c_miss = {c_miss:.4f} cents (mean of {n} real measured misses, cache_decisions)")

    path = plot_tau_star(curve["thresholds"], curve["tpr"], curve["fpr"], c_miss)
    print(f"wrote {path}")

    print("\n=== tau* at representative cost ratios ===")
    for rho in RHOS:
        row = []
        for C_over_cmiss in (0.1, 1, 10, 100, 1000):
            C = C_over_cmiss * c_miss
            tau_star = find_tau_star(curve["thresholds"], curve["tpr"], curve["fpr"], rho, c_miss, C)
            row.append(f"C/c_miss={C_over_cmiss}: tau*={tau_star:.3f}")
        print(f"rho={rho}: " + "  ".join(row))

    await pool().close()


if __name__ == "__main__":
    asyncio.run(main())
