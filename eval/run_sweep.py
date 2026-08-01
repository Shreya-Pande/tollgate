"""Phase 6 built this end to end against synthetic data (generate_synthetic_
pairs, still here as a fallback: `python eval/run_sweep.py --synthetic`).
Phase 7 points the same pipeline (scores, sweep_thresholds, bootstrap_auc,
the plots — none of it changed) at real eval_pairs rows and real
embeddings instead.

No gate yet (app/constraints.py is still empty, Phase 8) — every real-data
score_variants dict below has exactly one key, "cosine_only". Populations
are reported in pairs, never alone: ROC_GROUPS below defines ROC A
(P2_control positives + P2 negatives, both PAWS — within-distribution) and
ROC B (P3_control positives + P3 negatives) — see sweep_thresholds' single-
class guard for why running a population alone isn't just unsupported,
it's actively wrong (P2 and P3 are both all-negatives by construction). P1
(QQP) isn't in an ROC — it has no matched negative, so it's reported as a
standalone admission-rate number instead (p1_admission_rate).

Embeddings are batched (model.embed() on a list, never a per-text loop —
~62ms/single-text measured, ~4800 texts one at a time would be 5+ minutes
per run) and cached to disk keyed by (normalized-text-hash, model_name) in
data/embed_cache/, so re-running the sweep doesn't re-embed anything.
"""

import hashlib
import json
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from dotenv import load_dotenv
from fastembed import TextEmbedding
from sklearn.metrics import roc_auc_score, roc_curve

load_dotenv()

from app.config import settings  # noqa: E402
from app.db import init_pool, pool  # noqa: E402
from app.normalize import normalize  # noqa: E402

FIGURES_DIR = Path(__file__).parent / "figures"
EMBED_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "embed_cache"

# DECISION-V2 (revised): ROC A was originally P1 (QQP) vs P2 (PAWS). That
# AUC came back at 0.150 — a real number, but not a finding about safety.
# PAWS negatives share ~every token by construction (mean sim 0.962) while
# QQP positives are genuine paraphrases with different wording (0.864); a
# classifier on that pair separates the two DATASETS, not safe-from-unsafe
# — the same distribution confound P3_control exists to fix for ROC B,
# just unfixed here. P2_control (PAWS label==1: same word-scrambling
# construction, meaning-preserving instead of meaning-flipped) makes ROC A
# within-PAWS instead, isolating the actual signal. P1 is no longer in any
# ROC — see p1_admission_rate() for its standalone report.
ROC_GROUPS = {
    "roc_a": {
        "positives": "P2_control",
        "negatives": "P2",
        "label": "ROC A — PAWS meaning-preserving (P2_control) vs meaning-flipped (P2)",
    },
    "roc_b": {
        "positives": "P3_control",
        "negatives": "P3",
        "label": "ROC B — same-distribution control (P3_control) vs constraint perturbation (P3)",
    },
}


def generate_synthetic_pairs(n: int = 500, overlap: float = 0.03, seed: int = 0):
    """Labelled fake pairs with a realistic shape: positives (safe_to_reuse
    =True) cluster at higher similarity (mean 0.93), negatives at slightly
    lower (mean 0.90) — same means TOLLGATE.md's own reference snippet
    uses. `compatible` is correlated with y but imperfect: negatives are
    still occasionally "compatible" (an organic near-miss the gate wasn't
    designed to catch), positives are always compatible.

    overlap controls how much the two similarity distributions bleed into
    each other: it's the stddev of the per-pair noise added on top of the
    0.03 gap between the class means. Small overlap (e.g. 0.005) keeps the
    classes cleanly separated -> a clean ROC. Large overlap (e.g. 0.08)
    swamps the 0.03 gap -> a messy, close-to-diagonal ROC. Try both to see
    the difference before there's real data to be messy for real reasons.

    Returns:
      y: bool array (n,) — true label, safe_to_reuse
      sim: float array (n,) — cosine similarity, clipped to [0, 1]
      compatible: bool array (n,) — constraint-gate compatibility
    """
    rng = np.random.default_rng(seed)
    y = rng.random(n) > 0.5
    sim = np.clip(np.where(y, 0.93, 0.90) + rng.normal(0, overlap, n), 0, 1)
    compatible = np.where(y, True, rng.random(n) > 0.4)
    return y, sim, compatible


def scores(sim: np.ndarray, compatible: np.ndarray) -> dict[str, np.ndarray]:
    """# I'll write this

    The score-transform trick from TOLLGATE.md step 1.8: the gated rule
    (admit iff sim >= tau AND compatible) isn't a threshold on a
    continuous score, so it can't go into roc_curve/roc_auc_score
    directly. Mapping incompatible pairs to a value below any realistic
    threshold makes "gated" a plain threshold on a transformed score,
    putting both variants on one comparable axis.

    In:
      sim: float array, shape (n,) — cosine similarities, e.g. from
           generate_synthetic_pairs() or, in Phase 7, a real cosine scan.
      compatible: bool array, shape (n,) — constraint-gate compatibility.

    Out:
      dict[str, np.ndarray] with exactly two keys, each shape (n,):
        "cosine_only" — sim, unchanged.
        "gated"       — sim where compatible, else a sentinel below any
                        threshold ever swept (e.g. -1.0).
    """
    return {
        "cosine_only": sim,
        # DECISION-V2: sentinel -1.0 (cosine's theoretical floor), matching
        # the spec's reference snippet. Alternative: -np.inf, robust to any
        # future sweep range but risks NaN/inf handling elsewhere. -1.0 only
        # breaks if a threshold sweep ever includes tau <= -1.0 (ours never
        # does: tau in [0.70, 1.0]).
        "gated": np.where(compatible, sim, -1.0),
    }


def bootstrap_auc(
    y: np.ndarray, score: np.ndarray, n: int = 1000, seed: int = 0
) -> tuple[float, float]:
    """95% CI for AUC via the percentile bootstrap. Resamples that land on
    a single class are skipped (roc_auc_score is undefined there) rather
    than left to crash the run; if every resample does (degenerate y or
    tiny n), that's surfaced as a clear error instead of np.percentile
    silently choking on an empty list.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    out = []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[s])) < 2:
            continue
        out.append(roc_auc_score(y[s], score[s]))
    if not out:
        raise ValueError(
            "bootstrap_auc: every resample was single-class — can't compute a CI "
            "(check class balance / n)"
        )
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


# hit_rate_on_eval_set below is P(admit) on THIS eval set, not a production
# hit rate — the eval set is deliberately loaded with adversarial
# near-misses (P2/P3 in Phase 7), so its base rate has nothing to do with
# production traffic's. Same prevalence trap §9 warns about with rho in the
# cost model, surfacing here first. Name is deliberate — never shorten to
# hit_rate, and never let it end up in a resume bullet unqualified.
def sweep_thresholds(
    y: np.ndarray, scores_dict: dict[str, np.ndarray], thresholds: np.ndarray
) -> dict[str, dict[str, np.ndarray]]:
    """TOLLGATE.md §8's threshold sweep: for each tau in `thresholds` and
    each score variant, compute the confusion-matrix rates at that
    operating point (admit iff score >= tau). Generator-agnostic — works
    unchanged whether sim/compatible came from generate_synthetic_pairs()
    or, in Phase 7, real embeddings and a real cosine scan.

    In:
      y: bool array, shape (n,) — true labels.
      scores_dict: dict[str, np.ndarray] — one score array per variant,
                   each shape (n,); this is scores()'s output.
      thresholds: float array, shape (m,) — tau values to sweep, e.g.
                  np.arange(0.70, 1.001, 0.005).

    Out:
      dict[str, dict[str, np.ndarray]] — one key per score variant
      (matching scores_dict's keys), each mapping to a dict with keys
      "threshold", "tpr", "fpr", "hit_rate_on_eval_set", every array shape
      (m,), aligned to `thresholds`:
        tpr = P(admit | safe_to_reuse)
        fpr = P(admit | not safe_to_reuse)
        hit_rate_on_eval_set = P(admit), unconditional
    """
    y = np.asarray(y, dtype=bool)
    n_pos, n_neg, n = int(np.sum(y)), int(np.sum(~y)), len(y)
    # DECISION-V2 (revised from the Phase 6 version, which returned silent
    # 0.0 here instead): a single-class y now raises immediately. P2 and
    # P3 are both all-negatives on their own, so running either alone used
    # to give plausible-looking zeros and an undefined AUC rather than an
    # error — dangerous with real data. Populations get paired for
    # reporting (P1+P2, P3_control+P3), so this should never actually fire
    # in normal use; if it does, something upstream passed an unpaired
    # population, and that should be loud, not a quiet wrong number.
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"sweep_thresholds: y is single-class (n_pos={n_pos}, n_neg={n_neg}) "
            "— TPR/FPR/hit_rate are undefined here. Pair this population with "
            "its counterpart (e.g. P1+P2, or P3_control+P3) instead of "
            "sweeping it alone."
        )
    thresholds = np.asarray(thresholds, dtype=float)

    out = {}
    for name, score in scores_dict.items():
        score = np.asarray(score)
        tpr_list, fpr_list, hit_rate_list = [], [], []
        for tau in thresholds:
            # DECISION-V2: admit iff score >= tau (not >). A pair scoring
            # exactly tau admits. Only matters at the sentinel boundary
            # (scores()'s -1.0) if tau were ever <= -1.0 — this project's
            # sweep range (0.70-1.0) never reaches it.
            admit = score >= tau
            tp = int(np.sum(admit & y))
            fp = int(np.sum(admit & ~y))
            # n_pos/n_neg/n are guaranteed > 0 by the guard above, so these
            # divisions are always safe — including at both degenerate
            # thresholds (admits nothing -> 0.0/0.0/0.0; admits everything
            # -> 1.0/1.0/1.0), with no NaN and no special-casing needed.
            tpr_list.append(tp / n_pos)
            fpr_list.append(fp / n_neg)
            hit_rate_list.append((tp + fp) / n)
        out[name] = {
            "threshold": thresholds,
            "tpr": np.array(tpr_list),
            "fpr": np.array(fpr_list),
            "hit_rate_on_eval_set": np.array(hit_rate_list),
        }
    return out


def plot_roc(
    y: np.ndarray,
    scores_dict: dict[str, np.ndarray],
    out_path: Path = None,
    title: str = "ROC — cosine-only vs gated",
) -> Path:
    """One or more variants on one axis, AUC + 95% CI in each legend label."""
    out_path = out_path or FIGURES_DIR / "roc.png"
    fig, ax = plt.subplots(figsize=(7, 6.5))
    for name, score in scores_dict.items():
        fpr, tpr, _ = roc_curve(y, score)
        auc = roc_auc_score(y, score)
        lo, hi = bootstrap_auc(y, score)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f}, 95% CI [{lo:.3f}, {hi:.3f}])")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("\n".join(textwrap.wrap(title, width=48)), fontsize=11)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_hit_rate_vs_false_hit_rate(
    sweep_results: dict[str, dict[str, np.ndarray]],
    out_path: Path = None,
    title: str = "Hit rate vs false-hit rate",
) -> Path:
    """§8: 'more legible operationally than TPR/FPR'. x = false-hit rate
    (fpr), y = hit_rate_on_eval_set (unconditional P(admit) on THIS eval
    set, not tpr and not a production hit rate) — takes
    sweep_thresholds()'s output directly, one curve per score variant.
    """
    out_path = out_path or FIGURES_DIR / "hit_rate_vs_false_hit_rate.png"
    fig, ax = plt.subplots(figsize=(7, 6.5))
    for name, r in sweep_results.items():
        ax.plot(r["fpr"], r["hit_rate_on_eval_set"], marker="o", markersize=2, label=name)
    ax.set_xlabel("False-hit rate")
    ax.set_ylabel("Hit rate (this eval set, not production — see §9)")
    ax.set_title("\n".join(textwrap.wrap(title, width=48)), fontsize=11)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main_synthetic():
    """Phase 6's demo path. Kept as an explicit fallback (`--synthetic`) —
    not the default now that eval_pairs has real data (main_real, below)."""
    y, sim, compatible = generate_synthetic_pairs(n=500, overlap=0.03, seed=0)

    score_variants = scores(sim, compatible)
    roc_path = plot_roc(y, score_variants)
    print(f"wrote {roc_path}")

    thresholds = np.arange(0.70, 1.001, 0.005)
    sweep_results = sweep_thresholds(y, score_variants, thresholds)
    hr_path = plot_hit_rate_vs_false_hit_rate(sweep_results)
    print(f"wrote {hr_path}")

    for name, score in score_variants.items():
        auc = roc_auc_score(y, score)
        lo, hi = bootstrap_auc(y, score)
        print(f"{name}: AUC={auc:.3f}  95% CI=[{lo:.3f}, {hi:.3f}]")


def _cache_key(normalized_text: str, model_name: str) -> str:
    return hashlib.sha256(f"{model_name}\x00{normalized_text}".encode()).hexdigest()


def embed_texts_cached(texts: list[str], model: TextEmbedding, model_name: str) -> np.ndarray:
    """Batched embedding (model.embed() on the whole list — never a loop)
    with an on-disk cache keyed by (sha256 of the text actually fed to the
    model, model_name), so re-running the sweep only embeds what's new.
    `texts` should already be normalize()'d — the cache key is over what
    the model sees, not the raw prompt, so two raw texts that normalize
    identically correctly share one cache entry instead of two.

    Cache is one (matrix.npy, index.json) pair per model, appended to
    incrementally — not one file per text, which would be thousands of
    tiny files for ~4800+ pairs.
    """
    EMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = model_name.replace("/", "_")
    matrix_path = EMBED_CACHE_DIR / f"{safe_name}.npy"
    index_path = EMBED_CACHE_DIR / f"{safe_name}_index.json"

    if matrix_path.exists() and index_path.exists():
        matrix = np.load(matrix_path)
        index = json.loads(index_path.read_text())
    else:
        matrix = np.zeros((0, 0), dtype=np.float32)
        index = {}

    keys = [_cache_key(t, model_name) for t in texts]
    missing = [(i, k, t) for i, (k, t) in enumerate(zip(keys, texts)) if k not in index]

    if missing:
        missing_texts = [t for _, _, t in missing]
        new_vectors = np.array(list(model.embed(missing_texts)), dtype=np.float32)
        start_idx = matrix.shape[0] if matrix.size else 0
        matrix = new_vectors if matrix.size == 0 else np.vstack([matrix, new_vectors])
        for offset, (i, k, t) in enumerate(missing):
            index[k] = start_idx + offset
        np.save(matrix_path, matrix)
        index_path.write_text(json.dumps(index))

    return np.stack([matrix[index[k]] for k in keys])


def cosine_similarity_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between two (n, dim) arrays. Explicit
    re-normalization here rather than trusting fastembed's own claimed
    unit-norm output — TOLLGATE.md §10.4's own warning about not assuming
    normalization happened upstream. Clipped to [0, 1] as a float-noise
    safety net, same as generate_synthetic_pairs does."""
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    sim = np.sum(a_norm * b_norm, axis=1)
    return np.clip(sim, 0.0, 1.0)


async def fetch_eval_pairs(populations: list[str]) -> pd.DataFrame:
    rows = await pool().fetch(
        "SELECT population, family, prompt_a, prompt_b, safe_to_reuse, workload_label "
        "FROM eval_pairs WHERE population = ANY($1::text[])",
        populations,
    )
    return pd.DataFrame([dict(r) for r in rows])


def _embedding_input(text: str) -> str:
    """Same normalize() production applies before embedding — TOLLGATE.md's
    non-negotiable: the eval must exercise the same code path as the
    service, or the number it produces is fiction."""
    return normalize([{"role": "user", "content": text}])


async def run_roc_group(
    group_key: str, model: TextEmbedding, model_name: str, thresholds: np.ndarray
) -> dict:
    group = ROC_GROUPS[group_key]
    df = await fetch_eval_pairs([group["positives"], group["negatives"]])
    if df.empty:
        raise ValueError(
            f"{group_key}: no rows for populations {group['positives']!r}/"
            f"{group['negatives']!r} — run eval/build_dataset.py first"
        )

    y = df["safe_to_reuse"].to_numpy(dtype=bool)
    texts_a = [_embedding_input(t) for t in df["prompt_a"]]
    texts_b = [_embedding_input(t) for t in df["prompt_b"]]

    all_vecs = embed_texts_cached(texts_a + texts_b, model, model_name)
    vec_a, vec_b = all_vecs[: len(texts_a)], all_vecs[len(texts_a):]
    sim = cosine_similarity_rows(vec_a, vec_b)

    # No gate yet (app/constraints.py is empty — Phase 8). One key only.
    score_variants = {"cosine_only": sim}

    roc_path = plot_roc(
        y, score_variants, out_path=FIGURES_DIR / f"{group_key}.png", title=group["label"]
    )
    sweep_results = sweep_thresholds(y, score_variants, thresholds)
    hr_path = plot_hit_rate_vs_false_hit_rate(
        sweep_results,
        out_path=FIGURES_DIR / f"{group_key}_hit_rate.png",
        title=f"{group['label']} — hit rate vs false-hit rate",
    )

    auc = roc_auc_score(y, sim)
    lo, hi = bootstrap_auc(y, sim)
    n_pos, n_neg = int(y.sum()), int((~y).sum())

    print(f"\n=== {group['label']} ===")
    print(f"n={len(y)}  n_pos={n_pos} ({group['positives']})  n_neg={n_neg} ({group['negatives']})")
    print(f"cosine_only: AUC={auc:.3f}  95% CI=[{lo:.3f}, {hi:.3f}]")
    print(f"wrote {roc_path}")
    print(f"wrote {hr_path}")

    return {
        "group": group_key,
        "label": group["label"],
        "n": len(y),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "auc": auc,
        "ci": (lo, hi),
    }


async def p1_admission_rate(
    model: TextEmbedding, model_name: str, thresholds_of_interest=(0.90, 0.95, 0.99)
) -> dict:
    """P1 (QQP genuine duplicates) has no matched negative population
    anymore, so it's not an ROC — reported as a single number instead:
    what fraction of genuine paraphrases would be admitted at a given
    threshold. That answers "does the cache hit anything at all," which is
    what P1 was for, without pretending to be a classifier comparison it
    can't support alone. 0.95 is the current config default
    (SIMILARITY_THRESHOLD).
    """
    df = await fetch_eval_pairs(["P1"])
    texts_a = [_embedding_input(t) for t in df["prompt_a"]]
    texts_b = [_embedding_input(t) for t in df["prompt_b"]]
    all_vecs = embed_texts_cached(texts_a + texts_b, model, model_name)
    vec_a, vec_b = all_vecs[: len(texts_a)], all_vecs[len(texts_a):]
    sim = cosine_similarity_rows(vec_a, vec_b)

    print(f"\n=== P1 admission rate (QQP genuine duplicates, n={len(sim)}) ===")
    rates = {}
    for tau in thresholds_of_interest:
        rate = float(np.mean(sim >= tau))
        rates[tau] = rate
        flag = "  <- current config default" if abs(tau - settings.similarity_threshold) < 1e-9 else ""
        print(f"  at tau={tau}: {rate:.1%} of genuine paraphrases would be admitted{flag}")
    print(f"  mean sim={sim.mean():.3f}  median={np.median(sim):.3f}")
    return {"n": len(sim), "mean_sim": float(sim.mean()), "rates": rates}


def plot_family_similarity(p3_df: pd.DataFrame, control_mean: float, out_path: Path = None) -> Path:
    """The most valuable output of this phase: per-family mean cosine
    similarity within P3, against the P3_control baseline. Low bars =
    cosine already separates that family from a safe reworded pair. Bars
    at the control line = cosine cannot tell the difference at all —
    temporal in particular sits at ~0.98, right against the control line,
    which is also one of the three held-out families: expect near-zero
    held-out detection for it in Phase 8. That's the data saying so
    upfront, not a Phase 8 bug — held-out vs in-design is colored
    separately here so it isn't mistaken for one later.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    from eval.build_dataset import FAMILIES_HELD_OUT

    out_path = out_path or FIGURES_DIR / "p3_family_similarity.png"
    means = p3_df.groupby("family")["sim"].mean().sort_values()
    colors = ["#d95f02" if f in FAMILIES_HELD_OUT else "#1b9e77" for f in means.index]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(means.index, means.values, color=colors)
    ax.axvline(control_mean, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Mean cosine similarity within P3 (lower = more separable from a safe reworded pair)")
    ax.set_title("P3 mean similarity by family")
    handles = [
        Patch(color="#1b9e77", label="in-design (rules planned, Phase 8)"),
        Patch(color="#d95f02", label="held-out (no rules — generalization test)"),
        Line2D([0], [0], color="gray", linestyle="--", label=f"P3_control mean ({control_mean:.3f})"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


async def plot_p3_family_breakdown(model: TextEmbedding, model_name: str) -> Path:
    df = await fetch_eval_pairs(["P3", "P3_control"])
    texts_a = [_embedding_input(t) for t in df["prompt_a"]]
    texts_b = [_embedding_input(t) for t in df["prompt_b"]]
    all_vecs = embed_texts_cached(texts_a + texts_b, model, model_name)
    vec_a, vec_b = all_vecs[: len(texts_a)], all_vecs[len(texts_a):]
    df["sim"] = cosine_similarity_rows(vec_a, vec_b)

    p3 = df[df["population"] == "P3"]
    control_mean = df[df["population"] == "P3_control"]["sim"].mean()

    path = plot_family_similarity(p3, control_mean)
    print(f"\n=== P3 per-family similarity (P3_control mean={control_mean:.3f}) ===")
    print(p3.groupby("family")["sim"].agg(["mean", "median"]).sort_values("mean").to_string())
    print(f"wrote {path}")
    return path


async def main_real() -> dict:
    await init_pool()
    model_name = settings.embedding_model
    model = TextEmbedding(model_name, cache_dir=settings.fastembed_cache_path)
    thresholds = np.arange(0.70, 1.001, 0.005)

    results = {}
    for group_key in ROC_GROUPS:
        results[group_key] = await run_roc_group(group_key, model, model_name, thresholds)
    results["p1"] = await p1_admission_rate(model, model_name)
    await plot_p3_family_breakdown(model, model_name)

    await pool().close()
    return results


if __name__ == "__main__":
    if "--synthetic" in sys.argv:
        main_synthetic()
    else:
        import asyncio

        asyncio.run(main_real())
