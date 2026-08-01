"""Phase 6: measurement pipeline, wired end to end, before there's any real
data in it. Phase 7 swaps generate_synthetic_pairs() for real eval_pairs
rows + real embeddings — everything downstream (scores, sweep_thresholds,
bootstrap_auc, the plots) takes plain (y, sim, compatible) arrays and
doesn't care where they came from.

Performance note for Phase 7, not acted on here: embed with model.embed()
on a list of texts (fastembed batches internally), never in a per-text
loop. At ~62ms/single-text encode (measured, see numbers.md), looping over
~4800 real pairs is ~5 minutes per sweep run, and the sweep gets re-run
often.
"""

import numpy as np

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

FIGURES_DIR = Path(__file__).parent / "figures"


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
            # DECISION-V2: n_pos/n_neg/n == 0 returns 0.0 rather than
            # raising or propagating NaN. Alternative: raise ValueError on
            # a degenerate (single-class or empty) eval set. Picked silent
            # 0.0 so a malformed call doesn't crash the whole sweep;
            # bootstrap_auc still raises loudly for the equivalent
            # per-resample case, so this isn't unguarded elsewhere. Also
            # what makes both degenerate thresholds (admits nothing /
            # admits everything) land on clean 0.0/0.0/0.0 or 1.0/1.0/1.0
            # instead of NaN, for a normal (both-classes-present) eval set.
            tpr_list.append(tp / n_pos if n_pos else 0.0)
            fpr_list.append(fp / n_neg if n_neg else 0.0)
            hit_rate_list.append((tp + fp) / n if n else 0.0)
        out[name] = {
            "threshold": thresholds,
            "tpr": np.array(tpr_list),
            "fpr": np.array(fpr_list),
            "hit_rate_on_eval_set": np.array(hit_rate_list),
        }
    return out


def plot_roc(
    y: np.ndarray, scores_dict: dict[str, np.ndarray], out_path: Path = None
) -> Path:
    """Both variants on one axis, AUC + 95% CI in each legend label."""
    out_path = out_path or FIGURES_DIR / "roc.png"
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, score in scores_dict.items():
        fpr, tpr, _ = roc_curve(y, score)
        auc = roc_auc_score(y, score)
        lo, hi = bootstrap_auc(y, score)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f}, 95% CI [{lo:.3f}, {hi:.3f}])")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC — cosine-only vs gated")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_hit_rate_vs_false_hit_rate(
    sweep_results: dict[str, dict[str, np.ndarray]], out_path: Path = None
) -> Path:
    """§8: 'more legible operationally than TPR/FPR'. x = false-hit rate
    (fpr), y = hit_rate_on_eval_set (unconditional P(admit) on THIS eval
    set, not tpr and not a production hit rate) — takes
    sweep_thresholds()'s output directly, one curve per score variant.
    """
    out_path = out_path or FIGURES_DIR / "hit_rate_vs_false_hit_rate.png"
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, r in sweep_results.items():
        ax.plot(r["fpr"], r["hit_rate_on_eval_set"], marker="o", markersize=2, label=name)
    ax.set_xlabel("False-hit rate")
    ax.set_ylabel("Hit rate (this eval set, not production — see §9)")
    ax.set_title("Hit rate vs false-hit rate")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
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


if __name__ == "__main__":
    main()
