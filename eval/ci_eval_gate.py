"""CI merge gate: a statistical property (gated AUC) as a required check.
No model download, no API calls, deterministic, ~30s — the expensive part
(embedding, ~194 pairs) is frozen in ci_frozen_eval.npz. Only compat is
recomputed here, fresh, from the current app/constraints.py — sim is
frozen (the model-dependent part), compat is not (so a regression in the
gate logic itself still shows up here, not just embedding drift).
scripts/freeze_ci_subset.py generates ci_frozen_eval.npz — run manually
when eval_pairs changes enough to warrant a new baseline, not by CI.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.metrics import roc_auc_score

from app.constraints import compatible, extract
from eval.run_sweep import bootstrap_auc, scores

TOLERANCE = 0.02


def main():
    data = np.load(Path(__file__).parent / "ci_frozen_eval.npz", allow_pickle=True)
    y = data["y"]
    sim = data["sim"]
    prompt_a = data["prompt_a"]
    prompt_b = data["prompt_b"]
    baseline_auc = float(data["baseline_auc"][0])

    compat = np.array(
        [compatible(extract(a), extract(b))[0] for a, b in zip(prompt_a, prompt_b)],
        dtype=bool,
    )
    score_variants = scores(sim, compat)
    current_auc = roc_auc_score(y, score_variants["gated"])
    lo, hi = bootstrap_auc(y, score_variants["gated"])

    threshold = baseline_auc - TOLERANCE
    print(f"n={len(y)}")
    print(f"baseline gated AUC (frozen at generation time): {baseline_auc:.4f}")
    print(f"current gated AUC (this run, live app/constraints.py): {current_auc:.4f}  95% CI=[{lo:.3f}, {hi:.3f}]")
    print(f"gate: current >= baseline - {TOLERANCE} = {threshold:.4f}")

    if current_auc < threshold:
        print(f"FAIL: {current_auc:.4f} < {threshold:.4f}")
        sys.exit(1)
    print(f"PASS: {current_auc:.4f} >= {threshold:.4f}")


if __name__ == "__main__":
    main()
