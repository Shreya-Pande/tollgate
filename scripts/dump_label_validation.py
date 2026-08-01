"""TOLLGATE.md §2.4 (skipped in Phase 7, run now per explicit request):
dump 100 random P3 rows to a text file for hand review. Verify says
"under ~5%" error is the bar. Not run by CI - one-off, human-in-the-loop.

DECISION-V2: plain random sample (seed=0), not stratified by family. The
spec says "100 random P3 rows" - stratifying would make the sample easier
to defend evenly across families but isn't what was asked, and a skewed
draw is itself informative (if errors cluster in one family, that's worth
seeing rather than being smoothed out by forced even coverage).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from app.db import init_pool, pool  # noqa: E402

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "label_validation_sample.txt"


async def main():
    await init_pool()
    async with pool().acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, family, prompt_a, prompt_b, safe_to_reuse "
            "FROM eval_pairs WHERE population = 'P3'"
        )
    await pool().close()

    df = pd.DataFrame([dict(r) for r in rows])
    sample = df.sample(n=100, random_state=0).sort_values("family").reset_index(drop=True)

    lines = [
        "P3 label validation — TOLLGATE.md §2.4",
        "100 random rows (seed=0), all population=P3 (safe_to_reuse=False by construction).",
        "",
        "For each pair: mark WRONG if the label is wrong — a numeric flip that",
        "produced an equivalent prompt, a suffix that made the prompt incoherent,",
        "or any other case where A and B are NOT actually meaningfully different",
        "in the way the family claims. Leave blank / mark OK otherwise.",
        "",
        "Verify bar (TOLLGATE.md): error rate under ~5%. Above that, the",
        "perturbation templates need fixing before this data can be trusted.",
        "",
        "=" * 70,
        "",
    ]
    for i, row in sample.iterrows():
        lines.append(f"[{i + 1}/100]  id={row['id']}  family={row['family']}")
        lines.append(f"  A: {row['prompt_a']}")
        lines.append(f"  B: {row['prompt_b']}")
        lines.append("  Verdict: [ ] OK   [ ] WRONG — why: ______________________")
        lines.append("")

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_PATH} ({len(sample)} rows)")
    print(sample["family"].value_counts())


if __name__ == "__main__":
    asyncio.run(main())
