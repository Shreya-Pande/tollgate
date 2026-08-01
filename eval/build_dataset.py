"""Phase 7 (TOLLGATE.md §8, Day 2): builds all four eval_pairs populations
from the frozen data/raw/*.parquet files into Neon. Idempotent per
population — re-running after tweaking one population's logic clears and
rebuilds just that population, not the whole table.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from app.db import init_pool, pool  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

FAMILIES_IN_DESIGN = {
    "numeric": ("Answer in exactly 3 bullet points.", "Answer in exactly 10 bullet points."),
    "negation": ("Include examples from Python.", "Do not include anything related to Python."),
    "language": ("Answer in French.", "Answer in German."),
    "entity": ("Focus on Python.", "Focus on Go."),
    "format": ("Answer as a markdown table.", "Answer as a single prose paragraph."),
}
FAMILIES_HELD_OUT = {
    # DECISION-V2: "held out" means "generate pairs, but don't write
    # app/constraints.py rules for these" — a Phase 8 discipline decision,
    # not something this script can enforce technically. Nothing here
    # stops someone from writing a register/temporal/polarity rule later;
    # keeping them held out is a choice made at rule-writing time, not
    # data-generation time.
    "register": ("Explain it for a five-year-old.", "Explain it for a domain expert."),
    "temporal": ("Answer as of 2019.", "Answer as of today."),
    "polarity": ("Focus only on the benefits.", "Focus only on the drawbacks."),
}
ALL_FAMILIES = {**FAMILIES_IN_DESIGN, **FAMILIES_HELD_OUT}

# DECISION-V2: control pairs only cover 3 of the 5 in-design families
# (numeric, language, format), matching the exact 3 examples TOLLGATE.md's
# own step 2.3 gives. negation/entity have no given "trivial reword"
# example, and inventing one wasn't asked for — better to under-cover than
# to guess at what counts as trivial there.
CONTROL_FAMILIES = ("numeric", "language", "format")
NUMERIC_NS = [3, 5, 7, 10]
LANGUAGES = ["French", "German", "Spanish", "Hindi", "Japanese"]
FORMATS = ["markdown table", "JSON object", "bulleted list", "numbered list"]


def control_pair(family: str, rng: np.random.Generator) -> tuple[str, str]:
    if family == "numeric":
        n = rng.choice(NUMERIC_NS)
        return f"Answer in exactly {n} bullet points.", f"Respond using exactly {n} bullet points."
    if family == "language":
        lang = rng.choice(LANGUAGES)
        return f"Answer in {lang}.", f"Respond in {lang}."
    if family == "format":
        fmt = rng.choice(FORMATS)
        return f"Answer as a {fmt}.", f"Give the answer as a {fmt}."
    raise ValueError(f"no control template for family {family!r}")


def make_pair(base: str, suffix_a: str, suffix_b: str) -> tuple[str, str]:
    base = base.rstrip(".")
    return f"{base}. {suffix_a}", f"{base}. {suffix_b}"


def sample_stratified(df: pd.DataFrame, category_col: str, n: int, seed: int) -> pd.DataFrame:
    """~n rows spread evenly across df[category_col]'s distinct values.

    DECISION-V2: equal-per-category, not proportional-to-category-size.
    Proportional sampling would just reproduce Dolly's native imbalance
    (open_qa has ~5x creative_writing's row count) — the opposite of "so
    one task type doesn't dominate," which is the explicit ask.
    """
    rng = np.random.default_rng(seed)
    categories = sorted(df[category_col].unique())
    per_cat = n // len(categories)
    remainder = n - per_cat * len(categories)
    parts = []
    for i, cat in enumerate(categories):
        take = per_cat + (1 if i < remainder else 0)
        pool_df = df[df[category_col] == cat]
        take = min(take, len(pool_df))
        parts.append(pool_df.sample(n=take, random_state=int(rng.integers(0, 2**31))))
    return pd.concat(parts, ignore_index=True)


def build_p1(qqp: pd.DataFrame, n: int = 400, seed: int = 0) -> pd.DataFrame:
    pool_df = qqp[qqp["label"] == 1].copy()
    pool_df = pool_df[
        pool_df["question1"].str.len().between(20, 300)
        & pool_df["question2"].str.len().between(20, 300)
    ]
    # DECISION-V2: random sample (seeded, reproducible), not first-N. QQP
    # row order carries no meaning, but sampling makes that explicit
    # instead of an accidental artifact of parquet row order.
    sample = pool_df.sample(n=min(n, len(pool_df)), random_state=seed)
    return pd.DataFrame(
        {
            "population": "P1",
            "family": None,
            "prompt_a": sample["question1"].values,
            "prompt_b": sample["question2"].values,
            "safe_to_reuse": True,
            "workload_label": None,
        }
    )


def build_p2(paws: pd.DataFrame, n: int = 400, seed: int = 0) -> pd.DataFrame:
    pool_df = paws[paws["label"] == 0].copy()
    pool_df = pool_df[
        pool_df["sentence1"].str.len().between(20, 300)
        & pool_df["sentence2"].str.len().between(20, 300)
    ]
    # DECISION-V2: same choice as build_p1 — random seeded sample, not
    # first-N (see build_p1 for the reasoning).
    sample = pool_df.sample(n=min(n, len(pool_df)), random_state=seed)
    return pd.DataFrame(
        {
            "population": "P2",
            "family": None,
            "prompt_a": sample["sentence1"].values,
            "prompt_b": sample["sentence2"].values,
            "safe_to_reuse": False,
            "workload_label": None,
        }
    )


def build_p2_control(paws: pd.DataFrame, n: int = 400, seed: int = 0) -> pd.DataFrame:
    """PAWS label==1: same word-order-scrambling construction as P2, same
    lexical overlap — but meaning-preserving, not meaning-flipped. Exists
    to fix the same distribution confound P3_control fixes for ROC B: P1
    (QQP) vs P2 (PAWS) conflated "different dataset" with "different
    safety," since PAWS negatives share ~every token by construction
    (mean sim 0.962) while QQP positives are genuine paraphrases with
    different wording (0.864) — a classifier separates the datasets, not
    safe-from-unsafe. P2_control vs P2 is within-PAWS, so it isolates the
    meaning-preserving-vs-flipped signal instead.
    """
    pool_df = paws[paws["label"] == 1].copy()
    pool_df = pool_df[
        pool_df["sentence1"].str.len().between(20, 300)
        & pool_df["sentence2"].str.len().between(20, 300)
    ]
    sample = pool_df.sample(n=min(n, len(pool_df)), random_state=seed)
    return pd.DataFrame(
        {
            "population": "P2_control",
            "family": None,
            "prompt_a": sample["sentence1"].values,
            "prompt_b": sample["sentence2"].values,
            "safe_to_reuse": True,
            "workload_label": None,
        }
    )


def build_p3(dolly: pd.DataFrame, n_base: int = 150, seed: int = 0) -> pd.DataFrame:
    base_sample = sample_stratified(dolly, "category", n_base, seed)
    rows = []
    for _, row in base_sample.iterrows():
        base = row["instruction"]
        for family, (suffix_a, suffix_b) in ALL_FAMILIES.items():
            a, b = make_pair(base, suffix_a, suffix_b)
            rows.append(
                {
                    "population": "P3",
                    "family": family,
                    "prompt_a": a,
                    "prompt_b": b,
                    "safe_to_reuse": False,
                    "workload_label": row["category"],
                }
            )
    return pd.DataFrame(rows)


def build_p3_control(dolly: pd.DataFrame, n_base: int = 100, seed: int = 1) -> pd.DataFrame:
    # DECISION-V2: an independent base-prompt sample (seed=1) rather than
    # reusing P3's exact 150 rows (seed=0). Overlap wouldn't bias anything
    # — these are just base instructions, not the pairs themselves — but
    # an independent draw avoids P3 and P3_control being suspiciously
    # identical in base-prompt composition, which would need explaining
    # later for no benefit.
    base_sample = sample_stratified(dolly, "category", n_base, seed)
    rng = np.random.default_rng(seed)
    rows = []
    for _, row in base_sample.iterrows():
        base = row["instruction"]
        for family in CONTROL_FAMILIES:
            suffix_a, suffix_b = control_pair(family, rng)
            a, b = make_pair(base, suffix_a, suffix_b)
            rows.append(
                {
                    "population": "P3_control",
                    "family": family,
                    "prompt_a": a,
                    "prompt_b": b,
                    "safe_to_reuse": True,
                    "workload_label": row["category"],
                }
            )
    return pd.DataFrame(rows)


INSERT_SQL = """
INSERT INTO eval_pairs (population, family, prompt_a, prompt_b, safe_to_reuse, workload_label)
VALUES ($1, $2, $3, $4, $5, $6)
"""


async def write_population(df: pd.DataFrame) -> None:
    pop = df["population"].iloc[0]
    records = [
        (
            r.population,
            None if pd.isna(r.family) else r.family,
            r.prompt_a,
            r.prompt_b,
            bool(r.safe_to_reuse),
            None if pd.isna(r.workload_label) else r.workload_label,
        )
        for r in df.itertuples(index=False)
    ]
    async with pool().acquire() as conn:
        async with conn.transaction():
            # DECISION-V2: delete-then-insert scoped to THIS population
            # (WHERE population = $1), not a blanket eval_pairs wipe — so
            # re-running after changing one population's logic doesn't
            # disturb the other three. Alternative (append-only) would
            # silently accumulate duplicates on every re-run.
            await conn.execute("DELETE FROM eval_pairs WHERE population = $1", pop)
            await conn.executemany(INSERT_SQL, records)


def report(name: str, df: pd.DataFrame) -> None:
    len_a = df["prompt_a"].str.len()
    len_b = df["prompt_b"].str.len()
    print(f"\n{name}: {len(df)} pairs")
    print(
        f"  prompt_a length: min={len_a.min()} p50={len_a.median():.0f} "
        f"p95={len_a.quantile(0.95):.0f} max={len_a.max()}"
    )
    print(
        f"  prompt_b length: min={len_b.min()} p50={len_b.median():.0f} "
        f"p95={len_b.quantile(0.95):.0f} max={len_b.max()}"
    )
    if df["family"].notna().any():
        print(f"  family counts: {df['family'].value_counts().to_dict()}")
    if df["workload_label"].notna().any():
        print(f"  workload_label counts: {df['workload_label'].value_counts().to_dict()}")


async def main():
    await init_pool()

    dolly = pd.read_parquet(DATA_DIR / "dolly.parquet")
    paws = pd.read_parquet(DATA_DIR / "paws.parquet")
    qqp = pd.read_parquet(DATA_DIR / "qqp.parquet")

    populations = {
        "P1": build_p1(qqp),
        "P2": build_p2(paws),
        "P2_control": build_p2_control(paws),
        "P3": build_p3(dolly),
        "P3_control": build_p3_control(dolly),
    }

    for name, df in populations.items():
        report(name, df)
        await write_population(df)

    total = await pool().fetchval("SELECT count(*) FROM eval_pairs")
    print(f"\neval_pairs total rows: {total}")

    await pool().close()


if __name__ == "__main__":
    asyncio.run(main())
