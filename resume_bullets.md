# Resume bullets

TOLLGATE.md §14, brackets filled from measured numbers only — every figure
below traces to a specific dated entry in [numbers.md](numbers.md). No
bracket survives unfilled; where the original template's bracket shape
would have forced a misleading single number (the held-out detection rate,
see the second bullet), the wording was changed instead of the number.

---

Built an OpenAI-compatible LLM gateway (FastAPI, PostgreSQL/pgvector,
Docker, CI) whose cache admission decision is evaluated as a binary
classifier: **75.3%** hit rate at a measured **10.9%** false-hit rate,
cutting p50 latency from **3654ms** to **472ms**.

> Source: numbers.md "Hit rate" / "False-hit rate at operating point" (ROC
> B, MiniLM+gate, tau=0.95, TPR=0.753/FPR=0.109) and "p50/p99 — hit/miss
> path" (MISS_NO_CANDIDATE p50=3654ms, n=23; HIT_EXACT p50=472ms, n=67).

Showed cosine-only admission is dominated on a **2700**-pair labelled set
(QQP/PAWS + programmatically perturbed instructions); a **0.03ms**
rules-based constraint filter raised AUC from **0.700 to 0.940** (95% CI
[0.929, 0.952]) on a 22M-param embedding model and from **0.914 to 0.992**
(95% CI [0.988, 0.995]) on a 110M-param model — with **0% genuine detection**
on two of three held-out perturbation families never used in rule design
(the third's apparent 99.3% was traced to a numeral-matching artifact, not
real generalization, and is reported as such rather than folded into a
flattering average).

> Source: numbers.md "Corpus size at measurement" (2700), "embed / search /
> gate ms" (gate=0.030ms), "AUC per model per population" (ROC B all-8,
> both models, CIs added 2026-08-04), "In-design vs held-out vs P2
> detection" (register/polarity 0.0%, temporal 99.3%-but-artifact).

Benchmarked GPTCache's default configuration on the same near-duplicate
pairs and found zero discrimination between safe and unsafe reuse in that
regime (it separates unrelated topics correctly — verified with a control —
but the near-duplicate regime is the one a semantic cache exists to
operate in); added a zero-cost output-constraint audit giving a continuous
production false-hit lower bound of my own.

> Source: numbers.md "GPTCache default false-hit rate" (2300/2300 hit,
> TPR=FPR=1.0 on P2/P3, all four populations near-duplicates by
> construction; unrelated-topic control in `scripts/gptcache_smoke.py`
> correctly misses) and "Output-audit failure rate" (§7,
> `semantic_false_hit_rate`, live via `GET /stats`).
