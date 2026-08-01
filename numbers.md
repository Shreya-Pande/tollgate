# Numbers

Updated daily from Day 1 (TOLLGATE.md §15). Values filled in as measured — none yet.

**Standing constraint (set 2026-08-01): no paid models or paid APIs — free
tiers and local models only.** Relevant to Phase 8: TOLLGATE.md §8 calls for
a frontier API embedding model as the second model in the two-model sweep.
That's out under this constraint. The second model will be either Gemini's
free embedding API or a larger local fastembed model (e.g.
BAAI/bge-base-en-v1.5) — decide when Phase 8 is reached.

## Exact vs semantic hit split

- 2026-08-01, Phase 7 (Task 6), live replay of 200 requests through the
  running service (Zipf-weighted repetition — a few "hot" prompts repeated
  many times, most sent 0-1 extra times, 22 distinct prompts landed in
  cache_entries, top hit_count=18) — **no semantic layer exists yet
  (app/cache.py empty, Phase 7-8)**, so this measures exact-match alone:
  `MISS_UPSTREAM_ERROR`=113 (56.5%, Gemini free-tier per-minute quota hit
  during the burst — each one correctly refunded to $0 and logged, verified
  against cache_decisions/tenants.spent_cents), `HIT_EXACT`=66 (33%),
  `MISS_NO_CANDIDATE`=21 (10.5%). Restricting to successful requests only:
  **HIT_EXACT / (HIT_EXACT + MISS_NO_CANDIDATE) = 66/87 = 75.9%** — under
  this replay's repetition pattern, exact-match alone resolved over
  three-quarters of successfully-processed requests with zero semantic
  matching built. Small n (87); a firmer number comes from Phase 8's
  traffic-shaped eval, but this is an early, real data point toward "exact
  matching captures most of the achievable savings" (§17.3).

## Hit rate

## False-hit rate at operating point (+CI)

## AUC per model per population, cosine-only and gated (+CIs)

- 2026-08-01, Phase 7, model=sentence-transformers/all-MiniLM-L6-v2 (fastembed), cosine-only, no gate yet (app/constraints.py empty until Phase 8):
  - **ROC A, corrected** (P2_control PAWS meaning-preserving, n=400 vs P2 PAWS
    meaning-flipped, n=400 — both word-order-scrambled, same distribution):
    AUC = **0.634**, 95% CI [0.598, 0.671]. *Superseded finding, kept for the
    record:* the original ROC A paired P1 (QQP) against P2 (PAWS) and got
    AUC=0.150 — below chance. That number was real but was a distribution
    confound, not a safety signal: PAWS negatives share ~every token by
    construction (mean sim 0.962, median 0.978) while QQP positives are
    genuine paraphrases with different wording (mean sim 0.864, median
    0.880), so the classifier was separating datasets, not safe-from-unsafe.
    P2_control (PAWS label==1, same construction, meaning-preserving) fixes
    this the same way P3_control fixes it for ROC B. The underlying
    observation is still worth stating plainly: **any threshold that admits
    real paraphrases (mean sim 0.864) also admits nearly all of PAWS's
    meaning-flipped pairs (mean sim 0.962)** — a stronger, more defensible
    sentence than either AUC alone.
  - **ROC B** (P3_control same-distribution positive, n=300 vs P3 constraint perturbation, n=1200): AUC = **0.700**,
    95% CI [0.670, 0.735]. Low-to-middling, as expected/desired — real signal, well short of ceiling, meaning the
    constraint gate has genuine room to move this. Per-family mean similarity within P3 (see
    eval/figures/p3_family_similarity.png — the most valuable figure this phase produced): entity 0.806, format
    0.878, register 0.890, language 0.933, polarity 0.954, negation 0.966, temporal 0.980, numeric 0.985. numeric and
    temporal sit at/above P3_control's own mean (0.967) — cosine cannot tell "exactly 3" from "exactly 10" apart, or
    "as of 2019" from "as of today," at all. **temporal is one of the three held-out families** (no gate rule planned
    for it) — expect its Phase 8 held-out-detection number to be near zero. That's the data saying so in advance, not
    a Phase 8 bug. entity and format (0.806, 0.878) are already the most separable by cosine alone.
  - **P1 admission rate** (QQP genuine duplicates, n=400 — not an ROC, no matched negative population; "does the
    cache hit anything at all"): at tau=0.90, 42.5% of genuine paraphrases would be admitted; at tau=0.95 (current
    config default), 21.8%; at tau=0.99, 3.8%. mean sim=0.864, median=0.880.

## In-design vs held-out vs P2 detection

## Generation error rate (100-pair validation)

## Output-audit failure rate

## GPTCache default false-hit rate

## p50/p99 — hit path

- 2026-08-01, local dev (network-bound to Neon ap-southeast-1), n=1 (Phase 7 will
  give real percentiles): HIT_EXACT total_ms = 486ms, vs 8291ms for the miss
  it was cached from — embed_ms/upstream_ms both NULL, as designed (skipped
  entirely on the exact-match path).

## p50/p99 — miss path

- 2026-08-01, local dev (network-bound to Neon ap-southeast-1), n=2:
  MISS_NO_CANDIDATE total_ms = 8291ms (first, in-process cold embed) and
  4460ms (second, warm embed) — dominated by upstream_ms (3983ms / 3686ms),
  i.e. Gemini's own response latency, not this app's overhead.

## embed / search / gate ms

- 2026-08-01, local dev (Intel Core i3-1005G1 @ 1.20GHz, CPU-only ONNX via fastembed):
  embed (single-text, post-warmup mean over 100 encodes) = 62.17ms, dim=384, norm=1.0
- 2026-08-01, local dev, in-process via the running FastAPI server (not the
  standalone smoke test above) — same model, module-level singleton loaded
  at import time, but still shows real first-call vs warm variance:
  cold (first request after server start) embed_ms = 899ms;
  warm (second request) embed_ms = 112ms. The cold figure reflects ONNX
  session warm-up (thread pool spin-up) that a loaded-at-import model
  doesn't avoid on its very first inference call — worth noting since it
  contradicts the "load at import = no cold cost" assumption at a glance.

## LLM-gate latency and cost (rejected alternative)

## Cost per 1,000 requests — cached vs uncached

- 2026-08-01: total API spend for this project to date = **$0** — running
  entirely on Gemini's free tier (gemini-flash-latest). A real result, not a
  placeholder: it's the reason the standing constraint above exists, and
  worth keeping true through Phase 10.

## Corpus size at measurement

- 2026-08-01, Phase 7, eval_pairs total = 2700 rows: P1=400, P2=400,
  P2_control=400 (added after ROC A's revision — PAWS label==1), P3=1200
  (150 base Dolly instructions x 8 families, 5 in-design + 3 held-out),
  P3_control=300 (100 base instructions x 3 reworded families).

## Base rate per population

- 2026-08-01: P1 safe_to_reuse=True 400/400 (QQP label==1 only). P2
  safe_to_reuse=False 400/400 (PAWS label==0). P2_control safe_to_reuse=True
  400/400 (PAWS label==1). P3 safe_to_reuse=False 1200/1200. P3_control
  safe_to_reuse=True 300/300. Every population is single-class by
  construction — this is exactly why they're reported in pairs (ROC A =
  P2_control+P2, ROC B = P3_control+P3), never alone, and why P1 is a
  standalone number rather than an ROC (no matched negative population); see
  sweep_thresholds' single-class guard in eval/run_sweep.py.

## τ* surface over (ρ, C/c_miss)
