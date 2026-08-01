# Numbers

Updated daily from Day 1 (TOLLGATE.md §15). Values filled in as measured — none yet.

**Standing constraint (set 2026-08-01): no paid models or paid APIs — free
tiers and local models only.** Resolved in Phase 8: second embedding model
is **BAAI/bge-base-en-v1.5** via fastembed (local), not Gemini's embedding
API — checked (`gemini-embedding-001` returns 200, technically available),
but no published free-tier quota exists for it, and the full sweep needs
~5400 embeddings. Given chat generation's real quota turned out to be a
tight ~20/day (not documented anywhere public — found by hitting it), a
bulk job against an equally-undocumented embedding quota risked the same
stalling. This weakens the argument from "even a frontier API model" to
"the failure persists across two local/architecture scales" — noted in
the README.

**Upstream model pinned to `gemini-3.5-flash-lite`, not the
`gemini-flash-latest` alias.** The alias silently drifted mid-project (was
resolving to a different underlying model than when its pricing was first
looked up) — caught via a quota error message, not proactively. A pinned
version can still get deprecated (gemini-2.5-flash and
gemini-2.5-flash-lite both now 404 "no longer available to new users" on
this key, despite being listed in `/models`), but at least it won't change
which model is being paid for without anyone noticing.

## Dominated region (headline finding — README leads with this)

- 2026-08-01, Phase 7, model=MiniLM, cosine-only, at **tau=0.95** (the
  config default): only **21.8%** of genuine paraphrases (P1, QQP) would be
  admitted, while PAWS meaning-flipped pairs (P2) average **0.962**
  similarity — comfortably above 0.95, so they'd be **mostly admitted** at
  that exact same threshold. **No cosine-only threshold gives both an
  acceptable hit rate and an acceptable false-hit rate at once** — raising
  tau to exclude PAWS's near-duplicates also excludes most genuine
  paraphrases; lowering it to keep genuine paraphrases lets most of PAWS's
  meaning-flipped pairs through too. This is the dominated region the
  project's ROC curves visualize, stated as two directly comparable
  numbers instead: **21.8% vs ~96%, both at tau=0.95.** This is the reason
  the constraint gate exists — not an abstract AUC gap, but this specific,
  concrete pair of numbers.

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

- 2026-08-02, Phase 8, ROC B (P3_control+P3), minilm, at tau=0.95 (config
  default): TPR **identical** at 0.753 for cosine-only and gated — the
  gate does not cost any true positives at this operating point (matches
  §6's "can only lower hit rate, never raises it" — here it doesn't even
  lower it, at this specific tau). hit_rate_on_eval_set (unconditional
  P(admit), NOT a production number — see the caveat on that name
  elsewhere): cosine-only 0.555, gated 0.238.

## False-hit rate at operating point (+CI)

- 2026-08-02, Phase 8, ROC B, minilm, tau=0.95: FPR drops from **0.505**
  (cosine-only) to **0.109** (gated) — same TPR, less than a quarter of
  the false-hit rate. This is the concrete headline pairing for README
  Result 2: same hit rate, 4.6x fewer false hits, at the threshold
  currently in config.

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

- **2026-08-02, Phase 8, gated (app/constraints.py wired in), both models:**

  | ROC | model | cosine-only | gated |
  |---|---|---|---|
  | A (P2_control vs P2) | minilm | 0.634 | **0.599** (down) |
  | A (P2_control vs P2) | bge | 0.641 | **0.607** (down) |
  | B all 8 (P3_control vs P3) | minilm | 0.700 | **0.940** |
  | B all 8 (P3_control vs P3) | bge | 0.914 | **0.992** |
  | B in-design 5 only | minilm | 0.728 | 0.995 |
  | B in-design 5 only | bge | 0.910 | 0.997 |
  | B held-out 3 only | minilm | 0.654 | 0.850 |
  | B held-out 3 only | bge | 0.920 | 0.983 |

  **ROC A gets WORSE under gating, on both models — predicted in advance, and
  confirmed by investigation, not just observed.** Root cause: the `entities`
  dimension drives 282/296 (95.3%) of all gate rejections on P2+P2_control,
  and fires on 31.0% of P2_control (the *positives*) vs 43.0% of P2 (the
  negatives) — nearly symmetric, so it incorrectly suppresses true positives
  about as often as it correctly catches true negatives, net-negative for
  AUC. Concrete example: `"...actors from the Open Theater."` vs
  `"...actors from The Open Theatre."` — PAWS's word-scrambling moves "The"
  into/out of sentence-initial position between the two sides, and "Theater"
  vs "Theatre" is a spelling variant, not a real entity difference. This is
  `entities` doing its job (catch topic/subject changes) on text it was
  never designed for (adversarial word-scrambling, a different attack than
  constraint perturbation). **Not fixed** — tuning the gate to rescue ROC A
  would be overfitting to this eval set, the exact thing this project
  criticises elsewhere. Reported as a real, structural scope limitation:
  the gate helps ROC B and costs ROC A (0.634→0.599 minilm, 0.641→0.607
  bge), consistently across two architecturally different embedding models.

## In-design vs held-out vs P2 detection

- 2026-08-02, Phase 8, per-family detection rate (compatible()==False on
  P3, model-independent — the gate never touches embeddings):
  numeric 100.0%, negation 96.0%, language 100.0%, entity 100.0%,
  format 93.3% (in-design mean **97.9%**) — register 0.0%, temporal 99.3%,
  polarity 0.0% (held-out mean **33.1%**, but this mean is misleading, see
  below).

  **temporal's 99.3% is verified NOT to be genuine generalisation.** 149/150
  of its "detections" fire via `entities` alone, catching the bare numeral
  "2019" (present in one side: "Answer as of 2019.") against its absence in
  "Answer as of today." — nothing temporal-aware is happening; it's a
  coincidental artifact of entities' general numeral-catching applied to
  this specific family's specific phrasing, and would NOT catch a
  no-digits temporal contrast ("historically" vs "currently"). The honest
  held-out generalisation signal is **register 0.0% and polarity 0.0%** —
  genuinely uncaught, exactly as designed/expected for constraint types
  with no rule and no lexical overlap with any of the 5 dimensions.
  **Anywhere the 97.9%/33.1% (or the ROC B held-out AUCs, 0.850/0.983)
  appear, this caveat travels with them** — the blended held-out numbers
  are inflated by one family's coincidence, not genuine coverage.

## Generation error rate (100-pair validation)

## Output-audit failure rate

- 2026-08-02, Phase 8 (§7), reported as two SEPARATE rates via GET /stats —
  never pooled, since they mean different things and pooling would
  attribute the LLM's own non-compliance to the cache:
  - **llm_compliance_failure_rate** (HIT_EXACT audit failures — prompt is
    byte-identical to the one that produced the response; a failure is the
    LLM not following its own instruction, not a cache error; the floor
    the semantic path can't beat, since even a perfect cache inherits the
    model's own non-compliance): small n so far from live testing, real
    example observed and fixed — see below.
  - **semantic_false_hit_rate** (HIT_SEMANTIC audit failures — the cached
    response doesn't satisfy the NEW prompt; a genuine false hit, the
    metric this project exists to measure): 0/1 audited during Task 2 live
    verification (n=1, too small to report as a rate yet — Phase 9/10
    replay traffic will give a real n).
  - **Known limitation, fixed same day:** the audit's count-check
    originally only counted bullet/list lines. A live HIT_EXACT on "...in
    **one sentence**." registered `output_audit_ok=false` (diff
    `{"count":[1,None]}`) even though the cached response genuinely was one
    sentence — bullet_count is always None for prose. Fixed in
    `app/cache.py::_count_is_list_type` — the audit now re-checks whether
    the request's count noun is list-type (bullets/points/items/steps)
    before comparing; "N sentences"/"N words" requests are skipped instead
    of failing. Without this fix, llm_compliance_failure_rate would have
    been inflated by every non-bullet count constraint — noted here so the
    fix's motivation isn't lost.

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

- 2026-08-02, Phase 8 (Task 6), 8 real constraint checks via
  gemini-3.5-flash-lite, 2.5s apart, compared against this project's own
  measured hit/miss economics (67 HIT_EXACT + 2 HIT_SEMANTIC, 25 real
  misses, from cache_decisions):
  - LLM-gate check: mean **1508ms**, mean **0.00289¢**
  - Local rules-based gate (app/constraints.py): mean **0.030ms** (1000-call average) — statistically zero against either LLM-gate number
  - Cache hit (weighted mean, exact+semantic): **486ms**, **0¢**
  - Cache miss (real upstream call): **4018ms**, **0.00531¢**
  - Latency a hit saves vs a miss: 3532ms. Cost a hit saves: 0.00531¢.
  - **LLM-gate eats 42.7% of the latency saving and 54.4% of the cost
    saving**, before it even renders a verdict. Worse than a naive
    estimate would suggest: this project's real c_miss is unusually tiny
    (free-tier pricing, sub-cent/call), so the LLM-gate's own cost — using
    the same cheap model — consumes more than half an already-tiny
    savings pool. Full writeup: README.md.
  - Side note, not the point of the measurement: LLM-gate agreed with the
    eval label on 8/8 sampled pairs. Accuracy was never in question;
    rejection is purely economic.

## Cost per 1,000 requests — cached vs uncached

- 2026-08-01: total API spend for this project to date = **$0** — running
  entirely on Gemini's free tier (gemini-3.5-flash-lite, pinned — see the
  standing-constraint note above for why not the "-latest" alias). A real
  result, not a placeholder: it's the reason the standing constraint above
  exists, and worth keeping true through Phase 10.

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

- 2026-08-02, Phase 8 (§9), ROC B gated curve, minilm, real measured
  c_miss=0.0053¢ (mean of 25 real misses). See eval/figures/tau_star.png.
  rho supplied (0.1/0.3/0.5), never the eval set's own base rate — P3/
  P3_control are deliberately adversarial-heavy, using their base rate
  would be exactly the prevalence error this section warns against.
  Because c_miss is real and tiny (free-tier pricing), tau* saturates at
  1.0 for C/c_miss above ~0.2-0.3 regardless of rho — i.e. for any
  false-hit cost even a few tenths of a real miss's cost, the optimal
  policy is "don't admit anything semantically, exact-match only." The
  transition zone (C/c_miss roughly 0.001-0.3) shows tau* dropping from
  1.0 to ~0.70-0.81 depending on rho — higher rho (more genuine near-dups
  in the real workload) tolerates a lower tau* at the same cost ratio,
  since more of what gets admitted is a real win. Representative values:
  at C/c_miss=1 (a false hit costs the same as a miss), tau*=0.70-0.73
  across all three rho; at C/c_miss=10, tau*=0.98-1.00.
