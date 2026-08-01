# Tollgate

An OpenAI-compatible LLM semantic-cache gateway (FastAPI, PostgreSQL/pgvector,
Docker, CI) whose cache admission decision is evaluated as a binary
classifier, not assumed to be correct because the cosine similarity is high.

Full numbers, with dates and sample sizes, live in [numbers.md](numbers.md).
Anything stated here without a number attached is an opinion, not a result —
check numbers.md before trusting a specific figure quoted from memory.

**Live:** [tollgate-fu81.onrender.com](https://tollgate-fu81.onrender.com)
(Render free tier, Singapore — same region as the Neon database, so the
latency numbers in §8 are meaningful). Free tier sleeps after ~15 minutes
idle; the first request after a sleep is a real 30s+ cold start, not a bug.

## Try it

The exact-match path (no auth needed to see the shape, but the endpoint
itself requires a tenant key — see below for getting one):

```bash
curl https://tollgate-fu81.onrender.com/healthz
```

To see a real `HIT_SEMANTIC` — the cached prompt is "What is the Zhou
Dynasty?"; this asks a differently-worded question and gets the same
cached answer back with no upstream call:

```bash
curl -X POST https://tollgate-fu81.onrender.com/v1/chat/completions \
  -H "Authorization: Bearer $TOLLGATE_DEMO_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "gemini-flash-latest", "messages": [{"role": "user", "content": "What exactly is the Zhou Dynasty?"}], "temperature": 1.0, "max_tokens": 200}'
```

No demo key is committed here on purpose (it's a live credential against a
real, if budget-capped, deployment — a public repo isn't the place for it).
Get your own by running `python scripts/seed_tenant.py` against the same
`DATABASE_URL` (see §11, Setup) — it prints a fresh tenant + key once, not
stored anywhere in plaintext. **First request after an idle period will be
slow** (Render free-tier cold start, 30s+ before anything responds) — this
is expected, not the number in §8; that section's deployed-latency row is
measured warm, server-side, via `total_ms`.

## 1. The question

How often does a semantic cache serve a wrong answer, and how would you
know? Semantic caches ship with a default similarity threshold and, in most
cases, no way to answer that question at all — the threshold is a knob
someone picked, not a number anyone measured.

## 2. Why it isn't observable

A false hit returns HTTP 200 with a fluent, plausible-looking answer. There
is no exception, no malformed response, nothing that shows up in an error
dashboard. The cached response to "explain recursion in 3 bullet points" and
to "explain recursion in 10 bullet points" can both look like a completely
reasonable answer to *someone's* question — just not the one that was asked.
Without ground truth on every request, this failure mode is invisible by
construction.

The signal has to be manufactured: build pairs of prompts with a *known*
ground-truth label (would reusing A's cached response for B be safe, or
not?), and measure how well cosine similarity — and then a constraint filter
on top of it — recovers that label.

## 3. Method

Four populations, 2700 pairs total ([numbers.md, "Corpus size at
measurement"](numbers.md)):

| Population | n | Source | What it tests |
|---|---|---|---|
| P1 | 400 | QQP (genuine duplicate questions) | Admission rate on real paraphrases — no matched negative, reported as a single number |
| P2 / P2_control | 400 / 400 | PAWS (word-scrambled, meaning-flipped vs meaning-preserving) | ROC A — near-identical wording, different meaning |
| P3 / P3_control | 1200 / 300 | Dolly instructions, programmatically perturbed across 8 families vs reworded-but-equivalent | ROC B — the project's headline metric |

**Provenance matters here specifically because of the proxy-label problem**
(§9.4 below): QQP and PAWS give real human-labeled pairs, but they're
sentence pairs, not instruction pairs, and their negatives (PAWS) are a
different kind of "similar but wrong" than what an LLM-gateway constraint
gate is meant to catch. P3 exists to close that gap — instructions are
taken from Dolly (real, if clean, instruction-following data) and perturbed
along 8 axes that constraint-based caching specifically cares about: count,
negation, language, entities, format (**in-design** — the gate has an
explicit rule for these), and register, temporal, polarity (**held-out** —
deliberately *no* rule was written for these, so their detection rate
measures generalization, not memorization of the rule set).

**The proxy-label limitation, stated plainly:** a "paraphrase" label (QQP,
PAWS) is a proxy for "safe to reuse the cached response," not the same
thing. P3/P3_control's perturbation labels are a more direct proxy — but
still programmatic, not organic traffic. See Limitations (§9).

**Split:** the 5 in-design families' perturbation rules were written first;
the 3 held-out families were never looked at while writing
`app/constraints.py`. Their detection rate in Result 2 is a genuine
train/test split, not a resubstitution number.

## 4. Result 1 — cosine-only ROC, two models, per population

([numbers.md, "AUC per model per population"](numbers.md))

| ROC | model | AUC | 95% CI |
|---|---|---|---|
| A (P2_control vs P2) | MiniLM (22M) | 0.634 | [0.598, 0.671] |
| A (P2_control vs P2) | BGE-base (110M) | 0.641 | — |
| B (P3_control vs P3, all 8) | MiniLM | 0.700 | [0.670, 0.735] |
| B (P3_control vs P3, all 8) | BGE-base | 0.914 | — |

**The dominated region, stated as two directly comparable numbers instead
of an abstract AUC gap:** at tau=0.95 (this project's config default), only
**21.8%** of genuine paraphrases (P1, QQP) would be admitted, while PAWS's
meaning-*flipped* pairs (P2) sit at a mean similarity of **0.962** —
comfortably above 0.95, so they'd be *mostly admitted* at that same
threshold. No cosine-only threshold gives an acceptable hit rate and an
acceptable false-hit rate at once: raising tau to exclude PAWS's
near-duplicates also excludes most genuine paraphrases; lowering it to keep
genuine paraphrases lets most of PAWS's meaning-flipped pairs through too.
This dominated region — not an abstract AUC number — is the reason the
constraint gate exists.

Per-family mean cosine similarity within P3 (MiniLM):  entity 0.806, format
0.878, register 0.890, language 0.933, polarity 0.954, negation 0.966,
temporal 0.980, numeric 0.985. **numeric and temporal sit at or above
P3_control's own mean (0.967)** — cosine cannot distinguish "exactly 3" from
"exactly 10," or "as of 2019" from "as of today," at all. temporal is one of
the three held-out families; its Result 2 number should be read with that
in mind before it's even discussed.

## 5. Result 2 — the gate shifts the curve

([numbers.md, "AUC per model per population, cosine-only and gated" and
"In-design vs held-out vs P2 detection"](numbers.md))

| ROC B | model | cosine-only | gated |
|---|---|---|---|
| all 8 | MiniLM | 0.700 | **0.940** |
| all 8 | BGE-base | 0.914 | **0.992** |
| in-design 5 only | MiniLM | 0.728 | 0.995 |
| in-design 5 only | BGE-base | 0.910 | 0.997 |
| held-out 3 only | MiniLM | 0.654 | 0.850 |
| held-out 3 only | BGE-base | 0.920 | 0.983 |

**Best configuration, by this project's own data: BGE + gate (0.992).** An
earlier version of this section recommended MiniLM + gate, reasoning that
the smaller model paired with the gate beat BGE alone (0.940 vs 0.914) *and*
embedded faster. The latency half was wrong — re-measured back-to-back on
identical hardware, BGE embeds in 35.3ms vs MiniLM's 58.3ms (BGE is faster,
confirmed twice, not a fluke; see Result 8). BGE isn't dominated on any
axis — more accurate cosine-only *and* faster to embed than MiniLM — so
there's no case for the smaller model once both numbers are right.

The gate adds real value on top of the strongest embedding model tested, at
negligible cost (+0.078 AUC for ~0.03ms) — it is not a fallback for a weak
model. **How much it adds depends heavily on the base model:** +0.240 AUC on
MiniLM (0.700→0.940) vs +0.078 on BGE (0.914→0.992), roughly a 3x difference
between two embedding models. That gap is itself the argument for
*measuring* the gate's contribution against whatever embedding model is
actually deployed, rather than assuming a fixed benefit — "the gate adds
~0.2 AUC" was already wrong by the second model tested. If the trend
continues with stronger embeddings, a lexical gate like this one could
eventually become redundant for this specific failure mode — that's the
honest extrapolation from two points, not a demonstrated trend.

**Held-out generalization, three ways, never reported pooled:**
in-design detection mean **97.9%** (numeric 100%, negation 96.0%, language
100%, entity 100%, format 93.3%) vs held-out mean **33.1%** — but that mean
is misleading. **register 0.0% and polarity 0.0%** are the honest held-out
signal: genuinely uncaught, exactly as expected for constraint types with no
rule and no lexical overlap with any of the 5 in-design dimensions.
**temporal's 99.3% is verified NOT to be genuine generalization** — 149/150
of its "detections" fire via the `entities` dimension catching a bare
numeral ("2019" present on one side, absent on the other), not anything
temporal-aware; it would not catch a no-digit temporal contrast
("historically" vs "currently"). Wherever 97.9%/33.1% (or the held-out AUCs,
0.850/0.983) appear elsewhere, this caveat travels with them.

**ROC A gets *worse* under gating, on both models — flagged, not
tuned away:** 0.634→0.599 (MiniLM), 0.641→0.607 (BGE). Root cause,
investigated rather than assumed: the `entities` dimension drives 95.3% of
all gate rejections on P2+P2_control, firing on 31.0% of the positives vs
43.0% of the negatives — nearly symmetric, so it suppresses true positives
about as often as it correctly catches true negatives. Concrete example:
`"...actors from the Open Theater."` vs `"...actors from The Open
Theatre."` — PAWS's word-scrambling moves "The" into/out of sentence-initial
position, and "Theater"/"Theatre" is a spelling variant, not a real entity
change. This is `entities` doing its job on text it was never designed for
(adversarial word-scrambling, a different attack than constraint
perturbation). **Deliberately not fixed** — tuning the gate to rescue this
number would be overfitting to the eval set, the exact failure mode this
project exists to catch elsewhere.

**Why rules, not an LLM, for the constraint gate.** The obvious alternative
is asking an LLM to judge prompt-pair compatibility directly — more
flexible, no lexicon to maintain. Timed and costed against this project's
own real hit/miss economics (67 `HIT_EXACT` + 2 `HIT_SEMANTIC`, 25 real
misses, `cache_decisions`), not estimated:

| | Measured |
|---|---|
| LLM-gate check (gemini-3.5-flash-lite): mean latency | 1508ms |
| LLM-gate check: mean cost | 0.00289¢ |
| Local rules-based gate: mean latency (1000-call average) | 0.030ms |
| Cache hit: mean latency | 486ms |
| Cache miss (real upstream call): mean latency / cost | 4018ms / 0.00531¢ |

An LLM-gate check alone — before it renders a verdict — **eats 42.7% of the
latency saving and 54.4% of the cost saving** a cache hit exists to
provide; the rules-based gate's own overhead is statistically zero against
either number. This is a *worse* ratio than a naive estimate suggests,
precisely because this project's real `c_miss` is tiny (free-tier pricing) —
an LLM-gate's own cost, on the same cheap model, consumes more than half an
already-small savings pool. Side note, not the point of the measurement: the
LLM-gate agreed with this project's own eval labels on all 8 sampled pairs —
accuracy was never in question, the rejection is purely economic.

## 6. Result 3 — GPTCache defaults on the same pairs

([numbers.md, "GPTCache default false-hit rate"](numbers.md))

GPTCache (0.1.44), benchmarked on this project's own eval pairs (n=2300:
P2, P2_control, P3, P3_control — all four are near-duplicates by
construction, some safe, some not), pure defaults (`Onnx()` embedding,
`SearchDistanceEvaluation()`, `similarity_threshold=0.8`) — no threshold
tuning, matching the brief of reporting GPTCache's *defaults*, not a swept
best case.

**Precise claim:** on a set deliberately built from near-duplicates,
GPTCache admitted every pair (2300/2300, every population, every
perturbation family) — **it does not discriminate between safe and unsafe
reuse in the near-duplicate regime.** Read as this project's own ROC A/B
operating point (single decision, not swept): TPR=1.0, FPR=1.0 — no
separation at all.

**This is a defaults-calibration finding in the near-duplicate regime, not
a claim that the library is broken.** `scripts/gptcache_smoke.py` puts one
prompt in cache and queries it against a completely unrelated one ("What is
a hash map?" vs "...gardening?") — GPTCache **correctly misses.** It does
separate unrelated topics; it just doesn't separate near-duplicates that
differ in one constraint-relevant way (a swapped count, a negation, a
language, an entity).

**Why the near-duplicate regime is the one that matters, not a corner
case:** separating unrelated topics is the easy half of admission — the
control above shows GPTCache clears that bar trivially. The reason a
*semantic* cache exists at all, rather than plain exact-match, is to handle
near-duplicates. That's exactly the regime this measurement probes, and
exactly where the default threshold shows zero discrimination. A tool that
gets the easy case right and the operating case wrong is a calibration
finding about the default threshold — stated precisely, not overstated.

**Root-caused, not assumed:** the first full run showed the same 100% hit
rate, initially suspected to be a Windows `shutil.rmtree` file-lock bug
leaking state across pairs. Fixed (unique data directory per pair) and
re-run in full — the figure did not change, confirming it's a real property
of the default config on this data.

This project's own gated MiniLM ROC B reaches AUC 0.940 with a real
FPR/TPR tradeoff curve to pick an operating point from (§7); GPTCache's
defaults offer no such curve in the near-duplicate regime — at its own
out-of-the-box setting it doesn't discriminate within that regime at all.

## 7. Choosing an operating point

([numbers.md, "τ\* surface over (ρ, C/c_miss)" and "False-hit rate at
operating point"](numbers.md), `eval/cost_model.py`, `eval/figures/tau_star.png`)

The optimal threshold tau\* depends on two things this project doesn't get
to assume: rho (the real workload's rate of genuine reusable near-dups —
supplied, never taken from the eval set's own base rate, since P3/P3_control
are deliberately adversarial-heavy) and C/c_miss (the cost of a false hit
relative to the cost of a real miss). Computed over a grid of both, using
this project's real measured `c_miss` = 0.0053¢ (mean of 25 real misses):

**Because c_miss is real and tiny (free-tier pricing), tau\* saturates at
1.0 for C/c_miss above roughly 0.2–0.3, regardless of rho** — i.e. for any
false-hit cost even a few tenths of a real miss's cost, the cost-optimal
policy is "don't admit anything semantically, exact-match only." The
transition zone (C/c_miss ≈ 0.001–0.3) shows tau\* dropping from 1.0 to
~0.70–0.81 depending on rho — higher rho tolerates a lower tau\* at the same
cost ratio, since more of what gets admitted is a real win. Representative
values: at C/c_miss=1 (a false hit costs the same as a miss), tau\*=0.70–0.73
across all three rho tested; at C/c_miss=10, tau\*=0.98–1.00.

**This is a genuine, somewhat counterintuitive output of the model, not a
tuning choice: for a workload this cheap to serve on a cache miss, the
economically honest answer is often "don't cache this" once a false hit has
any real cost attached** — the savings from a hit are small in absolute
terms (sub-cent), so it doesn't take much false-hit cost to erase them. This
result should NOT be read as "no semantic caching is ever worth it" —
it's specific to this project's actual, unusually cheap upstream (free-tier
`gemini-3.5-flash-lite`); a workload with an expensive upstream model would
shift the whole curve.

**The operating point actually configured** (tau=0.95, MiniLM+gate): gating
drops the false positive rate from **0.505 to 0.109** while the true
positive rate stays exactly **0.753** — same hit rate, less than a quarter
of the false-hit rate. The same-shape result is expected to hold for BGE
given its own gated ROC B numbers, though the exact FPR/TPR pair at
tau=0.95 wasn't separately pulled for BGE.

## 8. Latency and cost, measured, hardware labelled

([numbers.md, "embed / search / gate ms", "p50/p99 — hit/miss path",
"LLM-gate latency and cost"](numbers.md))

**Local dev hardware: Intel Core i3-1005G1 @ 1.20GHz, CPU-only ONNX
(fastembed), network-bound to Neon (ap-southeast-1) over the open
internet.** These numbers are real but not the ones to trust for
production cost — see the deployed row below, which is what actually
belongs in a resume bullet.

| Stage | Mean (local dev) |
|---|---|
| Embed (MiniLM, single-text, post-warmup) | 58.3–62.2ms |
| Embed (BGE-base, single-text, post-warmup) | **35.3ms** |
| Embed, cold start (first request after server boot) | 899ms |
| Embed, warm (second request) | 112ms |
| Search (cosine scan, n=7 samples — small) | 198.0ms |
| Constraint gate (`app/constraints.py`, 1000-call average) | 0.030ms |
| Upstream call (mean across all logged misses, n=140) | 1473.2ms |

| Path | p50 | p99 | n |
|---|---|---|---|
| HIT_EXACT | 472ms | 612ms | 67 |
| HIT_SEMANTIC | 708–713ms (too small for a percentile) | — | 2 |
| MISS_NO_CANDIDATE | 3654ms | 11940ms (real Gemini variance) | 23 |
| MISS_UPSTREAM_ERROR | 1876ms | 2788ms | 115 (mostly one rate-limited burst) |

**Deployed, Render Singapore (same region as Neon), warm, `total_ms`
measured server-side** — this excludes the client network hop entirely,
unlike the local-dev numbers above:

| decision | p50 | p99 | n |
|---|---|---|---|
| HIT_EXACT | **9.9ms** | 189.8ms | 10 |
| HIT_SEMANTIC | **2668.7ms** | 3461.7ms | 6 |

HIT_EXACT is dramatically faster once the network hop is removed — a
straight hash lookup, no embedding call. **HIT_SEMANTIC is dramatically
*slower* than local dev** (2668.7ms vs 708–713ms), and it isn't the
network: embed_ms alone averaged **2266.7ms** on the deployed instance,
roughly 20–40x the local warm figure (58–112ms), measured server-side
with no DB or network component. The likely explanation is free-tier
CPU throttling/shared-vCPU contention on ONNX inference — plausible, not
independently confirmed. Full breakdown, plus an unresolved anomaly
(the first two live embedding calls on this container came back with
similarity scores that didn't match an offline recomputation of the same
text — most likely a redeploy-transition artifact, flagged rather than
swept under the rug) in numbers.md, "p50/p99 — deployed, Render
Singapore, warm, same region as Neon."

**Cost: $0 total project spend to date** — running entirely on Gemini's
free tier (`gemini-3.5-flash-lite`, pinned; see numbers.md for why not the
`-latest` alias). This is why the LLM-gate alternative (§5) looks as bad as
it does: with `c_miss` this small, an LLM-based gate check costs more,
relatively, than it would against an expensive frontier model.

## 9. Limitations

Named here deliberately, before a reviewer finds them:

1. **Real prompts are messier than Dolly.** The constraint extractor's
   recall on organic production traffic is unknown, and probably lower than
   on the clean, single-instruction Dolly prompts P3 is built from. There
   was no cheap way to measure this given the project's time and quota
   budget.
2. **`safe_to_reuse` may not be transitive.** A matches B and B matches C
   without A being safe for C. This project measures pairs; a real cache
   operates on a growing set where transitivity isn't guaranteed. No answer
   to this within scope.
3. **The exact-match path may be doing most of the work.** An early live
   replay (200 requests, Zipf-weighted repetition, no semantic layer built
   yet) found exact-match alone resolved **75.9%** of successfully-processed
   requests (66/87). If exact matching captures most of the achievable
   savings in a real workload, the semantic layer — and everything this
   project measures about it — carries all of the correctness risk for a
   comparatively small share of the benefit. Small n; a firmer number needs
   more live traffic than this project's LLM quota allowed.
4. **Proxy labels.** A paraphrase label (QQP/PAWS) is a proxy for
   "safe to reuse," not the same thing — sentence pairs, not prompt pairs.
   This is exactly why P3/P3_control exist, and why every population is
   reported separately rather than pooled into one number.
5. **The gate makes ROC A worse, not better** (§5) — a real, measured cost
   on a metric outside the gate's design scope, left unfixed on purpose to
   avoid overfitting the eval set.
6. **Held-out generalization is genuinely 0% on two of three families**
   (register, polarity) — the gate has no coverage at all for constraint
   types with no lexical overlap to its 5 in-design dimensions. The third
   held-out family's apparent 99.3% detection is a numeral-matching
   artifact (§5), not real generalization — the honest held-out number is
   closer to 0% than to 33%.
7. **Generation/label-quality validation was never run.** A planned 100-pair
   manual/LLM validation of P3's programmatically-perturbed prompts (do the
   perturbations actually produce sensible instructions with the intended
   label?) did not happen — cut for LLM quota, not because it was judged
   unnecessary. Noted here rather than silently dropped from numbers.md.
8. **Several numbers rest on very small samples**, flagged inline wherever
   used: `HIT_SEMANTIC` latency (n=2), `search_ms`/`gate_ms` in the live
   decision log (n=7), `MISS_GATE`/`MISS_LOW_SIM` latency (n=1 each). These
   are kept because they're genuine measurements, not because they're
   statistically solid.

## 10. Prior work

**GPTCache** is the closest existing tool to this project, and the one
directly benchmarked (§6): a semantic cache with a fixed similarity
threshold and, out of the box, no compatibility gate. Its caching
infrastructure — storage backends, eviction, multi-provider adapters — is
more mature and complete than anything built here; what's compared is
narrower and specific: its default admission decision within the
near-duplicate regime, where it shows zero discrimination between safe and
unsafe reuse (§6) — while correctly separating unrelated topics, which is
the easier case a semantic cache isn't really being tested on. Everything
else here — the binary-classifier evaluation methodology
(ROC/AUC/CI over labeled pairs), the held-out-family generalization test,
the score-transform trick that makes a hybrid rule+threshold admission rule
representable as one ROC axis, the output-constraint audit as a continuous
production lower bound, and the measured GPTCache-defaults number itself —
is this project's own contribution, not GPTCache's.

**LiteLLM** is a general-purpose LLM proxy/router that also offers an
optional caching layer (including semantic-caching backends). Its scope is
provider abstraction, routing, and observability across many upstream
models — caching-correctness measurement of the kind this project does
isn't its focus. No benchmark of LiteLLM's cache was run here; it's
mentioned for completeness of the landscape, not compared numerically.

**SAFE-CACHE** and **category-aware caching** are both research directions
in the semantic-caching literature that share this project's core
instinct — that raw cosine similarity is an insufficient admission signal
and needs a second check before serving a cached response, whether a
safety/consistency check on top of similarity (SAFE-CACHE) or conditioning
admission on an inferred category/intent rather than embedding distance
alone (category-aware caching). This project's constraint gate is a
lexical, rule-based version of that same idea, evaluated with an explicit
held-out generalization test. Their specific reported numbers were not
independently reproduced in this project — stated here at the level of
general approach, not as a verified comparison, since doing so honestly
would require reading and re-running their original evaluations, which was
out of scope for this project's time budget.

## 11. Setup

**Requirements:** Python 3.12, a Postgres database with the `pgvector`
extension (this project runs against Neon, ap-southeast-1), and either a
local venv or Docker.

```bash
python -m venv .venv
.venv/Scripts/activate       # .venv/bin/activate on Linux/macOS
pip install -r requirements/dev.txt
```

**Environment** (`.env`, see `.env.example`): `DATABASE_URL`,
`EMBEDDING_MODEL`, `UPSTREAM_API_KEY`, `UPSTREAM_BASE_URL`,
`FASTEMBED_CACHE_PATH`. No secrets are committed — `.env` is gitignored.

**Migrations:**

```bash
python migrations/run.py
```

**Run the service:**

```bash
uvicorn app.main:app --reload
```

**Run tests:** `pytest -v` (21 tests; `tests/test_tenant_isolation.py`
needs a real `DATABASE_URL`, everything else is offline).

**Run the eval sweep** (regenerates the ROC/AUC figures in
`eval/figures/`, not committed — see `.gitignore`):

```bash
python eval/run_sweep.py
```

**Docker:** `docker build -t tollgate .` — multi-stage build, the fastembed
model is baked into the image at build time so the first real request
doesn't pay a download. Not locally build-tested (Docker isn't installed in
this project's dev environment) — validated by the `build` job in
`.github/workflows/ci.yml` instead.

**CI** (`.github/workflows/ci.yml`): three jobs — `lint-test` (ruff +
pytest against a `pgvector/pgvector:pg16` service container), `build`
(`docker build`), and `eval-gate` (`eval/ci_eval_gate.py` — asserts the
gated AUC on a frozen 194-pair subset doesn't regress more than 0.02 from a
committed baseline; no model download, no API calls, ~30s). The frozen
subset is regenerated manually via `scripts/freeze_ci_subset.py` when
`eval_pairs` changes enough to warrant a new baseline — not run by CI
itself.

**Deployment:** live at
[tollgate-fu81.onrender.com](https://tollgate-fu81.onrender.com) (Render
free tier, Singapore, same region as Neon). Migrations were applied once
from local against the same Neon instance the deployed service uses —
`python migrations/run.py` is idempotent (tracks applied migrations in
`_migrations`) but isn't part of the deploy step itself; re-run it by hand
only after adding a new migration file.
