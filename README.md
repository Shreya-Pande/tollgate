# Tollgate

Full report structure (§13) is Phase 10 work. This file currently holds two
sections written early, out of order, because the numbers they depend on
were freshly measured: the headline result, and the rejected LLM-as-gate
alternative.

## The headline result: the gate adds value on top of the strongest model tested

*Corrected same day.* An earlier version of this section recommended
"MiniLM + gate" over BGE alone, reasoning that the smaller model paired
with the gate won on accuracy (0.940 vs 0.914) *and* embedded faster. The
latency claim was wrong — re-measured back-to-back on identical hardware,
BGE embeds in 35.3ms vs MiniLM's 58.3ms (BGE is faster, confirmed twice).
That changes the conclusion: **BGE isn't dominated on any axis** — more
accurate cosine-only *and* faster to embed than MiniLM. There's no case
for the smaller model once both numbers are right, so the recommendation
changes too.

| | ROC B, ungated, cosine-only | single-text embed latency |
|---|---|---|
| MiniLM (22M params) | 0.700 | 58-62ms |
| BAAI/bge-base-en-v1.5 (110M params) | **0.914** | **35.3ms** |

**Best configuration, by this project's own data: BGE + gate.**

    BGE (110M params) alone   = 0.914
    BGE (110M params) + gate  = 0.992

The gate adds real value on top of the strongest embedding model tested,
at negligible cost (+0.078 AUC for ~0.03ms) — it's not a fallback for a
weak model, it's a cheap addition on top of a strong one.

**How much it adds depends heavily on the base model** — +0.240 AUC on
MiniLM (0.700→0.940) vs +0.078 on BGE (0.914→0.992), roughly a 3x
difference in marginal contribution between two embedding models. That
gap is itself the argument for *measuring* the gate's contribution
against whatever embedding model is actually deployed, rather than
assuming a fixed benefit — "the gate adds ~0.2 AUC" was already wrong by
the second model tested. Two points don't establish a trend line, but if
it continues with stronger embeddings, a lexical gate like this one could
eventually become redundant for this specific failure mode. That's the
honest extrapolation, not a weakness to hide — a project that only shows
the case where its own contribution looks largest is less credible than
one that shows the trend working against it.

**The latency inversion has a hypothesis, not a proof.** fastembed ships
pre-exported ONNX per architecture; BGE-base's export is plausibly better
quantized/operator-fused than MiniLM's, independent of parameter count —
not verified. This is "I measured something that contradicted my
expectation, re-measured to confirm it wasn't a fluke, and I have an
untested guess why," stated as exactly that.

**The cleanest single number in the project, unaffected by any of the
above:** at the operating threshold (tau=0.95), gating drops the false
positive rate from **0.505 to 0.109** while the true positive rate stays
exactly **0.753** — the gate removes false hits at zero cost to true
positives (measured on MiniLM; BGE's gated ROC B numbers point the same
direction but the FPR/TPR pair at tau=0.95 specifically wasn't pulled for
BGE separately). See numbers.md, "Hit rate" and "False-hit rate at
operating point."

## Prior work: how this compares to GPTCache's defaults

*Placed here for now; will move into the full §13 "prior work" section.*

GPTCache (0.1.44) is the closest existing tool to this project — a semantic
cache with a fixed similarity threshold and, out of the box, no compatibility
gate. Benchmarked on this project's own eval pairs (n=2300: P2, P2_control,
P3, P3_control — the same populations behind ROC A/B above), using
`init_similar_cache()` with pure defaults (`Onnx()` embedding,
`SearchDistanceEvaluation()`, `similarity_threshold=0.8`), no threshold
tuning.

**Result: GPTCache hit on 2300/2300 pairs (100%) — every population, every
perturbation family, no variation.** Read as a single operating point
(GPTCache wasn't swept, matching its own out-of-the-box config): TPR=1.0,
FPR=1.0. It exercises zero discrimination on this dataset — it hits on a
constraint-violating near-duplicate exactly as often as a genuinely safe one.

This isn't "GPTCache is broken" — a smoke test against a wildly unrelated
pair correctly misses. It's that GPTCache's default threshold is calibrated
for everyday paraphrase reuse, not for the adversarially-constructed
near-duplicates this project's eval set exists to probe (a single swapped
word that flips a count, a negation, a language, an entity — exactly the
failure mode a cosine-only cache is structurally blind to). That's the
project's own opening claim, demonstrated on someone else's tool rather than
just argued in the abstract: **"semantic caches ship with a default
threshold and no way to know how often they serve a wrong answer"** — on
this dataset, GPTCache's default serves the wrong answer 100% of the time
the underlying prompts differ in a way that matters. This project's own
gated MiniLM reaches ROC B (all-8) AUC 0.940, with a real operating point
(FPR 0.109 at TPR 0.753) to choose from — GPTCache's defaults offer no such
curve because nothing in its default config rejects on constraint mismatch
at all.

What's genuinely GPTCache's: the caching infrastructure itself (storage
backends, eviction, multi-provider adapters) is more mature and complete
than anything built here. What's being compared is narrower and specific —
its default admission decision on this project's adversarial eval set, not
the tool as a whole. See numbers.md, "GPTCache default false-hit rate" for
the full breakdown and the Windows file-locking bug that produced a
misleading first run (root-caused and fixed; the 100% figure survived the
fix unchanged).

## Why rules, not an LLM, for the constraint gate (§6)

The obvious alternative to a regex-based constraint extractor is asking an
LLM to judge compatibility directly: "do these two prompts require the same
response?" It's more flexible than a fixed lexicon and doesn't need
maintenance as new constraint types show up. It was tried and rejected —
timed and costed against real numbers, not estimated.

**Method:** 8 real constraint checks through `gemini-3.5-flash-lite` (the
same pinned model the project uses elsewhere), each posed as prompt-pair
compatibility judgment, 2.5s apart. Compared against this project's own
measured hit/miss economics from `cache_decisions` (67 `HIT_EXACT` + 2
`HIT_SEMANTIC`, 25 real misses).

| | Measured |
|---|---|
| LLM-gate check: mean latency | 1508ms |
| LLM-gate check: mean cost | 0.00289¢ |
| Local rules-based gate (`app/constraints.py`): mean latency | 0.030ms (1000-call average) |
| Cache hit: mean latency | 486ms |
| Cache miss (real upstream call): mean latency | 4018ms |
| Cache miss: mean cost | 0.00531¢ |
| **Latency a hit saves vs a miss** | **3532ms** |
| **Cost a hit saves vs a miss** | **0.00531¢** |

An LLM-gate check alone (1508ms, 0.00289¢) — before it even renders a
verdict — **eats 42.7% of the latency saving and 54.4% of the cost saving**
a cache hit exists to provide. The rules-based gate's own overhead
(0.030ms) is statistically zero against either number.

This is a *worse* ratio than a naive estimate would suggest, precisely
because this project's real `c_miss` is unusually small (free-tier
`gemini-3.5-flash-lite` pricing, sub-cent per call) — the LLM-gate's own
cost, using the same cheap model, ends up consuming *more* than half of an
already-tiny savings pool. A judge call against a frontier model, or a
workload with a more expensive upstream model, would make this worse, not
better. The rules-based gate isn't "close enough" to justify on flexibility
— it's cheaper by roughly five orders of magnitude in latency and doesn't
carry its own error rate to evaluate on top of the cache's.

Interesting side note, not the point of this measurement: the LLM-gate
agreed with this project's own eval labels on all 8 sampled pairs. Accuracy
was never in question here — the rejection is purely economic.
