# Tollgate

Full report structure (§13) is Phase 10 work. This file currently holds one
section written early, out of order, because the numbers it depends on were
freshly measured: the rejected LLM-as-gate alternative.

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
