# Numbers

Updated daily from Day 1 (TOLLGATE.md §15). Values filled in as measured — none yet.

## Exact vs semantic hit split

## Hit rate

## False-hit rate at operating point (+CI)

## AUC per model per population, cosine-only and gated (+CIs)

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

## Corpus size at measurement

## Base rate per population

## τ* surface over (ρ, C/c_miss)
