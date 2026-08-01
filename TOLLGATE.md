# Tollgate

An OpenAI-compatible LLM gateway whose semantic-cache admission decision is treated as a binary classifier and evaluated as one.

**The claim this project defends:**

> Semantic caches ship with a default similarity threshold and no way to know how often they serve a wrong answer. I built the measurement, found the ROC is dominated where you'd want to operate, added a hot-path constraint filter that shifts the curve, and reported the number for my cache and for GPTCache's defaults.

Everything in this document exists to support that sentence. If a task doesn't, cut it.

---

## Contents

**Part I — Design**
[1. Stack](#1-stack) · [2. Architecture](#2-architecture) · [3. Schema](#3-schema) · [4. Request path](#4-request-path) · [5. Admission decision](#5-admission-decision) · [6. Constraint extractor](#6-constraint-extractor) · [7. Output-constraint audit](#7-output-constraint-audit) · [8. Evaluation design](#8-evaluation-design) · [9. Cost model](#9-cost-model) · [10. Measurement hygiene](#10-measurement-hygiene)

**Part II — Build**
[Day 0 Setup](#day-0--setup-2-hours) · [Day 1 Plumbing](#day-1--plumbing) · [Day 2 The measurement](#day-2--the-measurement-becomes-real) · [Day 3 The contribution](#day-3--the-contribution) · [Day 4 The defence](#day-4--the-defence) · [Day 5 Buffer and ship](#day-5--buffer-then-ship)

**Part III — Delivery**
[Repo layout](#11-repo-layout) · [CI](#12-ci) · [README](#13-readme-as-a-report) · [Resume bullets](#14-resume-bullets) · [Numbers](#15-numbers-to-capture) · [Cut order](#16-cut-order) · [Limitations](#17-limitations-name-these-before-a-reviewer-finds-them)

**Appendices**
[A. Gotchas](#appendix-a--the-five-things-most-likely-to-cost-you-an-hour) · [B. Daily stop conditions](#appendix-b--daily-stop-conditions)

---

# Part I — Design

## 1. Stack

Nothing here is decoration. A technology earns its place only if it closes a stated skill gap or the project cannot be built without it.

### Tier 1 — you will be interviewed on these

| Tech | Role | Have an answer for |
|---|---|---|
| **FastAPI** + Uvicorn | HTTP layer | Why async (the path is I/O-bound: embed → DB → upstream). Dependency injection. Pydantic validation. |
| **PostgreSQL** | Cache, decision log, budgets | Indexes and query plans. Why the decision log and cache entry commit together. Atomic `UPDATE ... RETURNING` for budgets, and what breaks under contention. |
| **pgvector** | Vector column, cosine distance | `vector(384)`, the `<=>` operator, why exact scan at this corpus size, and at what size that stops being true. |
| **Docker** + compose | Local Postgres, deployable image | Layer caching, why `pgvector/pgvector:pg16`, dev/prod parity. |
| **GitHub Actions** | CI, including the eval gate | Why a statistical property is a merge gate, and how you made it deterministic. |
| **sentence-transformers** | Embedding on the hot path | What an embedding is, why cosine, why *local* rather than an API model (measure it: ~50–150ms network against a ~25ms hit budget). |
| **scikit-learn** | `roc_curve`, `auc`, bootstrap CIs | The project's thesis lives here. Prevalence, base rates, why AUC needs a confidence interval. |

### Tier 2 — near-zero study cost

`asyncpg` (you write the SQL yourself) · `pydantic` (arrives with FastAPI) · `httpx` (requests, awaited) · `numpy` / `pandas` / `matplotlib` (already yours — this promotes them from listed to shipped) · `pytest` + `pytest-asyncio` · `ruff` · HuggingFace `datasets` (one call per dataset) · `GPTCache` (black box; you need its default config, not its internals).

### Deliberately absent

| Not used | Why, when asked |
|---|---|
| ORM (SQLAlchemy / Alembic) | Five tables, about a dozen queries. An ORM adds an abstraction to explain without removing any SQL I still need to know. Migrations are numbered `.sql` files and a small runner. |
| ANN index (HNSW / IVFFlat) | A few thousand entries; a brute-force scan is single-digit ms. An index would be an unmeasured optimisation for a scale I don't have. Here's the size at which I'd add one. |
| Redis | Budget counters need the same transactional guarantee as the spend log. One atomic `UPDATE` in Postgres does it — no second datastore, and the counter and the ledger cannot disagree. |
| Load-test tool | p50/p99 come from `percentile_cont` over my own decision log, which I'm writing anyway. |
| ONNX / quantisation | Latency isn't the thesis. I report measured latency with the hardware labelled instead. |
| Multi-provider fallback | Two API schemas for a feature that doesn't support the claim. |
| Shadow re-call sampler | The output-constraint audit (§7) gives a production false-hit signal for free, with no upstream call and no judge. |

**Hosting:** Fly.io or Render free tier + Neon (pgvector enabled). Two dashboards, not two technologies.

---

## 2. Architecture

```
  client ──POST /v1/chat/completions──► FastAPI
                                          │
                    1. exact-match on sha256(RAW prompt)      ──► HIT_EXACT (zero risk)
                    2. normalise · extract constraints (<1ms)
                    3. embed (local, ~5-15ms)
                    4. exact cosine scan, scoped by
                       tenant + model + params (~5ms)
                    5. admission: sim ≥ τ AND constraints compatible
                          │
                 HIT ─────┴───── MISS
                  │               │
        output-constraint    httpx → upstream (800-2000ms)
        audit (free, §7)     store entry + embedding
                  │               │
                  └──► cache_decisions row (EVERY request) ◄──┘

  PostgreSQL: tenants · cache_entries · cache_decisions · eval_pairs

  OFFLINE (plain scripts, importing the same modules the service uses):
    eval/build_dataset.py · eval/run_sweep.py · eval/cost_model.py · eval/bench_gptcache.py
```

**Non-negotiable:** the offline scripts import `app/constraints.py` and `app/embedder.py`. If the eval measures a different code path than production, the number is fiction.

---

## 3. Schema

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE tenants (
  id            UUID PRIMARY KEY,
  name          TEXT NOT NULL,
  budget_cents  INTEGER NOT NULL DEFAULT 10000,
  spent_cents   NUMERIC(12,4) NOT NULL DEFAULT 0
);

CREATE TABLE cache_entries (
  id                UUID PRIMARY KEY,
  tenant_id         UUID NOT NULL REFERENCES tenants(id),
  prompt_raw        TEXT NOT NULL,
  prompt_hash       TEXT NOT NULL,          -- sha256 of the RAW prompt
  prompt_normalized TEXT NOT NULL,          -- embedding input only
  constraints       JSONB NOT NULL,
  embedding         VECTOR(384) NOT NULL,
  embedding_model   TEXT NOT NULL,
  upstream_model    TEXT NOT NULL,
  params_hash       TEXT NOT NULL,          -- system prompt, temperature, max_tokens
  response_text     TEXT NOT NULL,
  response_tokens   INTEGER NOT NULL,
  cost_cents        NUMERIC(10,6) NOT NULL,
  hit_count         INTEGER NOT NULL DEFAULT 0,
  created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX ON cache_entries (tenant_id, prompt_hash, upstream_model, params_hash);
CREATE INDEX ON cache_entries (tenant_id, upstream_model, params_hash);

CREATE TABLE cache_decisions (
  id                UUID PRIMARY KEY,
  tenant_id         UUID NOT NULL REFERENCES tenants(id),
  prompt_raw        TEXT NOT NULL,
  constraints       JSONB NOT NULL,
  embedding         VECTOR(384) NOT NULL,
  candidate_id      UUID REFERENCES cache_entries(id),
  cosine_similarity REAL,
  threshold_used    REAL NOT NULL,
  gate_passed       BOOLEAN,
  constraint_diff   JSONB,                  -- WHICH dimension differed
  decision          TEXT NOT NULL,          -- HIT_EXACT | HIT_SEMANTIC | MISS_LOW_SIM
                                            -- | MISS_GATE | MISS_NO_CANDIDATE
  output_audit_ok   BOOLEAN,
  output_audit_diff JSONB,
  embed_ms REAL, search_ms REAL, gate_ms REAL, upstream_ms REAL, total_ms REAL,
  cost_cents        NUMERIC(10,6) NOT NULL DEFAULT 0,
  embedding_model   TEXT NOT NULL,
  workload_label    TEXT,
  created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE eval_pairs (
  id             SERIAL PRIMARY KEY,
  population     TEXT NOT NULL,   -- P1 | P2 | P3 | P3_control
  family         TEXT,            -- perturbation family; NULL for P1/P2
  prompt_a       TEXT NOT NULL,
  prompt_b       TEXT NOT NULL,
  safe_to_reuse  BOOLEAN NOT NULL,
  workload_label TEXT
);
```

Three things to defend out loud:

- The cache key includes `tenant_id` and `params_hash` — cross-tenant reuse is a data leak, and a different system prompt or temperature is a different question.
- `prompt_hash` is over the **raw** prompt, so the exact path carries zero false-hit risk by construction. Normalisation is lossy and belongs only on the fuzzy path.
- `constraint_diff` is stored, not just `gate_passed`. "Blocked" is undebuggable; "the bullet count changed" is the most quotable output in the project.

---

## 4. Request path

| # | Step | Budget |
|---|---|---|
| 1 | Parse and validate (Pydantic, OpenAI chat-completions shape) | <1ms |
| 2 | Resolve tenant from API key; atomic budget check | ~1ms |
| 3 | `sha256(raw)` exact-match lookup → return if hit | ~2ms |
| 4 | Normalise (lowercase, collapse whitespace, canonicalise messages) | <1ms |
| 5 | Extract constraint vector (§6) | <1ms |
| 6 | Embed (local sentence-transformer) | 5–15ms |
| 7 | Exact cosine scan, `ORDER BY embedding <=> $1 LIMIT k`, scoped | ~5ms |
| 8 | Admission decision (§5), `k` swept over {1,3,5} | <1ms |
| 9a | HIT → output-constraint audit, return, `hit_count++` | ~20ms total |
| 9b | MISS → upstream, store entry, add cost **in the same transaction as the decision row** | 800–2000ms |
| 10 | Always write `cache_decisions` — every request, no sampling | |

Budget enforcement in one statement:

```sql
UPDATE tenants SET spent_cents = spent_cents + $1
WHERE id = $2 AND spent_cents + $1 <= budget_cents
RETURNING spent_cents;
```

No explicit lock, no read-modify-write race, budget enforced in SQL rather than in application code. Returns `NULL` when over budget. Know the failure mode — row contention on one hot tenant — and the fix: sharded counters.

---

## 5. Admission decision

```
for c in candidates:                        # sorted by similarity desc
    if c.sim < τ:              return MISS_LOW_SIM
    ok, diff = compatible(req_constraints, c.constraints)
    if not ok:  log(diff);     continue
    return HIT_SEMANTIC
return MISS_GATE
```

Both `τ` and `k` are swept. Report the hit-rate / false-hit-rate surface over (τ, k) rather than defending a chosen pair. Anywhere you can replace a judgement call with a curve, do it — that is the ethos of the whole project.

---

## 6. Constraint extractor

Pure Python — regex plus lexicons. No model on the hot path.

Implement **five** dimensions:

| Dimension | Catches |
|---|---|
| `count` | cardinal near an output noun — "3 bullets" vs "10 bullets" |
| `negation` | negation cue and its scope — "list X" vs "list things that are **not** X" |
| `language` | "in / into / to \<language\>" — French vs German |
| `entities` | capitalised tokens and numerals, as a set — "Python and Java" vs "Python and Go" |
| `format` | lexicon → bullets / table / json / code / prose |

**Compatibility:** for every dimension, either both `None` or equal (set equality for `entities`). Strict. A `HIT_SEMANTIC` requires `sim ≥ τ` **and** compatible.

Three things to say before you're asked:

- **Why rules, not an LLM.** Time it: an LLM check is ~300ms and ~$0.0002 against a cached call worth ~1200ms and ~$0.002. It eats a third of the latency saving and a tenth of the cost saving, and carries its own error rate you'd then have to evaluate. Measure it, record it, reject it.
- **Why it can only lower hit rate.** It is a post-retrieval filter — it removes hits, never adds them. The mechanism is that *at a fixed false-hit budget it lets you lower τ, and lowering τ is what recovers hit rate.* Memorise that sentence; the naive version is wrong and will be caught.
- **Where it fails.** Constraints not expressible lexically ("make it sound more optimistic"), constraints in the system prompt, languages outside your lexicons. Precision high, recall moderate — report both.

---

## 7. Output-constraint audit

On every HIT, before returning:

1. Constraints of the **incoming** prompt — already extracted.
2. Observable properties of the **cached response**: bullet count, table present, JSON valid, detected language, length band.
3. Compare; write `output_audit_ok` and `output_audit_diff`.

No upstream call, no judge, no sampling. About thirty lines.

This is a **continuously measured lower bound on production false-hit rate**, produced by a different mechanism than the offline ROC and not dependent on the eval set being representative. When asked how you'd know the cache was misbehaving in production, this is the answer.

---

## 8. Evaluation design

**Base corpus:** `databricks-dolly-15k` — human-written instructions with task-category labels, which give you `workload_label` for free. Verify the licence on Day 0. OASST1 is the fallback.

**Four populations. Report separately, never pool.**

| Pop | Built from | Size | Label | Establishes |
|---|---|---|---|---|
| **P1** natural paraphrase | QQP duplicates | ~400 | `True` | True-positive rate — does the cache hit anything at all |
| **P2** natural near-miss | PAWS negatives | ~400 | `False` | False-hit rate on *organic* confusions the gate wasn't designed for — **the honesty control** |
| **P3** constraint perturbation | Dolly × 8 template families | ~1,200 | `False` | The failure mode the gate targets |
| **P3-control** positive control | Dolly, same constraint reworded | ~300 | `True` | Removes the distribution confound (below) |

**The confound P3-control fixes.** P1 comes from QQP and P3 from Dolly, so they are different distributions — a classifier could separate them on writing style alone, and part of your AUC would be an artifact rather than the constraint signal. P3-control gives P3 positives and negatives from a single distribution, so P3's AUC means what you claim it means.

**Breaking the circularity.** Generate P3 with **eight** families; implement gate rules for **five**. Held out: `register shift`, `temporal shift`, `polarity shift`. Report in-design families, held-out families, and P2 as three separate numbers. Held-out detection will be lower — **report it anyway.** A project that shows where its own method stops working reads as far more credible than one flattering number. Cut the held-out families and you are testing a regex against strings the same regex generated, and every adversarial number becomes worthless.

**Validation, not labelling.** Read 100 generated P3 pairs, confirm the labels, record the generation error rate. Forty minutes, and it is the answer to *"how do you know your labels are correct?"*

**The sweep** — `eval/run_sweep.py`, roughly 200 lines, and it must not become a framework. No plugin registry, no config schema, no metric abstraction; if you start designing those you are rebuilding DeepEval and should stop that day.

```
for model in [minilm, frontier_api]:        # brackets the capability range
    embed all pairs
    for τ in arange(0.70, 1.001, 0.005):
        for k in [1, 3, 5]:
            cosine_only  →  TPR, FPR, hit_rate
            with_gate    →  TPR, FPR, hit_rate
    roc_curve, auc, bootstrap CI ×1000
```

Two models rather than four: if the failure survives both the smallest and the strongest representation, the middle is interpolation. That is a better argument than sampling four points, and it directly answers *"you used a weak embedding model."*

**Figures:** ROC cosine-only vs gated, both models, per population · hit rate vs false-hit rate (more legible operationally than TPR/FPR) · AUC table with CIs · **false hits broken down by constraint dimension** · in-design vs held-out vs P2 detection.

---

## 9. Cost model

Inputs: `c_miss` (measured — upstream cost plus a latency penalty), `C` (operator-set cost of one false hit), `ρ` (fraction of incoming prompts with a genuine near-duplicate in cache — a property of the **workload**, not of your eval set).

```
P(hit)       = ρ·TPR(τ) + (1−ρ)·FPR(τ)
P(false_hit) = (1−ρ)·FPR(τ)
E[cost](τ)   = (1 − P(hit))·c_miss + P(false_hit)·C
τ*           = argmin E[cost](τ)
```

Deliverable: τ* against `C/c_miss` on a log axis, one curve per ρ ∈ {0.1, 0.3, 0.5}. Where τ* saturates at 1.0, that workload should not be semantically cached — and it falls out of the arithmetic instead of being an opinion.

This is the answer to *"who chose 0.95?"* — **nobody chooses a threshold; you choose a cost ratio and the threshold falls out.**

**Do not plug the eval set's base rate in for ρ.** The eval set is deliberately adversarial-heavy; that is the prevalence error and it will give you an absurdly strict τ*.

---

## 10. Measurement hygiene

1. **Split.** Choose τ on a dev split, report on held-out.
2. **Base rate.** Report each population's class balance explicitly; prefer precision–recall alongside ROC.
3. **Log everything.** A sampled decision log biases the ROC.
4. **Assert `sim ∈ [0,1]`** in a test. `<=>` is cosine *distance*; make sure you are not double-normalising.
5. **Latency honestly.** Local-warm and deployed as separate rows, hardware labelled on each.
6. **Measure the exact/semantic split from Day 2.** See §17.3.

---

# Part II — Build

Every step has an estimate, the work, a **✓ Verify** you can actually run, and a **⚠** for what will bite you. Do not advance past a failing Verify — each one is placed where a silent failure would otherwise surface two days later.

## Day 0 — Setup (2 hours)

Every item here eats a morning if you hit it cold on Day 1.

### 0.1 Repo and environment (10 min)

```bash
mkdir tollgate && cd tollgate && git init
python3.12 -m venv .venv && source .venv/bin/activate
python -V   # must print 3.12.x
```

### 0.2 Dependencies, split into two files (10 min)

`requirements.txt` — what ships in the Docker image:

```
fastapi==0.115.6
uvicorn[standard]==0.34.0
asyncpg==0.30.0
pgvector==0.3.6
pydantic==2.10.4
httpx==0.28.1
sentence-transformers==3.3.1
python-dotenv==1.0.1
```

`requirements-eval.txt` — offline only, never in the image:

```
-r requirements.txt
scikit-learn==1.6.0
pandas==2.2.3
numpy==2.2.1
matplotlib==3.10.0
datasets==3.2.0
pytest==8.3.4
pytest-asyncio==0.25.0
ruff==0.8.4
```

```bash
pip install -r requirements-eval.txt
```

**⚠** Pin versions. A silent minor-version bump mid-week is the stupidest possible way to lose three hours. Keeping sklearn/matplotlib/datasets out of the image is a deliberate choice you should be able to defend — a serving container has no business carrying a plotting library.

### 0.3 Postgres with pgvector (15 min)

`docker-compose.yml`:

```yaml
services:
  db:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_PASSWORD: tollgate
      POSTGRES_DB: tollgate
    ports: ["5432:5432"]
    volumes: ["pgdata:/var/lib/postgresql/data"]
volumes:
  pgdata:
```

```bash
docker compose up -d
docker compose exec db psql -U postgres -d tollgate \
  -c "CREATE EXTENSION IF NOT EXISTS vector;" \
  -c "SELECT '[1,2,3]'::vector <=> '[1,2,4]'::vector AS cosine_distance;"
```

**✓ Verify** — a small non-zero float prints. That single query proves the image, the extension and the distance operator all work.

**⚠** `<=>` is cosine **distance**, not similarity. `similarity = 1 - distance`. Getting this backwards inverts your entire ROC and the curve will look bizarre rather than wrong.

### 0.4 Datasets, downloaded and frozen (30 min)

```python
# scripts/fetch_data.py
from datasets import load_dataset
from pathlib import Path
Path("data/raw").mkdir(parents=True, exist_ok=True)

load_dataset("databricks/databricks-dolly-15k", split="train") \
    .to_parquet("data/raw/dolly.parquet")
load_dataset("paws", "labeled_final", split="train") \
    .select(range(20000)).to_parquet("data/raw/paws.parquet")
load_dataset("glue", "qqp", split="train") \
    .select(range(20000)).to_parquet("data/raw/qqp.parquet")
```

**✓ Verify** — `dolly[0]` has `instruction`, `context`, `response`, `category`; `set(dolly["category"])` gives brainstorming, classification, closed_qa, creative_writing, general_qa, information_extraction, open_qa, summarization. Those become `workload_label` for free.

**⚠** Check the Dolly licence page before committing to it and note it in your README. OASST1 is the fallback. Freeze to parquet — nothing in Days 1–5 should re-download anything.

### 0.5 Embedding smoke test, and your first measurement (15 min)

```python
from sentence_transformers import SentenceTransformer
import time, numpy as np

m = SentenceTransformer("all-MiniLM-L6-v2")
texts = ["summarize this passage in three bullet points"] * 100
m.encode(texts[:5], normalize_embeddings=True)          # warm up

t = time.perf_counter()
v = m.encode(texts, normalize_embeddings=True, batch_size=1)
print("per-encode ms:", (time.perf_counter() - t) / 100 * 1000)
print("dim:", v.shape[1], "norm:", np.linalg.norm(v[0]))
```

**✓ Verify** — dim 384, norm ~1.0, per-encode roughly 5–20ms on a laptop. **Write that number down.** It is the first row of your latency table.

**⚠** `normalize_embeddings=True` everywhere, always. Normalise in some places and not others and cosine distance stops being comparable across rows — you will not notice until the ROC is garbage.

### 0.6 Neon and the upstream key (20 min)

Create the Neon project, run `CREATE EXTENSION vector;` in its SQL editor, save the connection string. Get your upstream API key.

**⚠ Set a hard spend cap in the provider dashboard now.** You will run batch jobs over thousands of prompts; a loop bug at 3am is a real way to lose money.

`.env` — and add it to `.gitignore` in the same minute:

```
DATABASE_URL=postgresql://postgres:tollgate@localhost:5432/tollgate
NEON_URL=...
UPSTREAM_API_KEY=...
UPSTREAM_BASE_URL=...
EMBEDDING_MODEL=all-MiniLM-L6-v2
SIMILARITY_THRESHOLD=0.95
TOP_K=3
```

---

## Day 1 — Plumbing

Goal: a working proxy with **no cache**, and an eval script with **no data**. Nothing interesting happens today. Do it fast, don't polish.

### 1.1 Schema and migration runner (45 min)

Put §3 into `migrations/001_init.sql`. Runner:

```python
# migrations/run.py
import asyncio, asyncpg, os, pathlib
from dotenv import load_dotenv; load_dotenv()

async def main():
    conn = await asyncpg.connect(os.environ["DATABASE_URL"])
    await conn.execute("CREATE TABLE IF NOT EXISTS _migrations (name TEXT PRIMARY KEY)")
    done = {r["name"] for r in await conn.fetch("SELECT name FROM _migrations")}
    for f in sorted(pathlib.Path("migrations").glob("*.sql")):
        if f.name in done: continue
        print("applying", f.name)
        async with conn.transaction():
            await conn.execute(f.read_text())
            await conn.execute("INSERT INTO _migrations VALUES ($1)", f.name)
    await conn.close()

asyncio.run(main())
```

**✓ Verify** — run it twice; the second run prints nothing and errors nothing. That idempotence is the point of a migration runner and is worth being able to explain.

**⚠** Each migration runs inside a transaction, so a half-applied file cannot happen. Say that out loud in an interview — it is the difference between "I wrote migrations" and "I understand why migrations are transactional."

### 1.2 Database pool with vector support (30 min)

```python
# app/db.py
import asyncpg, os
from pgvector.asyncpg import register_vector

_pool = None

async def init_pool():
    global _pool
    _pool = await asyncpg.create_pool(
        os.environ["DATABASE_URL"], min_size=2, max_size=10,
        init=register_vector,      # ← per-connection; miss this and vectors come back as strings
    )
    return _pool

def pool():
    return _pool
```

**⚠** `init=register_vector` on the **pool**, not on a single connection. This is the most common pgvector-with-asyncpg mistake: without it `SELECT embedding` returns a string, numpy operations fail with something unhelpful, and you lose an hour.

**✓ Verify**

```python
async with pool().acquire() as c:
    r = await c.fetchval("SELECT $1::vector", np.zeros(384, dtype=np.float32))
    assert hasattr(r, "shape")     # numpy array, not str
```

### 1.3 Config, normalisation, embedder (45 min)

`app/config.py` — everything from env, no literals in code. `EMBEDDING_MODEL` and `WORKLOAD_LABEL` **must** be config rather than constants; that is the thirty-minute decision that keeps a multi-model sweep a config loop instead of a rewrite.

```python
# app/normalize.py
import hashlib, re

def raw_hash(messages) -> str:
    """Exact-match key. Over the RAW text — zero false-hit risk by construction."""
    blob = "\x00".join(f"{m['role']}:{m['content']}" for m in messages)
    return hashlib.sha256(blob.encode()).hexdigest()

_WS = re.compile(r"\s+")

def normalize(messages) -> str:
    """Embedding input ONLY. Lossy on purpose; never used as an exact key."""
    user = " ".join(m["content"] for m in messages if m["role"] == "user")
    return _WS.sub(" ", user.lower()).strip()
```

`app/embedder.py` — module-level singleton model, `encode(text) -> np.ndarray` with `normalize_embeddings=True`, returning elapsed ms alongside the vector so the caller can log it.

**⚠** Load the model at import, not inside the request handler — per-request loading costs ~2 seconds.

### 1.4 Upstream client with atomic budget (45 min)

```python
# app/upstream.py — the budget statement is the interesting part
CHARGE = """
UPDATE tenants SET spent_cents = spent_cents + $1
WHERE id = $2 AND spent_cents + $1 <= budget_cents
RETURNING spent_cents
"""
```

Returns `None` when over budget. Cost comes from the provider's token counts times a per-model rate held in config.

### 1.5 The endpoint, miss path only (90 min)

`app/main.py`: Pydantic models matching the OpenAI chat-completions request and response shapes, `POST /v1/chat/completions`, `GET /healthz`.

Today's flow: resolve tenant → charge budget → call upstream → store `cache_entries` row → write `cache_decisions` row → return.

**The entry and the decision row go in one transaction.** A decision log that disagrees with the cache is worthless as an eval substrate — this is the sentence that explains why the whole thing lives in Postgres rather than a vector DB.

**✓ Verify**

```bash
curl -s localhost:8000/v1/chat/completions \
  -H 'Authorization: Bearer test-key' -H 'Content-Type: application/json' \
  -d '{"model":"...","messages":[{"role":"user","content":"what is a b-tree"}]}' | jq .
```

```sql
SELECT decision, total_ms, upstream_ms, cost_cents FROM cache_decisions;
```

One row, `MISS_NO_CANDIDATE`, sensible timings.

### 1.6 Exact-match fast path (30 min)

Before embedding, look up `(tenant_id, prompt_hash, upstream_model, params_hash)` on the unique index. On hit return immediately, log `HIT_EXACT`, skip everything else.

**✓ Verify** — send the identical request twice; the second returns `HIT_EXACT` in single-digit ms with `upstream_ms IS NULL`.

**⚠** `params_hash` covers system prompt, temperature and max_tokens. Leave it out and you will serve a temperature-0 answer to a temperature-1 request — a genuine bug, and an embarrassing one to have found by an interviewer rather than by you.

### 1.7 Dockerfile and CI (45 min)

Multi-stage Dockerfile, only `requirements.txt` in the runtime layer. Pre-download the sentence-transformer at build time or the first request pays a 30-second model download.

`.github/workflows/ci.yml` — two jobs today: `lint-test` (ruff + pytest with a `postgres` service container) and `build` (docker build).

**✓ Verify** — push; both jobs green.

### 1.8 The eval script, against fake data (60 min)

**If the number in your resume bullet has no script computing it by end of Day 1, the project is at risk.**

```python
# eval/run_sweep.py
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

def scores(sim, compatible):
    """sim: float array. compatible: bool array."""
    return {
        "cosine_only": sim,
        "gated":       np.where(compatible, sim, -1.0),  # blocked pairs never admitted
    }

def bootstrap_auc(y, score, n=1000, seed=0):
    rng, idx, out = np.random.default_rng(seed), np.arange(len(y)), []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[s])) < 2: continue
        out.append(roc_auc_score(y[s], score[s]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))

if __name__ == "__main__":
    rng = np.random.default_rng(0)                    # synthetic today, real tomorrow
    y   = rng.random(500) > 0.5
    sim = np.clip(np.where(y, 0.93, 0.90) + rng.normal(0, .03, 500), 0, 1)
    comp = np.where(y, True, rng.random(500) > 0.4)
    # ... roc_curve for each score, plot both, save to eval/figures/roc.png
```

**The `np.where(compatible, sim, -1.0)` line is the trick worth understanding.** The gated rule is `sim ≥ τ AND compatible`, which is not a threshold on a continuous score, so you cannot feed it to `roc_curve` directly. Mapping incompatible pairs below any threshold makes the gated rule *equivalent* to a threshold on the transformed score — now both classifiers live on one axis and are directly comparable. Without this you will get stuck trying to plot a point instead of a curve.

**✓ Verify** — `eval/figures/roc.png` exists with two curves. The numbers are meaningless; the pipeline is real.

**End of Day 1:** proxy works, no cache. Eval script works, no data.

---

## Day 2 — The measurement becomes real

**Protect this day.** If it slips, the project has no result.

### 2.1 P1 and P2 from public data (45 min)

```python
# eval/build_dataset.py
# P1 natural paraphrase  ← QQP, label == 1, safe_to_reuse = True   (~400)
# P2 natural near-miss   ← PAWS, label == 0, safe_to_reuse = False (~400)
```

Filter both to pairs of 20–300 characters: very short pairs are trivially separable, very long ones are not prompt-shaped.

**⚠** P2 is the honesty control — organic confusions the gate was never designed to catch. Its numbers will be worse than P3's. Report it anyway.

### 2.2 P3 generation, eight families (90 min)

Take a Dolly instruction as `base`, inject two mutually incompatible constraints, and you have a high-cosine pair whose label is true by construction.

| Family | Variant A suffix | Variant B suffix | Rule implemented? |
|---|---|---|---|
| numeric | "Answer in exactly 3 bullet points." | "Answer in exactly 10 bullet points." | yes |
| negation | "Include examples from Python." | "Do not include anything related to Python." | yes |
| language | "Answer in French." | "Answer in German." | yes |
| entity | "Focus on Python." | "Focus on Go." | yes |
| format | "Answer as a markdown table." | "Answer as a single prose paragraph." | yes |
| **register** | "Explain it for a five-year-old." | "Explain it for a domain expert." | **held out** |
| **temporal** | "Answer as of 2019." | "Answer as of today." | **held out** |
| **polarity** | "Focus only on the benefits." | "Focus only on the drawbacks." | **held out** |

```python
def make_pair(base, family):
    a, b = FAMILIES[family]
    return f"{base.rstrip('.')}. {a}", f"{base.rstrip('.')}. {b}"
```

~150 base prompts × 8 families ≈ 1,200 negative pairs, `safe_to_reuse = False`, `family` recorded on every row.

**⚠ The held-out three are load-bearing.** Implement rules for the first five only.

### 2.3 P3 positive controls (30 min)

Same base, same constraint, trivially reworded:

```
("Answer in exactly 3 bullet points.", "Respond using exactly 3 bullet points.")
("Answer in French.",                  "Respond in French.")
("Answer as a markdown table.",        "Give the answer as a markdown table.")
```

~300 pairs, `safe_to_reuse = True`, same distribution as the P3 negatives. Without these, P3's AUC partly reflects Dolly-vs-QQP style rather than the constraint signal.

### 2.4 Validate 100 generated pairs by hand (40 min)

Dump 100 random P3 rows to a text file, read them, flag any with a wrong label (a numeric flip that produced an equivalent prompt; a suffix that made the prompt incoherent). Record the error rate.

**✓ Verify** — under ~5%. Above that, fix templates before continuing.

### 2.5 Similarity search and cosine-only admission (90 min)

```sql
SELECT id, response_text, constraints,
       1 - (embedding <=> $1) AS similarity
FROM cache_entries
WHERE tenant_id = $2 AND upstream_model = $3 AND params_hash = $4
ORDER BY embedding <=> $1
LIMIT $5
```

**⚠** `ORDER BY <=>` ascending — nearest first, because it is distance.

No gate yet. You need the un-gated baseline before you can show anything improved on it.

**✓ Verify** — "what is a b-tree", then "explain what a b-tree is". The second should be `HIT_SEMANTIC` at similarity ~0.9+.

### 2.6 The first real sweep (60 min)

Point `run_sweep.py` at `eval_pairs`. Embed both sides of every pair, compute cosine, sweep per population.

**✓ Verify** — ROC curves per population, AUC printed with 95% bootstrap CIs.

**⚠ Decision point.** If cosine-only already separates cleanly on P2 (AUC above ~0.9), your natural near-miss population is too easy. Tighten the PAWS filter toward higher lexical overlap **today**. Discovering this on Day 4 leaves no time to fix it.

### 2.7 Measure the exact/semantic split (20 min) — do not skip

```sql
SELECT decision, count(*), round(100.0*count(*)/sum(count(*)) OVER (), 1) AS pct
FROM cache_decisions GROUP BY decision;
```

Replay ~200 prompts with realistic repetition and look at `HIT_EXACT` vs `HIT_SEMANTIC`. If exact matching captures most of the achievable savings, the semantic layer is arguing over the remainder while carrying all of the correctness risk. Potentially the most interesting finding in the project — and you want to know before spending two more days defending the semantic path.

### 2.8 Extractor tests, written before the extractor (45 min)

`tests/test_constraints.py`, table-driven, one case per dimension, all failing. Tomorrow you make them pass.

**End of Day 2: you can state a number.** Everything after this improves it.

---

## Day 3 — The contribution

### 3.1 `constraints.py`, five dimensions (150 min)

```python
NUM_WORDS = {"one":1, "two":2, ..., "ten":10}
OUTPUT_NOUNS = r"(bullets?|points?|words?|sentences?|examples?|steps?|items?|paragraphs?|ways?|reasons?)"
COUNT_RE = re.compile(rf"\b(\d+|{'|'.join(NUM_WORDS)})\s+\w*\s*{OUTPUT_NOUNS}", re.I)
NEG_CUES = {"not","without","except","excluding","avoid","don't","do not","other than"}
FORMATS  = {"table":..., "json":..., "bullet":..., "prose":..., "code":...}
LANGS    = {"french","german","spanish","hindi","japanese", ...}

def extract(text: str) -> dict:
    return {"count": ..., "negation": ..., "language": ..., "entities": ..., "format": ...}

def compatible(a: dict, b: dict) -> tuple[bool, dict]:
    """Both None, or equal (set equality for entities). Returns (ok, diff)."""
```

**⚠** Return the **diff**, not just a boolean. `gate_passed = false` is undebuggable; `{"count": [3, 10]}` powers your best figure.

**✓ Verify** — every test from 2.8 passes; `extract()` runs under 1ms over 1,000 prompts.

### 3.2 Wire the gate in, with k fall-through (45 min)

Per §5. `k` from config, swept over {1, 3, 5} in the eval.

### 3.3 The gated sweep and the key figure (90 min)

2 embedding models × 4 populations × k ∈ {1,3,5} × cosine-only vs gated. The frontier API model runs offline in one batch — a few thousand embeddings, well under a dollar.

Figures per §8.

### 3.4 Output-constraint audit (60 min)

Per §7.

**✓ Verify** — seed an entry answering in 3 bullets, request 10 bullets with the gate disabled: `output_audit_ok = false`, diff names `count`.

### 3.5 Cost model (45 min)

Per §9.

### 3.6 Time the LLM-gate alternative and reject it (20 min)

Run 20 constraint checks through an LLM call; record latency and cost; compare against the cached call it protects. Write the rejection into the README with the numbers. *"I tried the obvious approach and it was economically incoherent"* is a strong beat for twenty minutes of work.

---

## Day 4 — The defence

### 4.1 Held-out family detection (45 min)

Detection on the five in-design families vs the three held-out vs P2, as three separate numbers. Expect held-out to be much lower. **Report it.** This table is the credibility of the whole result.

### 4.2 GPTCache benchmark (150 min)

`pip install gptcache`, wire its default configuration (default embedding, default `SearchDistanceEvaluation`, default threshold), push **the same eval pairs** through it, record its false-hit rate.

**⚠** Same pairs, same populations, same reporting. Change the data between systems and the comparison is void.

This is the answer to *"why not just use GPTCache?"* — a number instead of an opinion. Give it a real half-day.

### 4.3 Tenant isolation test (30 min)

Seed tenant A with an entry, request the near-identical prompt as tenant B, assert `MISS`. Cheap, and a genuine security property rather than a performance one.

### 4.4 Latency from your own log (30 min)

```sql
SELECT decision,
       percentile_cont(0.5)  WITHIN GROUP (ORDER BY total_ms) AS p50,
       percentile_cont(0.99) WITHIN GROUP (ORDER BY total_ms) AS p99,
       avg(embed_ms), avg(search_ms), avg(gate_ms)
FROM cache_decisions GROUP BY decision;
```

No load-testing tool. Report local-warm and deployed as separate rows with hardware labelled.

### 4.5 Assemble every figure (60 min)

All figures regenerate from one `make figures`. When you change a threshold on Day 5 you must not be hand-editing PNGs.

---

## Day 5 — Buffer, then ship

Whatever slipped comes first. Then:

1. **`/stats`** — JSON only: hit rate by type, output-audit failure rate, spend, p50/p99, decision breakdown. (60 min)
2. **Deploy** — Fly or Render + Neon. Run migrations against Neon. **Seed the cache** so the demo isn't cold. (90 min)
3. **CI eval-gate** — commit a frozen 200-pair subset with pre-computed embeddings as `.npz`, run the sweep, assert `AUC >= baseline - 0.02`. No model download, no API calls, ~30s, deterministic. (60 min)
4. **README as a report** — §13. **Three hours, and it is the deliverable.**
5. **Fill every bracket** in the resume bullets from `numbers.md`.

---

# Part III — Delivery

## 11. Repo layout

```
tollgate/
├── app/
│   ├── main.py            # FastAPI, /v1/chat/completions, /stats, /healthz
│   ├── db.py              # asyncpg pool, register_vector
│   ├── normalize.py
│   ├── constraints.py     # ← the core; imported by eval/ too
│   ├── embedder.py
│   ├── cache.py           # search + admission + output audit
│   ├── upstream.py
│   └── config.py          # τ, k, embedding_model, workload_label
├── eval/
│   ├── build_dataset.py
│   ├── run_sweep.py
│   ├── cost_model.py
│   ├── bench_gptcache.py
│   └── figures/
├── migrations/            # 001_init.sql, run.py
├── tests/
│   ├── test_constraints.py
│   ├── test_admission.py
│   ├── test_tenant_isolation.py
│   └── test_cache_key.py
├── data/raw/              # frozen parquet, gitignored
├── numbers.md             # updated daily from Day 1
├── docker-compose.yml · Dockerfile · .github/workflows/ci.yml · README.md
```

## 12. CI

1. **lint-test** — ruff + pytest, every push.
2. **build** — docker build.
3. **eval-gate** — `run_sweep.py` over a frozen 200-pair subset with a committed `.npz` of pre-computed embeddings, asserting `AUC >= baseline - 0.02`.

Job 3 is the one to talk about: a statistical property as a merge gate, made deterministic by freezing the subset and the embeddings.

## 13. README, as a report

1. The question — how often does a semantic cache serve a wrong answer, and how would you know?
2. Why it isn't observable — a false hit returns 200 with a fluent answer; the signal has to be manufactured.
3. Method — four populations, provenance, the held-out family design, the stated proxy-label limitation, the split.
4. Result 1 — cosine-only ROC, two models, per population, with CIs. The dominated region.
5. Result 2 — the gate shifts the curve; false hits by dimension; **in-design vs held-out vs natural**.
6. Result 3 — GPTCache defaults on the same pairs.
7. Choosing an operating point — the (ρ, C/c_miss) chart, and the workloads where the honest answer is *don't cache this*.
8. Latency and cost, measured, hardware labelled.
9. Limitations — §17.
10. Prior work — GPTCache, LiteLLM, SAFE-CACHE, category-aware caching. Say plainly what is theirs and what is yours.
11. *Then* setup instructions, at the bottom.

Section 10 is not optional. It is the clearest signal that you knew the landscape before you built.

## 14. Resume bullets

Fill every bracket from a script in the repo. No bracket survives to the printed version.

- Built an OpenAI-compatible LLM gateway (FastAPI, PostgreSQL/pgvector, Docker, CI) whose cache admission decision is evaluated as a binary classifier: **[X]%** hit rate at a measured **[Y]%** false-hit rate, cutting p50 latency from **[A]ms** to **[B]ms**.
- Showed cosine-only admission is dominated on a **[N]**-pair labelled set (QQP/PAWS + programmatically perturbed instructions); a **[<1]ms** rules-based constraint filter raised AUC from **[C]** to **[D]** (95% CI **[…]**) across two embedding models, with **[E]%** detection on three held-out perturbation families never used in rule design.
- Benchmarked GPTCache's default configuration on the same pairs at **[F]%** false hits, and added a zero-cost output-constraint audit giving a continuous production false-hit lower bound.

## 15. Numbers to capture

Keep `numbers.md` updated daily from Day 1.

Exact vs semantic hit split · hit rate · false-hit rate at operating point (+CI) · AUC per model per population, cosine-only and gated (+CIs) · in-design vs held-out vs P2 detection · generation error rate from the 100-pair validation · output-audit failure rate · GPTCache default false-hit rate · p50/p99 hit and miss paths · embed/search/gate ms · LLM-gate latency and cost (the rejected alternative) · cost per 1,000 requests cached vs uncached · corpus size at measurement · base rate per population · τ* surface over (ρ, C/c_miss).

## 16. Cut order

Cut strictly in this order:

1. `/stats` dashboard (keep the JSON)
2. CI eval-gate job
3. Frontier API embedding model — *costs you your strongest counter-argument; cut reluctantly*
4. Per-tenant budgets

— **do not cross this line** —

5. GPTCache benchmark · held-out family reporting · output-constraint audit · constraint gate + gated ROC · P2 natural near-misses · P3 positive controls · cosine baseline ROC · deployment

If you are cutting below the line, the project has failed and the right move is to stop and build something else. Say that to yourself on Day 3, not Day 5.

## 17. Limitations — name these before a reviewer finds them

1. **Real prompts are messier than Dolly.** The extractor's recall on organic traffic is unknown and probably lower. No cheap way to measure it in four days.
2. **`safe_to_reuse` may not be transitive.** A matches B and B matches C without A being safe for C. You measure pairs; the system operates on a growing set. No four-day answer.
3. **The exact-match path may be doing most of the work.** Measured in step 2.7. If exact matching captures most of the achievable savings, the semantic layer carries all of the correctness risk for a fraction of the benefit.
4. **Proxy labels.** A paraphrase label is a proxy for safe-to-reuse, not the same thing. QQP/PAWS are sentence pairs, not prompt pairs — which is exactly why P3 exists and why the populations are reported separately.

---

# Appendix A — The five things most likely to cost you an hour

| Symptom | Cause | Fix |
|---|---|---|
| Embeddings come back as `str` | `register_vector` not on the pool | `create_pool(..., init=register_vector)` |
| ROC looks inverted | `<=>` is distance, not similarity | `similarity = 1 - distance` |
| Similarities all ~0.99 | Not normalising embeddings | `normalize_embeddings=True` everywhere |
| Can't plot the gated rule as a curve | It isn't a threshold on a continuous score | `np.where(compatible, sim, -1.0)` |
| First request takes 30s | Model downloading at runtime | Bake it into the Docker image |

# Appendix B — Daily stop conditions

- **End of Day 1** — proxy answers, decisions log, a meaningless ROC renders. Otherwise you are behind; cut Day 5's `/stats` now.
- **End of Day 2** — you can state a real number, and you know your exact/semantic split. If not, **stop and reconsider the project.** Everything downstream assumes this exists.
- **End of Day 3** — the curve-shift figure exists. This is the contribution; nothing on Day 4 rescues its absence.
- **End of Day 4** — you can answer "why not GPTCache" with a number.

Better to lose two days than five.
