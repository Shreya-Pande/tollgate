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
