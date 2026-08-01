import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

import httpx  # noqa: E402
from fastapi import FastAPI, Header, HTTPException  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from app.config import settings  # noqa: E402
from app.db import init_pool, pool  # noqa: E402
from app.embedder import encode  # noqa: E402
from app.normalize import hash_api_key, normalize, params_hash, raw_hash  # noqa: E402
from app.upstream import (  # noqa: E402
    call_upstream,
    charge_budget,
    compute_cost_cents,
    reconcile_budget,
)

# ---------------------------------------------------------------------------
# 5a — Pydantic models: OpenAI-compatible request/response shapes.
# Malformed requests 422 automatically via FastAPI/Pydantic — no custom
# handling needed for that.
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float = 1.0
    max_tokens: int = 1024


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Choice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str | None = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage


# ---------------------------------------------------------------------------
# 5g — lifespan: pool opened on startup, closed on shutdown. Not a global
# opened at import, so tests can control the pool's lifetime explicitly.
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_pool()
    yield
    await pool().close()


app = FastAPI(lifespan=lifespan)


@app.get("/healthz")
async def healthz():
    try:
        await pool().fetchval("SELECT 1")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"db unreachable: {e}")
    return {"status": "ok"}


DECISION_INSERT_SQL = """
INSERT INTO cache_decisions (
    id, tenant_id, prompt_raw, constraints, embedding, candidate_id,
    cosine_similarity, threshold_used, gate_passed, constraint_diff,
    decision, output_audit_ok, output_audit_diff,
    embed_ms, search_ms, gate_ms, upstream_ms, total_ms,
    cost_cents, embedding_model, workload_label
) VALUES (
    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13,
    $14, $15, $16, $17, $18, $19, $20, $21
)
"""


async def log_decision(
    executor,
    *,
    tenant_id: uuid.UUID,
    prompt_raw: str,
    decision: str,
    cost_cents: float,
    embedding=None,
    candidate_id: uuid.UUID | None = None,
    cosine_similarity: float | None = None,
    embed_ms: float | None = None,
    upstream_ms: float | None = None,
    total_ms: float | None = None,
) -> None:
    """5f — every request writes exactly one of these, hit or miss, 429 or
    upstream error. search_ms/gate_ms are always NULL for now: nothing
    computes them yet (similarity search and the constraint gate are
    Phase 7/8). Same for gate_passed/constraint_diff/output_audit_*.
    """
    await executor.execute(
        DECISION_INSERT_SQL,
        uuid.uuid4(),
        tenant_id,
        prompt_raw,
        {},  # constraints — extractor not built yet (Phase 8)
        embedding,
        candidate_id,
        cosine_similarity,
        settings.similarity_threshold,
        None,  # gate_passed
        None,  # constraint_diff
        decision,
        None,  # output_audit_ok
        None,  # output_audit_diff
        embed_ms,
        None,  # search_ms
        None,  # gate_ms
        upstream_ms,
        total_ms,
        cost_cents,
        settings.embedding_model,
        settings.workload_label,
    )


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    req: ChatCompletionRequest, authorization: str | None = Header(default=None)
):
    t_start = time.perf_counter()

    # -----------------------------------------------------------------
    # 5c — auth + budget. A missing/malformed/unknown key is 401 and is
    # NOT logged to cache_decisions: tenant_id is NOT NULL there, and
    # there is no tenant to attribute the row to before this resolves.
    # -----------------------------------------------------------------
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing or malformed Authorization header")
    api_key = authorization.removeprefix("Bearer ").strip()

    tenant = await pool().fetchrow(
        "SELECT id FROM tenants WHERE api_key_hash = $1", hash_api_key(api_key)
    )
    if tenant is None:
        raise HTTPException(status_code=401, detail="unknown api key")
    tenant_id = tenant["id"]

    messages = [m.model_dump() for m in req.messages]
    prompt_raw = json.dumps(messages)

    # Zero-cost pre-flight gate: fails fast if the tenant is already at/over
    # budget, before doing any work at all (embed, upstream). Charging $0
    # still goes through the atomic UPDATE...WHERE, so it correctly rejects
    # an already-broke tenant without touching spent_cents.
    if await charge_budget(pool(), tenant_id, 0) is None:
        total_ms = (time.perf_counter() - t_start) * 1000
        await log_decision(
            pool(),
            tenant_id=tenant_id,
            prompt_raw=prompt_raw,
            decision="MISS_BUDGET_EXCEEDED",
            cost_cents=0,
            total_ms=total_ms,
        )
        raise HTTPException(status_code=429, detail="tenant budget exceeded")

    # -----------------------------------------------------------------
    # 5d — exact-match path. Before any embedding: look up the unique
    # index directly. On hit, return immediately — no embedding, no
    # upstream call, no similarity search of any kind.
    # -----------------------------------------------------------------
    p_hash = params_hash(messages, req.temperature, req.max_tokens, req.model)
    r_hash = raw_hash(messages)

    entry = await pool().fetchrow(
        """SELECT id, response_text, response_tokens FROM cache_entries
           WHERE tenant_id = $1 AND prompt_hash = $2
             AND upstream_model = $3 AND params_hash = $4""",
        tenant_id, r_hash, req.model, p_hash,
    )

    if entry is not None:
        total_ms = (time.perf_counter() - t_start) * 1000
        async with pool().acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "UPDATE cache_entries SET hit_count = hit_count + 1 WHERE id = $1",
                    entry["id"],
                )
                await log_decision(
                    conn,
                    tenant_id=tenant_id,
                    prompt_raw=prompt_raw,
                    decision="HIT_EXACT",
                    cost_cents=0,
                    candidate_id=entry["id"],
                    total_ms=total_ms,
                )
        return ChatCompletionResponse(
            id=f"tollgate-{uuid.uuid4()}",
            created=int(time.time()),
            model=req.model,
            choices=[
                Choice(
                    index=0,
                    message=ChatMessage(role="assistant", content=entry["response_text"]),
                    finish_reason="stop",
                )
            ],
            # prompt_tokens isn't persisted on cache_entries (only the
            # response's token count is) — out of scope to add for Day 1;
            # a hit's total_tokens undercounts by that amount.
            usage=Usage(
                prompt_tokens=0,
                completion_tokens=entry["response_tokens"],
                total_tokens=entry["response_tokens"],
            ),
        )

    # -----------------------------------------------------------------
    # 5e — miss path. Embed, call upstream, store both rows in one
    # transaction. Embedding is computed and stored even though nothing
    # reads it yet — Phase 7 needs it and this avoids a backfill.
    # -----------------------------------------------------------------
    normalized = normalize(messages)
    vector, embed_ms = encode(normalized)

    full_text = "".join(m["content"] for m in messages)
    estimated_prompt_tokens = len(full_text) // 4
    estimated_cost = compute_cost_cents(req.model, estimated_prompt_tokens, req.max_tokens)

    if await charge_budget(pool(), tenant_id, estimated_cost) is None:
        total_ms = (time.perf_counter() - t_start) * 1000
        await log_decision(
            pool(),
            tenant_id=tenant_id,
            prompt_raw=prompt_raw,
            decision="MISS_BUDGET_EXCEEDED",
            cost_cents=0,
            embedding=vector,
            embed_ms=embed_ms,
            total_ms=total_ms,
        )
        raise HTTPException(status_code=429, detail="tenant budget exceeded")

    upstream_t0 = time.perf_counter()
    try:
        upstream_response = await call_upstream(
            req.model, messages, req.temperature, req.max_tokens
        )
    except (httpx.HTTPError, httpx.HTTPStatusError) as e:
        upstream_ms = (time.perf_counter() - upstream_t0) * 1000
        total_ms = (time.perf_counter() - t_start) * 1000
        # Full refund — the estimate was charged but no real cost was
        # incurred (the upstream call itself failed).
        await charge_budget(pool(), tenant_id, -estimated_cost)
        await log_decision(
            pool(),
            tenant_id=tenant_id,
            prompt_raw=prompt_raw,
            decision="MISS_UPSTREAM_ERROR",
            cost_cents=0,
            embedding=vector,
            embed_ms=embed_ms,
            upstream_ms=upstream_ms,
            total_ms=total_ms,
        )
        raise HTTPException(status_code=502, detail=f"upstream error: {e}")
    upstream_ms = (time.perf_counter() - upstream_t0) * 1000

    response_text = upstream_response["choices"][0]["message"]["content"]
    usage = upstream_response["usage"]
    real_cost = compute_cost_cents(req.model, usage["prompt_tokens"], usage["completion_tokens"])
    reconcile_delta = real_cost - estimated_cost

    entry_id = uuid.uuid4()
    total_ms = (time.perf_counter() - t_start) * 1000

    async with pool().acquire() as conn:
        async with conn.transaction():
            # Reconcile the estimate to the real cost in the same
            # transaction as the cache writes. Unconditional — the upstream
            # call already happened by this point, so this write must not
            # be able to fail and take the cache_entries/cache_decisions
            # rows down with it. See reconcile_budget's docstring.
            await reconcile_budget(conn, tenant_id, reconcile_delta)
            await conn.execute(
                """INSERT INTO cache_entries (
                    id, tenant_id, prompt_raw, prompt_hash, prompt_normalized,
                    constraints, embedding, embedding_model, upstream_model,
                    params_hash, response_text, response_tokens, cost_cents
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)""",
                entry_id, tenant_id, prompt_raw, r_hash, normalized,
                {}, vector, settings.embedding_model, req.model,
                p_hash, response_text, usage["completion_tokens"], real_cost,
            )
            await log_decision(
                conn,
                tenant_id=tenant_id,
                prompt_raw=prompt_raw,
                decision="MISS_NO_CANDIDATE",
                cost_cents=real_cost,
                embedding=vector,
                embed_ms=embed_ms,
                upstream_ms=upstream_ms,
                total_ms=total_ms,
            )

    return ChatCompletionResponse(
        id=f"tollgate-{uuid.uuid4()}",
        created=int(time.time()),
        model=req.model,
        choices=[
            Choice(
                index=0,
                message=ChatMessage(role="assistant", content=response_text),
                finish_reason="stop",
            )
        ],
        usage=Usage(
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            total_tokens=usage["total_tokens"],
        ),
    )
