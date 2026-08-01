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

from app.cache import admit, audit_output, constraints_to_json, search_candidates  # noqa: E402
from app.config import settings  # noqa: E402
from app.constraints import extract  # noqa: E402
from app.db import init_pool, pool  # noqa: E402
from app.embedder import encode  # noqa: E402
from app.normalize import (  # noqa: E402
    constraint_input,
    hash_api_key,
    normalize,
    params_hash,
    raw_hash,
)
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
    # Liveness only — no DB error text, no version string. This is a public,
    # unauthenticated endpoint (Render/uptime checks hit it); the real
    # exception (which can contain host/connection details) goes to server
    # logs via print, never into the HTTP response.
    try:
        await pool().fetchval("SELECT 1")
    except Exception as e:
        print(f"healthz: db unreachable: {e}")
        raise HTTPException(status_code=503, detail="unhealthy")
    return {"status": "ok"}


async def resolve_tenant(authorization: str | None) -> uuid.UUID:
    """Shared by /v1/chat/completions and /stats — same Bearer-token
    resolution, same 401s, so the two endpoints can't drift apart on what
    counts as a valid key.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing or malformed Authorization header")
    api_key = authorization.removeprefix("Bearer ").strip()

    tenant = await pool().fetchrow(
        "SELECT id FROM tenants WHERE api_key_hash = $1", hash_api_key(api_key)
    )
    if tenant is None:
        raise HTTPException(status_code=401, detail="unknown api key")
    return tenant["id"]


@app.get("/stats")
async def stats(authorization: str | None = Header(default=None)):
    """Minimal for now — just the audit-failure rates, added because they
    must never be pooled. Full dashboard (hit rate by type, spend,
    p50/p99, decision breakdown) is Phase 10 (§13, §16 — first thing to
    cut under time pressure, but this pair isn't part of that).

    # DECISION-V2: same Bearer-token auth as /v1/chat/completions
    (resolve_tenant), scoped to WHERE tenant_id = $1 — not a separate
    admin secret. This endpoint exposes spend/hit-rate/activity figures;
    once deployed publicly those are per-tenant business data, not
    something any caller should see for every tenant. A tenant can see
    its own numbers (the same trust boundary chat_completions already
    uses), nothing more.
    """
    tenant_id = await resolve_tenant(authorization)
    rows = await pool().fetch(
        """SELECT decision,
                  count(*) FILTER (WHERE output_audit_ok IS NOT NULL) AS audited,
                  count(*) FILTER (WHERE output_audit_ok = false) AS failures
           FROM cache_decisions
           WHERE tenant_id = $1 AND decision IN ('HIT_EXACT', 'HIT_SEMANTIC')
           GROUP BY decision""",
        tenant_id,
    )
    by_decision = {r["decision"]: {"audited": r["audited"], "failures": r["failures"]} for r in rows}

    def rate(decision: str):
        d = by_decision.get(decision, {"audited": 0, "failures": 0})
        return {
            "rate": (d["failures"] / d["audited"]) if d["audited"] else None,
            "failures": d["failures"],
            "audited": d["audited"],
        }

    return {
        "llm_compliance_failure_rate": rate("HIT_EXACT"),
        "semantic_false_hit_rate": rate("HIT_SEMANTIC"),
    }


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
    constraints: dict | None = None,
    embedding=None,
    candidate_id: uuid.UUID | None = None,
    cosine_similarity: float | None = None,
    gate_passed: bool | None = None,
    constraint_diff=None,
    output_audit_ok: bool | None = None,
    output_audit_diff: dict | None = None,
    embed_ms: float | None = None,
    search_ms: float | None = None,
    gate_ms: float | None = None,
    upstream_ms: float | None = None,
    total_ms: float | None = None,
) -> None:
    """5f/8: every request writes exactly one of these — hit or miss, 429
    or upstream error. `constraints` is the extracted request-constraint
    dict (None before any embedding happens, e.g. the pre-flight 429).
    """
    await executor.execute(
        DECISION_INSERT_SQL,
        uuid.uuid4(),
        tenant_id,
        prompt_raw,
        constraints_to_json(constraints) if constraints is not None else {},
        embedding,
        candidate_id,
        cosine_similarity,
        settings.similarity_threshold,
        gate_passed,
        constraint_diff,
        decision,
        output_audit_ok,
        output_audit_diff,
        embed_ms,
        search_ms,
        gate_ms,
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
    tenant_id = await resolve_tenant(authorization)

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
        # §7 output-constraint audit — even on an exact-hash match, the
        # cached response was generated once, in the past, and the
        # upstream model's own instruction-following at that time might
        # not have been perfect (asked for 3 bullets, got 2). This is not
        # re-checking the gate (that compares two prompts); it's checking
        # the response against what was actually asked, every time it's
        # served, for free.
        request_text = constraint_input(messages)
        request_constraints = extract(request_text)
        audit_ok, audit_diff = audit_output(request_constraints, request_text, entry["response_text"])

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
                    constraints=request_constraints,
                    candidate_id=entry["id"],
                    output_audit_ok=audit_ok,
                    output_audit_diff=audit_diff or None,
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
    # 6a — embed + extract constraints. Needed whether this turns into a
    # semantic hit or a genuine miss, so do both before deciding which.
    # -----------------------------------------------------------------
    normalized = normalize(messages)
    vector, embed_ms = encode(normalized)
    request_text = constraint_input(messages)
    request_constraints = extract(request_text)

    # -----------------------------------------------------------------
    # 6b — semantic search (§4 step 7) + admission (§5), with fall-
    # through: a candidate that fails the gate doesn't end the search,
    # the next-nearest candidate still gets a chance. No budget charged
    # yet — search and gate cost nothing against the upstream budget,
    # only a real miss does.
    # -----------------------------------------------------------------
    search_t0 = time.perf_counter()
    candidates = await search_candidates(
        pool(), tenant_id, req.model, p_hash, vector, settings.top_k
    )
    search_ms = (time.perf_counter() - search_t0) * 1000

    gate_t0 = time.perf_counter()
    result = admit(candidates, request_constraints, settings.similarity_threshold)
    gate_ms = (time.perf_counter() - gate_t0) * 1000

    if result["decision"] == "HIT_SEMANTIC":
        candidate = result["candidate"]
        audit_ok, audit_diff = audit_output(request_constraints, request_text, candidate["response_text"])
        total_ms = (time.perf_counter() - t_start) * 1000
        async with pool().acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "UPDATE cache_entries SET hit_count = hit_count + 1 WHERE id = $1",
                    candidate["id"],
                )
                await log_decision(
                    conn,
                    tenant_id=tenant_id,
                    prompt_raw=prompt_raw,
                    decision="HIT_SEMANTIC",
                    cost_cents=0,
                    constraints=request_constraints,
                    embedding=vector,
                    candidate_id=candidate["id"],
                    cosine_similarity=result["similarity"],
                    gate_passed=True,
                    output_audit_ok=audit_ok,
                    output_audit_diff=audit_diff or None,
                    embed_ms=embed_ms,
                    search_ms=search_ms,
                    gate_ms=gate_ms,
                    total_ms=total_ms,
                )
        return ChatCompletionResponse(
            id=f"tollgate-{uuid.uuid4()}",
            created=int(time.time()),
            model=req.model,
            choices=[
                Choice(
                    index=0,
                    message=ChatMessage(role="assistant", content=candidate["response_text"]),
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=0,
                completion_tokens=candidate["response_tokens"],
                total_tokens=candidate["response_tokens"],
            ),
        )

    # -----------------------------------------------------------------
    # 6c — genuine miss (MISS_LOW_SIM / MISS_GATE / MISS_NO_CANDIDATE).
    # Same miss path as before, plus: log the specific miss reason
    # instead of one hardcoded decision, carry through the admission
    # attempt's similarity/diff for debugging, and store the REAL
    # extracted constraints on the new cache_entries row instead of {} —
    # future searches need them to gate against.
    # -----------------------------------------------------------------
    miss_decision = result["decision"]
    # gate_passed: True only means HIT. False means the gate ran and
    # rejected every candidate (MISS_GATE). None means the gate was never
    # reached at all — no candidates existed, or the top one was already
    # below the similarity threshold — not "ran and failed."
    gate_passed = False if miss_decision == "MISS_GATE" else None

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
            constraints=request_constraints,
            embedding=vector,
            cosine_similarity=result["similarity"],
            gate_passed=gate_passed,
            constraint_diff=result["diff"],
            embed_ms=embed_ms,
            search_ms=search_ms,
            gate_ms=gate_ms,
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
            constraints=request_constraints,
            embedding=vector,
            cosine_similarity=result["similarity"],
            gate_passed=gate_passed,
            constraint_diff=result["diff"],
            embed_ms=embed_ms,
            search_ms=search_ms,
            gate_ms=gate_ms,
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
                constraints_to_json(request_constraints), vector, settings.embedding_model,
                req.model, p_hash, response_text, usage["completion_tokens"], real_cost,
            )
            await log_decision(
                conn,
                tenant_id=tenant_id,
                prompt_raw=prompt_raw,
                decision=miss_decision,
                cost_cents=real_cost,
                constraints=request_constraints,
                embedding=vector,
                cosine_similarity=result["similarity"],
                gate_passed=gate_passed,
                constraint_diff=result["diff"],
                embed_ms=embed_ms,
                search_ms=search_ms,
                gate_ms=gate_ms,
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
