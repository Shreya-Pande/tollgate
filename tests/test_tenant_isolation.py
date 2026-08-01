import uuid

import pytest
import pytest_asyncio

from app.cache import search_candidates
from app.db import init_pool, pool


@pytest_asyncio.fixture
async def db():
    await init_pool()
    yield
    await pool().close()


@pytest.mark.asyncio
async def test_cross_tenant_search_returns_nothing(db):
    """§3/§4: tenant_id is part of the cache key specifically because
    cross-tenant reuse is a data leak, not a performance concern. This is
    the live-DB security property; test_cache_key.py's
    test_different_tenants_same_prompt_have_different_cache_keys is the
    unit-level version of the same claim, over the compound key tuple
    rather than a real search_candidates() call against Neon.
    """
    tenant_a, tenant_b, entry_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    vector = [0.1] * 384  # arbitrary but fixed — same vector used to insert and to search

    async with pool().acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                "INSERT INTO tenants (id, name, budget_cents, spent_cents, api_key_hash) "
                "VALUES ($1, 'iso-test-a', 10000, 0, $2)",
                tenant_a, f"isotest-{uuid.uuid4()}",
            )
            await conn.execute(
                "INSERT INTO tenants (id, name, budget_cents, spent_cents, api_key_hash) "
                "VALUES ($1, 'iso-test-b', 10000, 0, $2)",
                tenant_b, f"isotest-{uuid.uuid4()}",
            )
            await conn.execute(
                """INSERT INTO cache_entries (
                    id, tenant_id, prompt_raw, prompt_hash, prompt_normalized,
                    constraints, embedding, embedding_model, upstream_model,
                    params_hash, response_text, response_tokens, cost_cents
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)""",
                entry_id, tenant_a, "what is a hash map", "hashA", "what is a hash map",
                {}, vector, "test-model", "test-model", "paramsA", "a hash map is...", 10, 0,
            )

    try:
        # Tenant A, searching with the near-identical (here: identical)
        # embedding it was seeded with, finds its own entry.
        candidates_a = await search_candidates(pool(), tenant_a, "test-model", "paramsA", vector, k=3)
        assert len(candidates_a) == 1

        # Tenant B — same model, same params, same embedding vector —
        # must get nothing. The WHERE tenant_id = $2 scoping in
        # search_candidates is what makes this a security property
        # rather than a performance one; if this ever returns a row,
        # tenant B just read tenant A's cached response.
        candidates_b = await search_candidates(pool(), tenant_b, "test-model", "paramsA", vector, k=3)
        assert len(candidates_b) == 0
    finally:
        async with pool().acquire() as conn:
            async with conn.transaction():
                await conn.execute("DELETE FROM cache_entries WHERE id = $1", entry_id)
                await conn.execute(
                    "DELETE FROM tenants WHERE id = ANY($1::uuid[])", [tenant_a, tenant_b]
                )
