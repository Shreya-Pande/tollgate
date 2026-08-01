import asyncio
import os
import uuid

import asyncpg
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.normalize import hash_api_key


@pytest.fixture
def tenant():
    # Plain sync fixture, asyncio.run per DB call — TestClient manages its
    # own event loop for the app's lifespan (app.db's pool); mixing that
    # with a pytest-asyncio async fixture in the same test risks a nested
    # event loop, so setup/teardown here use a fully separate connection
    # and never touch app.db's pool at all.
    tenant_id = uuid.uuid4()
    api_key = f"stats-auth-test-{uuid.uuid4()}"

    async def _create():
        conn = await asyncpg.connect(os.environ["DATABASE_URL"])
        try:
            await conn.execute(
                "INSERT INTO tenants (id, name, budget_cents, spent_cents, api_key_hash) "
                "VALUES ($1, 'stats-auth-test', 10000, 0, $2)",
                tenant_id, hash_api_key(api_key),
            )
        finally:
            await conn.close()

    async def _delete():
        conn = await asyncpg.connect(os.environ["DATABASE_URL"])
        try:
            await conn.execute("DELETE FROM tenants WHERE id = $1", tenant_id)
        finally:
            await conn.close()

    asyncio.run(_create())
    yield tenant_id, api_key
    asyncio.run(_delete())


def test_stats_requires_auth(tenant):
    """/stats was reachable with no auth at all before this fix — exposed
    spend/hit-rate/tenant activity to anyone with the URL. §3/§4's tenant
    isolation applies here too: this is the same trust boundary
    /v1/chat/completions already enforces, not a new one.
    """
    with TestClient(app) as client:
        assert client.get("/stats").status_code == 401
        assert client.get("/stats", headers={"Authorization": "Bearer not-a-real-key"}).status_code == 401

        _, api_key = tenant
        ok = client.get("/stats", headers={"Authorization": f"Bearer {api_key}"})
        assert ok.status_code == 200
        assert "llm_compliance_failure_rate" in ok.json()


def test_healthz_does_not_leak_internals():
    with TestClient(app) as client:
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
