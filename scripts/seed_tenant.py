import asyncio
import secrets
import sys
import uuid
from pathlib import Path

# Running as `python scripts\seed_tenant.py` puts scripts/ on sys.path[0],
# not the repo root, so `import app` fails without this.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from app.db import init_pool, pool  # noqa: E402
from app.normalize import hash_api_key  # noqa: E402


async def main():
    await init_pool()
    tenant_id = uuid.uuid4()
    api_key = secrets.token_urlsafe(32)
    await pool().execute(
        "INSERT INTO tenants (id, name, budget_cents, spent_cents, api_key_hash) VALUES ($1, $2, $3, 0, $4)",
        tenant_id, "dev-test-tenant", 10000, hash_api_key(api_key),
    )
    print("tenant_id:", tenant_id)
    print("api_key:", api_key, "  <- shown once, not stored anywhere in plaintext")
    print("budget_cents: 10000")
    await pool().close()


if __name__ == "__main__":
    asyncio.run(main())
