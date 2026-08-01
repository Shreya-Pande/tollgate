import asyncpg
import json
import os
from pgvector.asyncpg import register_vector

_pool: asyncpg.Pool | None = None


async def _init_connection(conn: asyncpg.Connection) -> None:
    await register_vector(conn)
    # asyncpg doesn't serialize dict -> JSON automatically; without this,
    # every write to a JSONB column (constraints, constraint_diff,
    # output_audit_diff) needs a manual json.dumps + ::jsonb cast at every
    # call site. Same category of gotcha as register_vector, different type.
    await conn.set_type_codec(
        "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog", format="text"
    )


async def init_pool() -> asyncpg.Pool:
    global _pool
    _pool = await asyncpg.create_pool(
        os.environ["DATABASE_URL"],
        min_size=2,
        max_size=10,
        init=_init_connection,
    )
    return _pool


def pool() -> asyncpg.Pool:
    return _pool


# Note: pgvector==0.5.0 (current, unpinned) returns a pgvector.vector.Vector
# wrapper from SELECT ... embedding, not a bare numpy.ndarray as older
# pgvector-python (e.g. 0.3.6, what TOLLGATE.md was written against) did.
# Call .to_numpy() on it. register_vector on the pool is still what makes
# this work at all instead of returning a raw string.
