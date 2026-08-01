import asyncio
import os
import pathlib

import asyncpg
from dotenv import load_dotenv

load_dotenv()


async def main():
    conn = await asyncpg.connect(os.environ["DATABASE_URL"])
    await conn.execute("CREATE TABLE IF NOT EXISTS _migrations (name TEXT PRIMARY KEY)")
    done = {r["name"] for r in await conn.fetch("SELECT name FROM _migrations")}
    for f in sorted(pathlib.Path(__file__).parent.glob("*.sql")):
        if f.name in done:
            continue
        print("applying", f.name)
        async with conn.transaction():
            await conn.execute(f.read_text())
            await conn.execute("INSERT INTO _migrations VALUES ($1)", f.name)
    await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
