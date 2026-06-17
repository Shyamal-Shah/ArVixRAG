import os
import re
import asyncio
import asyncpg
from dotenv import load_dotenv
from pgvector.asyncpg import register_vector

load_dotenv()

DEFAULT_EMBEDDING_DIMENSION = 1024
IVFFLAT_LISTS = 100


def _validate_identifier(name: str | None) -> str:
    """Validate a SQL identifier (e.g. database name) against a strict allowlist.

    DDL statements cannot use bind parameters, so the value is interpolated
    directly; restricting it to a safe charset prevents injection/breakage.
    """
    if not name or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(f"Invalid/unsafe SQL identifier: {name!r}")
    return name


class PGVectorDB:
    @classmethod
    async def create(cls, reset: bool = False):
        self = cls()
        await self.initialize(reset=reset)
        return self

    def __init__(self):
        self.pool: asyncpg.Pool | None = None

    async def initialize(self, reset: bool = False):
        db_name = _validate_identifier(os.getenv("POSTGRES_DB"))
        host = os.getenv("POSTGRES_HOST")
        password = os.getenv("POSTGRES_PASSWORD")
        port = os.getenv("POSTGRES_PORT")
        user = os.getenv("POSTGRES_USER")
        dim = int(os.getenv("EMBEDDING_DIMENSION", str(DEFAULT_EMBEDDING_DIMENSION)))

        conn = await asyncpg.connect(
            user=user, password=password, database="postgres", host=host, port=port
        )
        if reset:
            await conn.execute(f"""
                SELECT pg_terminate_backend(pg_stat_activity.pid)
                FROM pg_stat_activity
                WHERE pg_stat_activity.datname = '{db_name}'
                AND pid <> pg_backend_pid()
            """)
            await conn.execute(f'DROP DATABASE IF EXISTS "{db_name}"')

        try:
            await conn.execute(f'CREATE DATABASE "{db_name}"')
        except asyncpg.exceptions.DuplicateDatabaseError:
            pass

        await conn.close()

        conn = await asyncpg.connect(
            user=user, password=password, database=db_name, host=host, port=port
        )
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await conn.close()

        async def init(conn):
            await register_vector(conn)

        self.pool = await asyncpg.create_pool(
            user=user,
            password=password,
            database=db_name,
            host=host,
            port=port,
            init=init,
            min_size=1,
            max_size=5,
        )

        async with self.pool.acquire() as conn:
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS papers (
                    id SERIAL PRIMARY KEY,
                    arxiv_id VARCHAR(120),
                    title TEXT,
                    chunk_index INTEGER,
                    content TEXT,
                    embedding vector({dim}),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (arxiv_id, chunk_index)
                )
            """)

            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS papers_embedding_idx
                ON papers
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = {IVFFLAT_LISTS})
            """)

    async def close(self):
        if self.pool:
            await self.pool.close()

    def acquire(self) -> asyncpg.pool.PoolAcquireContext:
        if self.pool is None:
            raise RuntimeError("pool not initialized — call create() first")
        return self.pool.acquire()


if __name__ == "__main__":

    async def run():
        vector_db = await PGVectorDB.create(reset=True)
        await vector_db.close()

    asyncio.run(run())
