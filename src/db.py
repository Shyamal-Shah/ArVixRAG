import os
import asyncio
import asyncpg
from dotenv import load_dotenv
from pgvector.asyncpg import register_vector

load_dotenv()


class PGVectorDB:
    @classmethod
    async def create(cls, reset: bool = False):
        self = cls()
        await self.initialize(reset=reset)
        return self

    def __init__(self):
        self.pool: asyncpg.Pool | None = None

    async def initialize(self, reset: bool = False):
        db_name = os.environ.get("POSTGRES_DB")
        host = os.environ.get("POSTGRES_HOST")
        password = os.environ.get("POSTGRES_PASSWORD")
        port = os.environ.get("POSTGRES_PORT")
        user = os.environ.get("POSTGRES_USER")

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
            await conn.execute(f"DROP DATABASE IF EXISTS {db_name}")

        try:
            await conn.execute(f"CREATE DATABASE {db_name}")
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
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS papers (
                    id SERIAL PRIMARY KEY,
                    arxiv_id VARCHAR(120),
                    title TEXT,
                    chunk_index INTEGER,
                    content TEXT,
                    embedding vector(1024),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS papers_embedding_idx
                ON papers
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            """)

    async def close(self):
        if self.pool:
            await self.pool.close()

    def get_db(self) -> asyncpg.pool.PoolAcquireContext:
        assert self.pool is not None, "pool not initialized — call create() first"
        return self.pool.acquire()


if __name__ == "__main__":

    async def run():
        vectorDb = await PGVectorDB.create(reset=True)
        await vectorDb.close()

    asyncio.run(run())
