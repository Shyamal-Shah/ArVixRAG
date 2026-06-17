import dotenv, os
from loguru import logger
import asyncio

dotenv.load_dotenv()

from embed import TextEmbedder
from db import PGVectorDB


class Retriever:
    def __init__(self) -> None:
        self.embedder = TextEmbedder()

    async def fetch_k_chunks(self, query: str, k: int = 5):
        provider = os.getenv("EMBEDDING_PROVIDER", "OLLAMA").upper()
        logger.info(f"Embedding query of len{len(query)} using {provider}...")

        if provider == "OPENAI":
            embeddings = self.embedder.openai_embed([query])[0]
        elif provider == "OPENROUTER":
            embeddings = self.embedder.open_router_embed([query])[0]
        else:
            embeddings = self.embedder.ollama_embed([query])[0]

        logger.info(f"Query Embedded with dimensions: {len(embeddings)}")

        top_chunks = []
        db = await PGVectorDB.create()
        async with db.get_db() as conn:
            top_chunks = await conn.fetch(
                f"SELECT * FROM papers ORDER BY embedding <-> $1 LIMIT {k}", embeddings
            )
        logger.info(f"Returned {len(top_chunks)} similar chunks to query")

        return top_chunks


if __name__ == "__main__":

    async def run():
        retriever = Retriever()
        chunks = await retriever.fetch_k_chunks(
            "How does attention mechanism work?", 10
        )
        for chunk in chunks:
            logger.success(
                f'{chunk["id"]}. Title: {chunk["title"]}, chunk_index:{chunk["chunk_index"]}, Content:{chunk["content"]}'
            )

    asyncio.run(run())
