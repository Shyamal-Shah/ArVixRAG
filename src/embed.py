from pathlib import Path
import os
import asyncio
from loguru import logger
import requests
from dotenv import load_dotenv

try:
    from .models import Chunk
    from .db import PGVectorDB
    from .persistence import load_chunks
except ImportError:
    from models import Chunk
    from db import PGVectorDB
    from persistence import load_chunks

load_dotenv()

DEFAULT_BATCH_SIZE = 100


def sanitize_text(text: str | None) -> str:
    """Remove null bytes and other control characters that break PostgreSQL UTF-8."""
    if not text:
        return ""
    return text.replace("\x00", "").replace("\r", "")


class TextEmbedder:

    def __init__(self, chunks_path: str | None = None) -> None:
        self.chunks_path = (
            Path(chunks_path) if chunks_path else Path("data", "chunks.json")
        )

    def ollama_embed(self, contents: list[str]) -> list[list[float]]:
        model = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:0.6b")
        response = requests.post(
            "http://localhost:11434/api/embed",
            json={"model": model, "input": contents},
        )
        response.raise_for_status()
        return response.json()["embeddings"]

    def openai_embed(self, contents: list[str]) -> list[list[float]]:
        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        response = requests.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": model, "input": contents},
        )
        response.raise_for_status()
        data = response.json()["data"]
        return [item["embedding"] for item in data]

    def openrouter_embed(self, contents: list[str]) -> list[list[float]]:
        api_key = os.getenv("OPENROUTER_API_KEY")
        model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        response = requests.post(
            "https://openrouter.ai/api/v1/embeddings",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": model, "input": contents},
        )
        response.raise_for_status()
        data = response.json()["data"]
        return [item["embedding"] for item in data]

    def embed(self, contents: list[str]) -> list[list[float]]:
        """Embed a batch of texts using the configured provider."""
        provider = os.getenv("EMBEDDING_PROVIDER", "OLLAMA").upper()
        if provider == "OPENAI":
            return self.openai_embed(contents)
        elif provider == "OPENROUTER":
            return self.openrouter_embed(contents)
        return self.ollama_embed(contents)

    async def embed_chunks(self):
        chunks: list[Chunk] = load_chunks(self.chunks_path)

        if not chunks:
            logger.warning("No chunks loaded from JSON!")
            return

        contents = [chunk.content or "" for chunk in chunks]
        provider = os.getenv("EMBEDDING_PROVIDER", "OLLAMA").upper()
        logger.info(f"Embedding {len(chunks)} chunks using {provider}...")

        db = await PGVectorDB.create()

        for batch_start in range(0, len(chunks), DEFAULT_BATCH_SIZE):
            batch_chunks = chunks[batch_start : batch_start + DEFAULT_BATCH_SIZE]
            batch_contents = contents[batch_start : batch_start + DEFAULT_BATCH_SIZE]
            logger.debug(
                f"Embedding batch {batch_start // DEFAULT_BATCH_SIZE + 1} ({len(batch_chunks)} chunks)..."
            )

            embeddings_list = await asyncio.to_thread(self.embed, batch_contents)
            if len(embeddings_list) != len(batch_contents):
                raise ValueError(
                    f"Provider returned {len(embeddings_list)} embeddings for "
                    f"{len(batch_contents)} inputs"
                )

            for chunk, embedding in zip(batch_chunks, embeddings_list):
                chunk.embedding = embedding

            async with db.acquire() as conn:
                await conn.executemany(
                    """
                    INSERT INTO papers(arxiv_id, title, chunk_index, content, embedding)
                    VALUES($1, $2, $3, $4, $5)
                    ON CONFLICT (arxiv_id, chunk_index) DO NOTHING
                    """,
                    [
                        (
                            sanitize_text(c.arxiv_id),
                            sanitize_text(c.paper_title),
                            c.chunk_index,
                            sanitize_text(c.content),
                            c.embedding,
                        )
                        for c in batch_chunks
                    ],
                )
                logger.info(
                    f"Inserted batch ending at chunk {batch_start + len(batch_chunks)}"
                )

        async with db.acquire() as conn:
            count = await conn.fetchval("SELECT COUNT(*) FROM papers")
            logger.info(f"Total rows in db: {count}")

        await db.close()


if __name__ == "__main__":

    async def run():
        embedder = TextEmbedder()
        await embedder.embed_chunks()

    asyncio.run(run())
