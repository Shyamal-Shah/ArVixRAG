from pathlib import Path
import json
import os
from loguru import logger
try:
    from .models import Chunk
except ImportError:
    from models import Chunk
import requests
import asyncio
from dotenv import load_dotenv

try:
    from .db import PGVectorDB
except ImportError:
    from db import PGVectorDB

load_dotenv()


def sanitize_text(text: str | None) -> str:
    """Remove null bytes and other control characters that break PostgreSQL UTF-8."""
    if not text:
        return ""
    return text.replace("\x00", "").replace("\r", "")


class TextEmbedder:

    def __init__(
        self,
        chunks_path: str | None = None,
    ) -> None:
        if chunks_path:
            self.chunks_path = Path(chunks_path)
        else:
            self.chunks_path = Path("data", "chunks.json")

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

    def open_router_embed(self, contents: list[str]) -> list[list[float]]:
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

    async def embed_chunks(self):
        chunks: list[Chunk] = []
        try:
            with open(self.chunks_path, "r") as fp:
                chunks = json.load(
                    fp, object_hook=lambda x: Chunk(**x) if isinstance(x, dict) else x
                )
        except FileNotFoundError:
            logger.error(f"Chunks file not found: {self.chunks_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse chunk JSON: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to parse chunk: {e}")
            raise

        if not chunks:
            logger.warning("No chunks loaded from JSON!")
            return

        contents = [chunk.content or "" for chunk in chunks]
        provider = os.getenv("EMBEDDING_PROVIDER", "OLLAMA").upper()
        logger.info(f"Embedding {len(chunks)} chunks using {provider}...")

        db = await PGVectorDB.create()

        batch_size = 100
        for batch_start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[batch_start : batch_start + batch_size]
            batch_contents = contents[batch_start : batch_start + batch_size]
            logger.debug(
                f"Embedding batch {batch_start // batch_size + 1} ({len(batch_chunks)} chunks)..."
            )

            if provider == "OPENAI":
                embeddings_list = self.openai_embed(batch_contents)
            elif provider == "OPENROUTER":
                embeddings_list = self.open_router_embed(batch_contents)
            else:
                embeddings_list = self.ollama_embed(batch_contents)

            for chunk, embedding in zip(batch_chunks, embeddings_list):
                chunk.embeddings = embedding

            async with db.get_db() as conn:
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
                            c.embeddings,
                        )
                        for c in batch_chunks
                    ],
                )
                logger.info(f"Inserted batch ending at chunk {batch_start + len(batch_chunks)}")

        async with db.get_db() as conn:
            count = await conn.fetchval("SELECT COUNT(*) FROM papers")
            logger.info(f"Total rows in db: {count}")

        await db.close()


if __name__ == "__main__":

    async def run():
        embedder = TextEmbedder()
        await embedder.embed_chunks()

    asyncio.run(run())
