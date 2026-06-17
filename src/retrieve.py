import json
import os
import asyncio
from typing import TypedDict

import asyncpg
import requests
from dotenv import load_dotenv
from loguru import logger

try:
    from .embed import TextEmbedder
    from .db import PGVectorDB
except ImportError:
    from embed import TextEmbedder
    from db import PGVectorDB

load_dotenv()


class ChunkResult(TypedDict):
    id: int
    arxiv_id: str
    title: str
    chunk_index: int
    content: str


class RagResult(TypedDict):
    question: str
    answer: str
    chunks: list[ChunkResult]
    retrieval_time: str
    latency: str


class Retriever:
    def __init__(self) -> None:
        self.embedder = TextEmbedder()
        self.db: PGVectorDB | None = None

    async def _get_db(self) -> PGVectorDB:
        """Create the connection pool once and reuse it across queries."""
        if self.db is None:
            self.db = await PGVectorDB.create()
        return self.db

    async def close(self) -> None:
        if self.db is not None:
            await self.db.close()
            self.db = None

    async def fetch_k_chunks(self, query: str, k: int = 5) -> list[asyncpg.Record]:
        provider = os.getenv("EMBEDDING_PROVIDER", "OLLAMA").upper()
        logger.info(f"Embedding query of len {len(query)} using {provider}...")

        query_embedding = (await asyncio.to_thread(self.embedder.embed, [query]))[0]
        logger.info(f"Query embedded with dimensions: {len(query_embedding)}")

        db = await self._get_db()
        async with db.acquire() as conn:
            top_chunks = await conn.fetch(
                "SELECT * FROM papers ORDER BY embedding <=> $1 LIMIT $2",
                query_embedding,
                k,
            )
        logger.info(f"Returned {len(top_chunks)} similar chunks to query")
        return top_chunks

    async def ollama_complete(self, system_prompt: str, user_message: str) -> str:
        model = os.getenv("COMPLETION_MODEL", "qwen3:8b")
        response = await asyncio.to_thread(
            requests.post,
            "http://localhost:11434/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "stream": False,
            },
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    async def openrouter_complete(self, system_prompt: str, user_message: str) -> str:
        api_key = os.getenv("OPENROUTER_API_KEY")
        model = os.getenv("COMPLETION_MODEL", "openai/gpt-4o-mini")
        response = await asyncio.to_thread(
            requests.post,
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    async def openai_complete(self, system_prompt: str, user_message: str) -> str:
        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("COMPLETION_MODEL", "gpt-4o-mini")
        response = await asyncio.to_thread(
            requests.post,
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    async def complete(self, system_prompt: str, user_message: str) -> str:
        provider = os.getenv(
            "COMPLETION_PROVIDER", os.getenv("EMBEDDING_PROVIDER", "OLLAMA")
        ).upper()
        logger.info(f"Generating completion using {provider}...")
        if provider == "OPENAI":
            return await self.openai_complete(system_prompt, user_message)
        elif provider == "OPENROUTER":
            return await self.openrouter_complete(system_prompt, user_message)
        return await self.ollama_complete(system_prompt, user_message)

    async def query_rag(self, question: str) -> RagResult:
        start = asyncio.get_event_loop().time()
        chunks = await self.fetch_k_chunks(question, 5)
        chunk_end = asyncio.get_event_loop().time()

        context = "".join(
            f"ArXiv Id: {chunk['arxiv_id']}\nPaper Title: {chunk['title']}\n"
            f"Chunk Index: {chunk['chunk_index']}\nContent: {chunk['content']}\n\n"
            for chunk in chunks
        )
        system_prompt = (
            "You are an intelligent research assistant. You are given the following "
            "context from research papers:\n"
            f"{context}\n"
            "Based on the context provided, answer the following question concisely "
            "and provide references to the papers you used. If you cannot find an "
            'answer, say "I don\'t know, context not sufficient, please provide more '
            'details in your query".\n'
            "Always return the answer in markdown format."
        )

        answer = await self.complete(system_prompt, question)
        end = asyncio.get_event_loop().time()

        return {
            "question": question,
            "answer": answer,
            "chunks": [
                {
                    "id": c["id"],
                    "arxiv_id": c["arxiv_id"],
                    "title": c["title"],
                    "chunk_index": c["chunk_index"],
                    "content": c["content"],
                }
                for c in chunks
            ],
            "retrieval_time": "{:.2f} seconds".format(chunk_end - start),
            "latency": "{:.2f} seconds".format(end - start),
        }


if __name__ == "__main__":

    async def run():
        retriever = Retriever()
        try:
            with open("data/questions.json", "r") as f:
                questions = json.load(f)
            results = []
            for q in questions:
                result = await retriever.query_rag(q)
                logger.success(f"Q: {result['question']}")
                logger.success(f"A: {result['answer']}")
                logger.info(
                    f"Retrieval: {result['retrieval_time']} | Total: {result['latency']}"
                )
                results.append(result)

            with open("data/naive_rag_outputs.json", "w") as f:
                json.dump(results, f, indent=2)
            print("---")
        finally:
            await retriever.close()

    asyncio.run(run())
