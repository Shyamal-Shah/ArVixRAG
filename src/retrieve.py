import json

import dotenv, os
from loguru import logger
import asyncio
import requests

dotenv.load_dotenv()

try:
    from .embed import TextEmbedder
    from .db import PGVectorDB
except ImportError:
    from embed import TextEmbedder
    from db import PGVectorDB


class Retriever:
    def __init__(self) -> None:
        self.embedder = TextEmbedder()

    async def fetch_k_chunks(self, query: str, k: int = 5):
        provider = os.getenv("EMBEDDING_PROVIDER", "OLLAMA").upper()
        logger.info(f"Embedding query of len {len(query)} using {provider}...")

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

    async def ollama_completions(self, query: str, prompt: str) -> str:
        model = os.getenv("COMPLETION_MODEL", "qwen3:8b")
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": query},
                ],
                "stream": False,
            },
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    async def open_router_completions(self, query: str, prompt: str) -> str:
        api_key = os.getenv("OPENROUTER_API_KEY")
        model = os.getenv("COMPLETION_MODEL", "openai/gpt-4o-mini")
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": query},
                ],
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    async def openai_completions(self, query: str, prompt: str) -> str:
        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("COMPLETION_MODEL", "gpt-4o-mini")
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": query},
                ],
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    async def query_rag(self, question: str) -> dict[str, str | list[dict]]:
        start = asyncio.get_event_loop().time()
        chunks = await self.fetch_k_chunks(question, 5)
        chunk_end = asyncio.get_event_loop().time()

        context = "".join(
            [
                f"ArXiv Id: {chunk['arxiv_id']}\nPaper Title: {chunk['title']}\nChunk Index: {chunk['chunk_index']}\nContent: {chunk['content']}\n\n"
                for chunk in chunks
            ]
        )
        prompt = f"""You are an intelligent research assistant. You are given the following context from research papers:
        {context}
        Based on the context provided, answer the following question concisely and provide references to the papers you used. If you cannot find an answer, say "I don't know, context not sufficient, please provide more details in your query"
        Always return the answer in markdown format.
        ."""

        provider = os.getenv(
            "COMPLETION_PROVIDER", os.getenv("EMBEDDING_PROVIDER", "OLLAMA")
        ).upper()
        logger.info(f"Generating completion using {provider}...")

        if provider == "OPENAI":
            answer = await self.openai_completions(question, prompt)
        elif provider == "OPENROUTER":
            answer = await self.open_router_completions(question, prompt)
        else:
            answer = await self.ollama_completions(question, prompt)
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

    asyncio.run(run())
