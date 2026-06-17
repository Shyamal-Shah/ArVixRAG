import asyncio
import json
from loguru import logger

from src import ArxivClient, TextChunker, PaperParser, Retriever, TextEmbedder


async def main():
    logger.info("Hello from arxiv-rag!")
    logger.info("Let's start by downloading some sample papers")

    arxivClient = ArxivClient()

    results = arxivClient.ingest_papers(
        query="cat:cs.AI OR cat:cs.LG OR cat:cs.CL OR cat:cs.CV",
        max_results=50,
    )
    logger.info(
        f"{len(results)} papers fetched for query: cat:cs.AI OR cat:cs.LG OR cat:cs.CL OR cat:cs.CV"
    )

    logger.info("Ingestion complete. Lets parse the papers next...")

    parser = PaperParser()
    parsed_papers = parser.parse_papers()

    logger.info(f"Total Parsed Papers: {len(parsed_papers)}")

    logger.info("Parsing complete. Let's chunk the papers next...")

    chunker = TextChunker()
    chunks = chunker.chunk_papers()

    logger.info(f"Total Chunks: {len(chunks)}")

    logger.info("Chunking complete. Let's embed the chunks next...")

    embedder = TextEmbedder()
    await embedder.embed_chunks()

    logger.info("Embedding complete. Let's retrieve some chunks next...")

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


if __name__ == "__main__":
    asyncio.run(main())
