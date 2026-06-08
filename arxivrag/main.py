from ingestion import ArXivFetcher, ArXivResult

if __name__ == "__main__":
    arxivFetcher = ArXivFetcher()

    queries = [
        "retrieval augmented generation",
        "large language model agents",
        "transformer attention mechanism",
        "instruction fine-tuning LLM",
        "chain of thought reasoning",
    ]

    for q in queries:
        results = arxivFetcher.fetch_by_query(query=f"all:{q}")
        print(f"Following papers fetched for query: {q}")
        for i, r in enumerate(results):
            print(f"{i}. {r.title}")
