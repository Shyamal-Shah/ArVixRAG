from arxiv import Client, Search, SortCriterion
from loguru import logger
from urllib.request import urlretrieve
from pathlib import Path
import json

try:
    from .models import Paper
except ImportError:
    from models import Paper


class ArxivClient:
    def __init__(self, download_dir: str | None = None) -> None:
        self.client = Client()
        if download_dir:
            self.download_dir = Path(download_dir)
        else:
            self.download_dir = Path("data", "papers")
        self.download_dir.mkdir(parents=True, exist_ok=True)

    def ingest_papers(self, query: str, max_results: int = 10) -> list:
        new_metadata: list[Paper] = []
        try:

            logger.debug(f"Searching ArXiv with query: '{query}'")
            search_query = Search(
                query, max_results=max_results, sort_by=SortCriterion.SubmittedDate
            )
            papers = self.client.results(search_query)

            logger.info(f"Fetched papers from ArXiv with query: '{query}'")

            for i, paper in enumerate(papers):
                logger.debug(f"{i}. Downloading: {paper.title}")
                raw_path = self.download_dir.joinpath(f"{paper.title}.pdf").as_posix()
                new_metadata.append(
                    Paper(
                        id=paper.entry_id,
                        title=paper.title,
                        authors=[a.name for a in paper.authors or []],
                        primary_category=paper.primary_category,
                        categories=paper.categories,
                        summary=paper.summary,
                        published=str(paper.published),
                        pdf_url=paper.pdf_url,
                        raw_path=str(raw_path),
                    )
                )
                if paper.pdf_url:
                    urlretrieve(paper.pdf_url, str(raw_path))
                logger.debug(f"{i}. Downloaded: {paper.title}")

            metadata = []
            metadata_file_path = self.download_dir.joinpath("..", "metadata.json")
            try:
                with open(metadata_file_path, "r") as fp:
                    metadata = json.load(
                        fp,
                        object_hook=lambda x: Paper(**x) if isinstance(x, dict) else x,
                    )
            except (FileNotFoundError, json.JSONDecodeError) as e:
                logger.debug(f"Could not load metadata.json, starting fresh: {e}")
            metadata.extend(new_metadata)
            try:
                with open(metadata_file_path, "w") as fp:
                    json.dump(
                        metadata,
                        fp,
                        indent=2,
                        default=lambda obj: obj.__dict__,
                    )
            except IOError as e:
                logger.error(f"Failed to write metadata.json: {e}")

            return new_metadata

        except Exception as e:
            logger.error(f"Failed to ingest papers: {e}")
            raise


if __name__ == "__main__":
    arxivClient = ArxivClient()

    queries = [
        "retrieval augmented generation",
        "large language model agents",
        "transformer attention mechanism",
        "instruction fine-tuning LLM",
        "LLM-Agent Workflows",
    ]

    for q in queries:
        results = arxivClient.ingest_papers(query=f"({q}) AND cat:cs.CL")
        print(f"{len(results)} papers fetched for query: {q}")
