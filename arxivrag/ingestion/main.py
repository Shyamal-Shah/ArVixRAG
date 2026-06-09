from arxiv import Client, Search, SortCriterion
from urllib.request import urlretrieve
import os
import json
from datetime import datetime
from loguru import logger

_DEFAULT_TIME = datetime.min


class ArXivResult:
    entry_id: str
    title: str
    authors: list[str] | None
    summary: str
    published: str | None = None
    pdf_url: str | None
    primary_category: str
    categories: list[str] | None
    local_path: str | None = None

    def __init__(
        self,
        entry_id: str,
        title: str = "",
        authors: list[str] | None = None,
        summary: str = "",
        published: datetime = _DEFAULT_TIME,
        primary_category: str = "",
        categories: list[str] | None = None,
        pdf_url: str | None = None,
        local_path: str | None = None,
    ):
        self.entry_id = entry_id
        self.title = title
        self.authors = authors
        self.summary = summary
        self.published = str(published)
        self.primary_category = primary_category
        self.categories = categories
        self.pdf_url = pdf_url
        self.local_path = local_path


class ArXivFetcher:

    def __init__(self, download_dir: str = "./data/raw") -> None:
        self.download_dir = download_dir
        os.makedirs(self.download_dir, exist_ok=True)
        self.client = Client()

    def fetch_by_query(self, query: str, max_results: int = 10) -> list[ArXivResult]:
        new_data: list[ArXivResult] = []
        old_data: list[ArXivResult] = []

        search = Search(
            query=query,
            max_results=max_results,
            sort_by=SortCriterion.SubmittedDate,
        )

        results = self.client.results(search)

        for r in results:
            local_path = os.path.join(self.download_dir, f"{r.title}.pdf")
            logger.debug(f"Downloading {local_path}...", end="")
            new_data.append(
                ArXivResult(
                    entry_id=r.entry_id,
                    title=r.title,
                    authors=[a.name for a in r.authors or []],
                    primary_category=r.primary_category,
                    categories=r.categories,
                    summary=r.summary,
                    published=r.published,
                    pdf_url=r.pdf_url,
                    local_path=local_path,
                )
            )
            if r.pdf_url:
                urlretrieve(r.pdf_url, local_path)
            logger.debug("Completed.")

        try:
            with open(
                os.path.join(self.download_dir, "..", "papers_metadata.json"), "r"
            ) as fp:
                old_data = json.load(fp)
        except:
            logger.error(f"Error Opening the papers_metadata.json")

        old_data.extend(new_data)

        with open(
            os.path.join(os.path.join(self.download_dir, "..", "papers_metadata.json")),
            "w",
        ) as fp:
            json.dump(old_data, fp=fp, default=lambda obj: obj.__dict__, indent=2)

        return new_data
