from dataclasses import dataclass


@dataclass
class Paper:
    id: str
    title: str = ""
    authors: list[str] | None = None
    summary: str = ""
    published: str | None = None
    primary_category: str = ""
    categories: list[str] | None = None
    pdf_url: str | None = None
    raw_path: str | None = None
    parsed_path: str | None = None


@dataclass
class Chunk:
    arxiv_id: str | None = None
    paper_title: str | None = None
    chunk_index: int | None = None
    content: str | None = None
    embedding: list[float] | None = None
