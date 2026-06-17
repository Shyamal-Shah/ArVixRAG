class Paper:
    id: str
    title: str
    authors: list[str] | None
    summary: str
    published: str | None = None
    pdf_url: str | None
    primary_category: str
    categories: list[str] | None
    raw_path: str | None = None
    parsed_path: str | None = None

    def __init__(
        self,
        id: str,
        title: str = "",
        authors: list[str] | None = None,
        summary: str = "",
        published: str = "",
        primary_category: str = "",
        categories: list[str] | None = None,
        pdf_url: str | None = None,
        raw_path: str | None = None,
        parsed_path: str | None = None,
    ):
        self.id = id
        self.title = title
        self.authors = authors
        self.summary = summary
        self.published = published
        self.primary_category = primary_category
        self.categories = categories
        self.pdf_url = pdf_url
        self.raw_path = raw_path
        self.parsed_path = parsed_path
