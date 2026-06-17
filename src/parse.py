from pathlib import Path
from loguru import logger
import pymupdf

try:
    from .models import Paper
    from .clean import clean_text
    from .persistence import load_papers, save_papers, safe_filename
except ImportError:
    from models import Paper
    from clean import clean_text
    from persistence import load_papers, save_papers, safe_filename


class PaperParser:
    def __init__(
        self,
        metadata_path: str | None = None,
        parse_dir: str | None = None,
    ) -> None:
        self.metadata_path = (
            Path(metadata_path) if metadata_path else Path("data", "metadata.json")
        )
        self.parse_dir = Path(parse_dir) if parse_dir else Path("data", "text")
        self.parse_dir.mkdir(parents=True, exist_ok=True)

    def parse_paper(self, paper_metadata: Paper) -> None:
        raw_path = Path(paper_metadata.raw_path or "")
        if not raw_path.exists():
            logger.error(f"Paper: {paper_metadata.title} does not exist")
            raise FileNotFoundError(raw_path)

        output_path = self.parse_dir / f"{safe_filename(paper_metadata.id)}.txt"

        # Extract text from all pages
        full_text = ""
        with pymupdf.open(raw_path) as pdf:
            for page in pdf:
                text = page.get_text()
                if isinstance(text, str):
                    full_text += text + "\n"

        # Clean the extracted text
        cleaned_text = clean_text(full_text)

        # Write cleaned text to file
        with open(output_path, "w", encoding="utf8") as fp:
            fp.write(cleaned_text)

        paper_metadata.parsed_path = output_path.as_posix()

    def parse_papers(self) -> list[Paper]:
        metadata = load_papers(self.metadata_path)

        for paper in metadata:
            try:
                self.parse_paper(paper)
                logger.info(f"Parsed paper: {paper.title}")
            except Exception as e:
                logger.error(f"Failed to parse {paper.title}: {e}")

        save_papers(metadata, self.metadata_path)
        return metadata


if __name__ == "__main__":
    parser = PaperParser()

    parsed_metadata = parser.parse_papers()
