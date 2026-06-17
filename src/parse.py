from pathlib import Path
import json
from loguru import logger
import pymupdf

from models import Paper
from clean import clean_text


class PaperParser:
    def __init__(
        self,
        metadata_path: str | None = None,
        parse_dir: str | None = None,
    ) -> None:
        if metadata_path:
            self.metadata_path = Path(metadata_path)
        else:
            self.metadata_path = Path("data", "metadata.json")
        if parse_dir:
            self.parse_dir = Path(parse_dir)
        else:
            self.parse_dir = Path("data", "text")
        self.parse_dir.mkdir(parents=True, exist_ok=True)

    def parse_paper(self, paper_metadata: Paper):
        raw_path = Path(paper_metadata.raw_path or "")
        if not raw_path.exists():
            logger.error(f"Paper: {paper_metadata.title} does not exist")
            raise FileNotFoundError(raw_path)

        output_path = self.parse_dir / f"{paper_metadata.title}.txt"

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
        try:
            with open(self.metadata_path, "r") as fp:
                metadata = json.load(
                    fp, object_hook=lambda x: Paper(**x) if isinstance(x, dict) else x
                )
        except FileNotFoundError:
            logger.error(f"Metadata file not found: {self.metadata_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse metadata JSON: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to parse metadata: {e}")
            raise

        for paper in metadata:
            try:
                self.parse_paper(paper)
                logger.info(f"Parsed paper: {paper.title}")

            except Exception as e:
                logger.error(f"Failed to parse {paper.title}: {e}")

        try:
            with open(self.metadata_path, "w") as fp:
                json.dump(
                    metadata,
                    fp,
                    indent=2,
                    default=lambda obj: obj.__dict__,
                )
        except FileNotFoundError:
            logger.error(f"Metadata file not found: {self.metadata_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse metadata JSON: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to parse metadata: {e}")
            raise

        return metadata


if __name__ == "__main__":
    parser = PaperParser()

    parsed_metadata = parser.parse_papers()
