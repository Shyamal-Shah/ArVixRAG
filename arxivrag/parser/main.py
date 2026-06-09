import pymupdf4llm
from pathlib import Path
from loguru import logger


class PDFparser:

    def __init__(self):
        pass

    def parse_file(self, file_path: str, parsed_dir: str = "./data/parsed_md") -> Path:
        if not file_path:
            raise Exception("File path is not passed")

        file = Path(file_path)

        if not file.exists():
            raise Exception(f"File not found: {file_path}")

        parsed_path = Path(parsed_dir)
        parsed_path.mkdir(parents=True, exist_ok=True)

        md_text = pymupdf4llm.to_markdown(file_path)
        if isinstance(md_text, str):
            markdown_text = md_text
        else:
            markdown_text = "\n\n".join(str(page.get("text", "")) for page in md_text)

        output_file = parsed_path / f"{file.stem}.md"
        output_file.write_text(markdown_text, encoding="utf-8")
        return output_file

    def parse_dir(
        self, raw_dir: str = "./data/raw", parsed_dir: str = "./data/parsed_md"
    ):
        raw_path = Path(raw_dir)
        for item in raw_path.iterdir():
            logger.debug(f"Currently parsing: {item}")
            if item.is_file() and item.suffix.lower() == ".pdf":
                self.parse_file(str(item), parsed_dir=parsed_dir)
