from pathlib import Path
import json
from loguru import logger
from langchain_text_splitters import RecursiveCharacterTextSplitter

from models import Paper


class TextChunker:
    def __init__(
        self,
        metadata_path: str | None = None,
        chunks_path: str | None = None,
        chunk_size: int = 512,
        overlap: int = 50,
    ) -> None:
        if metadata_path:
            self.metadata_path = Path(metadata_path)
        else:
            self.metadata_path = Path("data", "metadata.json")
        if chunks_path:
            self.chunks_path = Path(chunks_path)
        else:
            self.chunks_path = Path("data", "chunks.json")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,  # Maximum characters per chunk
            chunk_overlap=overlap,  # Number of overlapping characters between chunks
            length_function=len,  # Function to calculate chunk size
        )

    def chunk_paper(self, paper_metadata: Paper) -> list[dict]:
        parsed_path = Path(paper_metadata.parsed_path or "")
        if not parsed_path.exists():
            logger.error(f"Parsed Paper: {paper_metadata.title} does not exist")
            raise FileNotFoundError(parsed_path)

        chunks = []
        with open(parsed_path, "r") as fp:
            texts = self.text_splitter.split_text(fp.read())
            for i, text in enumerate(texts):
                chunks.append(
                    {
                        "arxiv_id": paper_metadata.id,
                        "paper_title": paper_metadata.title,
                        "chunk_index": i,
                        "chunk_text": text,
                    }
                )
        return chunks

    def chunk_papers(self):
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

        chunks = []
        for paper in metadata:
            try:
                chunks.extend(self.chunk_paper(paper))
                logger.info(f"Chunked paper: {paper.title}")
            except Exception as e:
                logger.error(f"Failed to chunk {paper.title}: {e}")

        try:
            with open(self.chunks_path, "w") as fp:
                json.dump(chunks, fp, indent=2, default=lambda obj: obj.__dict__)
        except IOError as e:
            logger.error(f"Failed to write chunks file: {e}")
            raise

        return chunks


if __name__ == "__main__":
    chunker = TextChunker()

    chunks = chunker.chunk_papers()

    logger.info(f"Total Chunks: {len(chunks)}")
