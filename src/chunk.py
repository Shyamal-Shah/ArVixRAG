from pathlib import Path
from loguru import logger
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from .models import Paper, Chunk
    from .persistence import load_papers, save_chunks
except ImportError:
    from models import Paper, Chunk
    from persistence import load_papers, save_chunks

DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 50


class TextChunker:
    def __init__(
        self,
        metadata_path: str | None = None,
        chunks_path: str | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        self.metadata_path = (
            Path(metadata_path) if metadata_path else Path("data", "metadata.json")
        )
        self.chunks_path = (
            Path(chunks_path) if chunks_path else Path("data", "chunks.json")
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,  # Maximum characters per chunk
            chunk_overlap=overlap,  # Number of overlapping characters between chunks
            length_function=len,  # Function to calculate chunk size
        )

    def chunk_paper(self, paper_metadata: Paper) -> list[Chunk]:
        parsed_path = Path(paper_metadata.parsed_path or "")
        if not parsed_path.exists():
            logger.error(f"Parsed Paper: {paper_metadata.title} does not exist")
            raise FileNotFoundError(parsed_path)

        chunks = []
        with open(parsed_path, "r") as fp:
            texts = self.text_splitter.split_text(fp.read())
            for i, text in enumerate(texts):
                chunks.append(
                    Chunk(
                        arxiv_id=paper_metadata.id,
                        paper_title=paper_metadata.title,
                        chunk_index=i,
                        content=text,
                    )
                )
        return chunks

    def chunk_papers(self) -> list[Chunk]:
        metadata = load_papers(self.metadata_path)

        chunks: list[Chunk] = []
        for paper in metadata:
            try:
                chunks.extend(self.chunk_paper(paper))
                logger.info(f"Chunked paper: {paper.title}")
            except Exception as e:
                logger.error(f"Failed to chunk {paper.title}: {e}")

        save_chunks(chunks, self.chunks_path)
        return chunks


if __name__ == "__main__":
    chunker = TextChunker()

    chunks = chunker.chunk_papers()

    logger.info(f"Total Chunks: {len(chunks)}")
