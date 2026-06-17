# arxiv-rag/src/__init__.py

# Expose the main orchestrator class
from .models import Paper, Chunk
from .ingest import ArxivClient
from .chunk import TextChunker
from .db import PGVectorDB
from .embed import TextEmbedder
from .parse import PaperParser
from .retrieve import Retriever

# Define package metadata
__version__ = "0.1.0"

# Explicitly declare what is available when someone uses `from arxivrag import *`
__all__ = [
    "ArxivClient",
    "TextChunker",
    "Paper",
    "Chunk",
    "PGVectorDB",
    "TextEmbedder",
    "PaperParser",
    "Retriever",
]
