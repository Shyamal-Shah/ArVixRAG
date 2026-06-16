# arxiv-rag/src/__init__.py

# Expose the main orchestrator class
from .models import Paper
from .ingest import ArxivClient

# Define package metadata
__version__ = "0.1.0"

# Explicitly declare what is available when someone uses `from arxivrag import *`
__all__ = ["ArxivClient", "Paper"]
