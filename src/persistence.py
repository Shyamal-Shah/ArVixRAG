import dataclasses
import json
import re
from pathlib import Path

from loguru import logger

try:
    from .models import Paper, Chunk
except ImportError:
    from models import Paper, Chunk


def safe_filename(arxiv_id: str | None, max_len: int = 200) -> str:
    """Derive a filesystem-safe filename stem from an arxiv id/URL.

    ArXiv titles routinely contain '/', ':', and exceed the OS name limit, so we
    derive the stem from the stable id (e.g. ".../abs/2506.12345v1") instead.
    """
    stem = (arxiv_id or "").rstrip("/").split("/")[-1]
    stem = re.sub(r"[^A-Za-z0-9._-]", "_", stem)
    return stem[:max_len] or "paper"


def _load(path: str | Path, factory):
    try:
        with open(path, "r") as fp:
            return json.load(
                fp, object_hook=lambda x: factory(x) if isinstance(x, dict) else x
            )
    except FileNotFoundError:
        logger.error(f"File not found: {path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON {path}: {e}")
        raise


def _save(items, path: str | Path) -> None:
    try:
        with open(path, "w") as fp:
            json.dump([dataclasses.asdict(item) for item in items], fp, indent=2)
    except (OSError, TypeError) as e:
        logger.error(f"Failed to write {path}: {e}")
        raise


def load_papers(path: str | Path) -> list[Paper]:
    return _load(path, lambda x: Paper(**x))


def load_chunks(path: str | Path) -> list[Chunk]:
    return _load(path, lambda x: Chunk(**x))


def save_papers(papers: list[Paper], path: str | Path) -> None:
    _save(papers, path)


def save_chunks(chunks: list[Chunk], path: str | Path) -> None:
    _save(chunks, path)
