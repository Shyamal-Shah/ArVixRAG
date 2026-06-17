from collections import Counter
from loguru import logger


def clean_text(text: str, min_line_freq: int = 5) -> str:
    """
    Clean parsed PDF text by removing noise and normalizing whitespace.

    Args:
        text: Raw text from PDF parsing
        min_line_freq: Minimum frequency threshold to consider a line as header/footer

    Returns:
        Cleaned text
    """
    lines = text.split("\n")

    # Count line frequencies to identify repeated headers/footers
    line_freq = Counter(lines)
    header_footer_lines = {
        line
        for line, count in line_freq.items()
        if count >= min_line_freq and len(line.strip()) < 20
    }

    cleaned_lines = []
    for line in lines:
        # Skip header/footer patterns
        if line in header_footer_lines:
            continue

        # Skip lines that are just page numbers or whitespace
        stripped = line.strip()
        if not stripped or stripped.isdigit():
            continue

        # Remove non-ASCII characters (broken equation parsing)
        cleaned_line = stripped.encode("ascii", errors="ignore").decode("ascii")

        # Collapse repeated whitespace
        cleaned_line = " ".join(cleaned_line.split())

        if cleaned_line:
            cleaned_lines.append(cleaned_line)

    return "\n".join(cleaned_lines)
