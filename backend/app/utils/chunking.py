"""
Text chunking utilities.
Splits extracted PDF text into overlapping chunks
suitable for embedding and retrieval.
"""

import re
import logging
from dataclasses import dataclass

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class TextChunk:
    """A single chunk of text with its metadata."""
    chunk_id: str          # "{document_id}_chunk_{index}"
    document_id: str
    filename: str
    content: str
    page: int | None       # Source page number (if available)
    chunk_index: int       # Position in the document
    total_chunks: int      # Filled in after all chunks are created


def clean_text(text: str) -> str:
    """
    Normalize raw PDF-extracted text.

    Operations:
    - Collapse multiple blank lines into one
    - Strip leading/trailing whitespace per line
    - Remove null bytes and non-printable control chars
    - Normalize unicode dashes and quotes
    """
    # Remove null bytes
    text = text.replace("\x00", "")

    # Normalize unicode punctuation
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")

    # Remove non-printable control characters (keep \n and \t)
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Strip each line
    lines = [line.strip() for line in text.split("\n")]

    # Collapse 3+ consecutive blank lines into 2
    cleaned_lines: list[str] = []
    blank_count = 0
    for line in lines:
        if line == "":
            blank_count += 1
            if blank_count <= 2:
                cleaned_lines.append(line)
        else:
            blank_count = 0
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def split_into_chunks(
    text: str,
    document_id: str,
    filename: str,
    page_map: dict[int, int] | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[TextChunk]:
    """
    Split text into overlapping chunks using a sliding window.

    Args:
        text:         Full cleaned text of the document.
        document_id:  Unique ID assigned to this document.
        filename:     Original filename (for metadata).
        page_map:     Maps character offset → page number.
                      Used to tag each chunk with its source page.
        chunk_size:   Max characters per chunk (default from settings).
        chunk_overlap: Characters to repeat between chunks (default from settings).

    Returns:
        List of TextChunk objects (total_chunks filled in at end).
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    if not text.strip():
        logger.warning(f"[{document_id}] Empty text passed to chunker.")
        return []

    chunks: list[TextChunk] = []
    text_length = len(text)
    start = 0
    chunk_index = 0

    while start < text_length:
        end = start + chunk_size

        # ── Try to break at a sentence boundary ──────────────────────────
        if end < text_length:
            # Look backwards up to 100 chars for ". " or "\n"
            boundary = _find_boundary(text, end, lookback=100)
            if boundary:
                end = boundary

        chunk_text = text[start:end].strip()

        if chunk_text:
            # Determine source page from character offset
            page = _get_page_for_offset(page_map, start) if page_map else None

            chunks.append(
                TextChunk(
                    chunk_id=f"{document_id}_chunk_{chunk_index}",
                    document_id=document_id,
                    filename=filename,
                    content=chunk_text,
                    page=page,
                    chunk_index=chunk_index,
                    total_chunks=0,  # backfilled below
                )
            )
            chunk_index += 1

        # Slide forward, minus the overlap
        start = end - chunk_overlap
        if start >= text_length:
            break

    # Backfill total_chunks
    total = len(chunks)
    for chunk in chunks:
        chunk.total_chunks = total

    logger.info(
        f"[{document_id}] Split into {total} chunks "
        f"(size={chunk_size}, overlap={chunk_overlap})"
    )
    return chunks


def _find_boundary(text: str, end: int, lookback: int = 100) -> int | None:       #Instead of cutting text randomly, this function tries to end chunks naturally at:Paragraph break (\n\n), Sentence end (. ! ?)
    """
    Search backward from `end` for a good sentence/paragraph boundary.
    Priority: paragraph break > sentence end > word boundary.
    Returns the position just after the boundary, or None.
    """
    search_start = max(0, end - lookback)
    segment = text[search_start:end]

    # Paragraph break
    idx = segment.rfind("\n\n")
    if idx != -1:
        return search_start + idx + 2

    # Sentence end
    for marker in (". ", "! ", "? "):
        idx = segment.rfind(marker)
        if idx != -1:
            return search_start + idx + len(marker)

    # Word boundary (space)
    idx = segment.rfind(" ")
    if idx != -1:
        return search_start + idx + 1

    return None


def _get_page_for_offset(page_map: dict[int, int], offset: int) -> int:  #Map Text to PDF Page
    """
    Given a character offset, return the page number.
    page_map: {cumulative_char_offset: page_number}
    """
    page = 1
    for boundary, pg in sorted(page_map.items()):
        if offset >= boundary:
            page = pg
        else:
            break
    return page