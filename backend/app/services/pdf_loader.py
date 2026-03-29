"""
PDF loading and text extraction service.

Strategy:
1. Try pdfplumber first  — best for text-heavy/structured PDFs
2. Fall back to pypdf    — handles more edge cases
3. Build a page_map for chunk ↔ page attribution
"""

import hashlib
import logging, time
from dataclasses import dataclass
from pathlib import Path

import pdfplumber
from pypdf import PdfReader

from app.config import get_settings
from app.utils.chunking import TextChunk, clean_text, split_into_chunks

logger = logging.getLogger(__name__)
settings = get_settings()

@dataclass
class PDFLoadResult:
    """Everything produced from loading and chunking one PDF."""
    document_id: str
    filename: str
    raw_text: str
    chunks: list[TextChunk]
    num_pages: int
    processing_time: float
    extraction_method: str   # "pdfplumber" | "pypdf"
    
def generate_document_id(filename: str, content: bytes) -> str:
    """
    Deterministic document ID based on filename + file content hash.
    Same file uploaded twice → same ID (deduplication).
    """
    
    hash_input= filename.encode() + content[:4096] #first 4kb is enough. Converts the filename (a string) into bytes filename.encode().
    return hashlib.sha256(hash_input).hexdigest()[:16] # hexdigest() Converts the hash into a hexadecimal string (readable format)

def extract_text_pdfplumber(filepath: Path) -> tuple[str, dict[int, int], int]:
    """
    Extract text using pdfplumber.

    Returns:
        (full_text, page_map, num_pages)
        page_map: {cumulative_char_offset: page_number}
    """
    full_text_parts: list[str] = []
    page_map: dict[int, int] = {}
    cumulative_offset = 0

    with pdfplumber.open(filepath) as pdf:
        num_pages = len(pdf.pages)
        for page_num, page in enumerate(pdf.pages, start=1):
            page_map[cumulative_offset] = page_num
            text = page.extract_text() or ""
            full_text_parts.append(text)
            cumulative_offset += len(text) + 1  # +1 for the "\n" join

    return "\n".join(full_text_parts), page_map, num_pages

def extract_text_pypdf(filepath: Path) -> tuple[str, dict[int, int], int]:
    """
    Fallback extraction using pypdf.

    Returns:
        (full_text, page_map, num_pages)
    """
    reader = PdfReader(str(filepath))
    full_text_parts: list[str] = []
    page_map: dict[int, int] = {}
    cumulative_offset = 0
    num_pages = len(reader.pages)

    for page_num, page in enumerate(reader.pages, start=1):
        page_map[cumulative_offset] = page_num + 1
        text = page.extract_text() or ""
        full_text_parts.append(text)
        cumulative_offset += len(text) + 1

    return "\n".join(full_text_parts), page_map, num_pages


def load_and_chunk_pdf(
    filepath: Path,
    filename: str,
    file_content: bytes,
) -> PDFLoadResult:
    """
    Main entry point: load a PDF file and return chunks ready for embedding.

    Steps:
        1. Generate deterministic document_id
        2. Extract text (pdfplumber → pypdf fallback)
        3. Clean extracted text
        4. Split into overlapping chunks
        5. Return PDFLoadResult

    Args:
        filepath:     Absolute path to the saved PDF on disk.
        filename:     Original filename from the upload.
        file_content: Raw bytes of the PDF (for ID generation).
    """
    start = time.perf_counter()
    document_id = generate_document_id(filename, file_content)

    logger.info(f"[{document_id}] Loading PDF: {filename}")

    # ── Step 1: Extract text ───────────────────────────────────────────────
    extraction_method = "pdfplumber"
    try:
        raw_text, page_map, num_pages = extract_text_pdfplumber(filepath)

        # If pdfplumber returns almost nothing, try pypdf
        if len(raw_text.strip()) < 100:
            logger.warning(
                f"[{document_id}] pdfplumber returned minimal text, "
                "falling back to pypdf."
            )
            raise ValueError("Insufficient text from pdfplumber")

    except Exception as e:
        logger.warning(f"[{document_id}] pdfplumber failed ({e}), using pypdf.")
        extraction_method = "pypdf"
        raw_text, page_map, num_pages = extract_text_pypdf(filepath)

    logger.info(
        f"[{document_id}] Extracted {len(raw_text):,} chars "
        f"from {num_pages} pages via {extraction_method}"
    )

    # ── Step 2: Clean text ─────────────────────────────────────────────────
    cleaned_text = clean_text(raw_text)

    # ── Step 3: Chunk ──────────────────────────────────────────────────────
    chunks = split_into_chunks(
        text=cleaned_text,
        document_id=document_id,
        filename=filename,
        page_map=page_map,
    )

    processing_time = time.perf_counter() - start
    logger.info(
        f"[{document_id}] Done in {processing_time:.3f}s — "
        f"{len(chunks)} chunks produced."
    )

    return PDFLoadResult(
        document_id=document_id,
        filename=filename,
        raw_text=cleaned_text,
        chunks=chunks,
        num_pages=num_pages,
        processing_time=processing_time,
        extraction_method=extraction_method,
    )