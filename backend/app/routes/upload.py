"""
PDF Upload API route.

POST /api/v1/upload        — Upload and process a PDF
GET  /api/v1/documents     — List all uploaded documents
DELETE /api/v1/documents/{document_id} — Remove a document
"""

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.config import get_settings
from app.models.schemas import (
    DocumentInfo,
    DocumentListResponse,
    UploadResponse,
)
from app.services.pdf_loader import load_and_chunk_pdf

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter()

# ── Constants ──────────────────────────────────────────────────────────────────
MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"application/pdf", "application/x-pdf"}

# Metadata store path (simple JSON file — replaced by a DB in production)
METADATA_FILE = Path(settings.docs_dir) / "_metadata.json"


# ── Metadata helpers ───────────────────────────────────────────────────────────

def _load_metadata() -> dict:
    """Load document metadata from the JSON store."""
    if METADATA_FILE.exists():
        try:
            return json.loads(METADATA_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Metadata file corrupted, resetting.")
    return {}


def _save_metadata(metadata: dict) -> None:
    """Persist document metadata to disk."""
    METADATA_FILE.write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and process a PDF document",
)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file. The pipeline will:
    1. Validate file type and size
    2. Save to disk
    3. Extract and clean text
    4. Chunk the text
    5. Return document_id and chunk count

    The document_id is deterministic — uploading the same file twice
    returns the same ID without reprocessing.
    """

    # ── Validate content type ──────────────────────────────────────────────
    content_type = file.content_type or ""
    filename = file.filename or "unknown.pdf"

    if content_type not in ALLOWED_CONTENT_TYPES and not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Only PDF files are accepted. Got: '{content_type}'",
        )

    # ── Read file content ──────────────────────────────────────────────────
    file_content = await file.read()

    if len(file_content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    if len(file_content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {MAX_FILE_SIZE_MB} MB.",
        )

    # ── Save PDF to disk ───────────────────────────────────────────────────
    docs_dir = Path(settings.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize filename
    safe_filename = _sanitize_filename(filename)
    save_path = docs_dir / safe_filename

    try:
        save_path.write_bytes(file_content)
        logger.info(f"Saved uploaded file: {save_path}")
    except OSError as e:
        logger.error(f"Failed to save file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save uploaded file.",
        )

    # ── Load and chunk PDF ─────────────────────────────────────────────────
    try:
        result = load_and_chunk_pdf(
            filepath=save_path,
            filename=safe_filename,
            file_content=file_content,
        )
    except Exception as e:
        # Clean up saved file if processing fails
        save_path.unlink(missing_ok=True)
        logger.error(f"PDF processing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not process PDF: {str(e)}",
        )

    # ── Deduplication check ────────────────────────────────────────────────
    metadata = _load_metadata()

    if result.document_id in metadata:
        logger.info(
            f"Duplicate upload detected for document_id={result.document_id}. "
            "Returning cached result."
        )
        existing = metadata[result.document_id]
        return UploadResponse(
            success=True,
            message="Document already processed (duplicate detected).",
            document_id=result.document_id,
            filename=existing["filename"],
            num_chunks=existing["num_chunks"],
            processing_time_seconds=result.processing_time,
        )

    # ── Persist metadata ───────────────────────────────────────────────────
    metadata[result.document_id] = {
        "filename": result.filename,
        "num_chunks": len(result.chunks),
        "num_pages": result.num_pages,
        "extraction_method": result.extraction_method,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "file_path": str(save_path),
    }
    _save_metadata(metadata)

    # ── Store chunks for later steps (embedding) ───────────────────────────
    # Chunks are serialized to a JSON sidecar file.
    # In Step 5 (Embeddings), these will be read and vectorized.
    chunks_path = docs_dir / f"{result.document_id}_chunks.json"
    chunks_data = [
        {
            "chunk_id": c.chunk_id,
            "document_id": c.document_id,
            "filename": c.filename,
            "content": c.content,
            "page": c.page,
            "chunk_index": c.chunk_index,
            "total_chunks": c.total_chunks,
        }
        for c in result.chunks
    ]
    chunks_path.write_text(
        json.dumps(chunks_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"Chunks saved to {chunks_path}")

    return UploadResponse(
        success=True,
        message=f"PDF processed successfully via {result.extraction_method}.",
        document_id=result.document_id,
        filename=result.filename,
        num_chunks=len(result.chunks),
        processing_time_seconds=round(result.processing_time, 4),
    )


@router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List all uploaded documents",
)
async def list_documents():
    """Return metadata for every document that has been uploaded."""
    metadata = _load_metadata()

    documents = [
        DocumentInfo(
            document_id=doc_id,
            filename=info["filename"],
            num_chunks=info["num_chunks"],
            uploaded_at=datetime.fromisoformat(info["uploaded_at"]),
        )
        for doc_id, info in metadata.items()
    ]

    return DocumentListResponse(documents=documents, total=len(documents))


@router.delete(
    "/documents/{document_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete a document and its associated data",
)
async def delete_document(document_id: str):
    """
    Remove a document's PDF, chunks, and metadata entry.
    Vector store entries will be cleaned in Step 6.
    """
    metadata = _load_metadata()

    if document_id not in metadata:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )

    doc_info = metadata[document_id]
    docs_dir = Path(settings.docs_dir)

    # Remove PDF file
    pdf_path = Path(doc_info.get("file_path", ""))
    if pdf_path.exists():
        pdf_path.unlink()
        logger.info(f"Deleted PDF: {pdf_path}")

    # Remove chunks sidecar
    chunks_path = docs_dir / f"{document_id}_chunks.json"
    if chunks_path.exists():
        chunks_path.unlink()
        logger.info(f"Deleted chunks: {chunks_path}")

    # Remove from metadata
    del metadata[document_id]
    _save_metadata(metadata)

    return {
        "success": True,
        "message": f"Document '{doc_info['filename']}' deleted successfully.",
        "document_id": document_id,
    }


# ── Helpers ────────────────────────────────────────────────────────────────────

def _sanitize_filename(filename: str) -> str:
    """
    Make a filename safe for disk storage.
    Strips path separators, replaces spaces, and limits length.
    """
    # Remove directory traversal characters
    filename = Path(filename).name

    # Replace spaces and special chars
    import re
    filename = re.sub(r"[^\w.\-]", "_", filename)

    # Limit to 200 chars (before extension)
    stem = Path(filename).stem[:180]
    suffix = Path(filename).suffix or ".pdf"
    return f"{stem}{suffix}"