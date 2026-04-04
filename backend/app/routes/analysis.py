"""
Analysis Routes — Summarization + Classification.

POST /api/v1/analysis/summarize/{document_id}   — Summarize a document
POST /api/v1/analysis/classify/{document_id}    — Classify a document
POST /api/v1/analysis/analyze/{document_id}     — Both at once
GET  /api/v1/analysis/categories                — List all valid categories
"""

import logging
import time

from fastapi import APIRouter, HTTPException, Query, status

from app.models.schemas import (
    ClassifyResponse,
    SummarizeResponse,
)
from app.services.classifier import CATEGORIES, classify_document
from app.services.summarizer import summarize_document

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Summarize ──────────────────────────────────────────────────────────────────

@router.post(
    "/analysis/summarize/{document_id}",
    response_model=SummarizeResponse,
    status_code=status.HTTP_200_OK,
    summary="Summarize an uploaded document",
)
async def summarize(
    document_id: str,
    max_words: int = Query(
        default=500,
        ge=100,
        le=1500,
        description="Approximate maximum word count of the summary.",
    ),
):
    """
    Generate a structured summary for a previously uploaded document.

    - Short documents  (< 3000 chars) → single-pass LLM summarization.
    - Long  documents  (≥ 3000 chars) → map-reduce: summarize each
      chunk independently, then combine into a final summary.

    The response includes:
    - `summary`   : Structured text with Overview, Key Points, Conclusion.
    - `strategy`  : Which approach was used ("single_pass" | "map_reduce").
    """
    try:
        result = summarize_document(
            document_id=document_id,
            max_words=max_words,
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            f"Summarization failed for {document_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Summarization failed: {str(e)}",
        )

    return SummarizeResponse(
        document_id=result.document_id,
        filename=result.filename,
        summary=result.summary,
        processing_time_seconds=round(result.processing_time, 4),
    )


# ── Classify ───────────────────────────────────────────────────────────────────

@router.post(
    "/analysis/classify/{document_id}",
    response_model=ClassifyResponse,
    status_code=status.HTTP_200_OK,
    summary="Classify an uploaded document into a category",
)
async def classify(document_id: str):
    """
    Classify a previously uploaded document.

    Classification uses a two-layer approach:
    1. **LLM (Groq)**    — Zero-shot classification with structured JSON output.
    2. **TF-IDF**        — Keyword cosine similarity as fallback/validator.

    The response includes:
    - `predicted_category` : The winning category label.
    - `confidence`         : Score 0.0–1.0 for the predicted category.
    - `all_scores`         : Confidence scores for every category.
    """
    try:
        result = classify_document(document_id=document_id)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            f"Classification failed for {document_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Classification failed: {str(e)}",
        )

    return ClassifyResponse(
        document_id=result.document_id,
        filename=result.filename,
        predicted_category=result.predicted_category,
        confidence=round(result.confidence, 4),
        all_scores=result.all_scores,
        processing_time_seconds=round(result.processing_time, 4),
    )


# ── Combined Analysis ──────────────────────────────────────────────────────────

@router.post(
    "/analysis/analyze/{document_id}",
    status_code=status.HTTP_200_OK,
    summary="Run summarization + classification together",
)
async def full_analysis(
    document_id: str,
    max_words: int = Query(default=500, ge=100, le=1500),
):
    """
    Run both summarization and classification in sequence
    and return combined results in a single response.

    Useful for the frontend dashboard to populate both
    the summary panel and the classification badge at once.
    """
    pipeline_start = time.perf_counter()
    response: dict = {"document_id": document_id}

    # ── Summarize ──────────────────────────────────────────────────────────
    try:
        summary_result = summarize_document(
            document_id=document_id,
            max_words=max_words,
        )
        response["summary"] = {
            "text":              summary_result.summary,
            "strategy":          summary_result.strategy,
            "num_chunks_used":   summary_result.num_chunks_used,
            "processing_time":   round(summary_result.processing_time, 4),
        }
        response["filename"] = summary_result.filename
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Summarization failed in full_analysis: {e}", exc_info=True)
        response["summary"] = {"error": str(e)}

    # ── Classify ───────────────────────────────────────────────────────────
    try:
        classify_result = classify_document(document_id=document_id)
        response["classification"] = {
            "predicted_category": classify_result.predicted_category,
            "confidence":         round(classify_result.confidence, 4),
            "all_scores":         classify_result.all_scores,
            "method":             classify_result.method,
            "reasoning":          classify_result.reasoning,
            "processing_time":    round(classify_result.processing_time, 4),
        }
    except Exception as e:
        logger.error(f"Classification failed in full_analysis: {e}", exc_info=True)
        response["classification"] = {"error": str(e)}

    response["total_processing_time"] = round(
        time.perf_counter() - pipeline_start, 4
    )

    return response


# ── Categories List ────────────────────────────────────────────────────────────

@router.get(
    "/analysis/categories",
    status_code=status.HTTP_200_OK,
    summary="List all document classification categories",
)
async def list_categories():
    """
    Return all available classification categories and their
    representative keywords. Useful for frontend display and
    for understanding the classification taxonomy.
    """
    return {
        "total": len(CATEGORIES),
        "categories": [
            {
                "name":     name,
                "keywords": keywords[:8],   # show first 8 as preview
            }
            for name, keywords in CATEGORIES.items()
        ],
    }