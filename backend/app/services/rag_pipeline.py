"""
RAG Pipeline — Orchestrator.

Wires together all services into one clean pipeline:

  User Question
       │
       ▼
  [1] Rewrite question (if chat history exists)
       │
       ▼
  [2] Embed query → vector
       │
       ▼
  [3] Retrieve top-k chunks from vector store
       │
       ▼
  [4] Format retrieved chunks → context string
       │
       ▼
  [5] Call LLM (Groq) with system prompt + context + question
       │
       ▼
  [6] Return answer + source citations + metadata
"""

import logging
import time
from dataclasses import dataclass, field

from app.config import get_settings
from app.models.schemas import ChatMessage, QueryRequest, QueryResponse, SourceChunk
from app.services.llm import (
    build_context_string,
    generate_answer,
    rewrite_question_for_context,
)
from app.services.vector_store import SearchResult, get_vector_store

logger = logging.getLogger(__name__)
settings = get_settings()


# ── Pipeline Result ────────────────────────────────────────────────────────────

@dataclass
class RAGResult:
    """Complete output of one RAG pipeline run."""
    answer: str
    sources: list[SearchResult]
    question_original: str
    question_rewritten: str
    context_used: str
    model_used: str
    retrieval_time: float
    generation_time: float
    total_time: float
    num_chunks_retrieved: int


# ── Main Pipeline ──────────────────────────────────────────────────────────────

def run_rag_pipeline(
    question: str,
    chat_history: list[dict] | None = None,
    document_id: str | None = None,
    top_k: int | None = None,
) -> RAGResult:
    """
    Execute the full RAG pipeline synchronously.

    Args:
        question:     The user's raw question string.
        chat_history: List of previous {"role", "content"} dicts.
                      Used for question rewriting and LLM context.
        document_id:  If set, restrict retrieval to one document.
                      If None, search across all indexed documents.
        top_k:        Number of chunks to retrieve. Defaults to
                      settings.top_k_results if not provided.

    Returns:
        RAGResult with answer, sources, and timing breakdown.

    Raises:
        RuntimeError: If retrieval is empty and LLM cannot generate.
        ValueError:   If question is empty.
    """
    chat_history = chat_history or []
    top_k = top_k or settings.top_k_results
    pipeline_start = time.perf_counter()

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    logger.info(
        f"RAG pipeline started | "
        f"question='{question[:80]}' | "
        f"history_turns={len(chat_history)} | "
        f"document_id={document_id!r} | "
        f"top_k={top_k}"
    )

    # ── STEP 1: Rewrite question for retrieval ─────────────────────────────
    question_rewritten = rewrite_question_for_context(question, chat_history)

    # ── STEP 2 & 3: Retrieve relevant chunks ──────────────────────────────
    retrieval_start = time.perf_counter()

    store = get_vector_store()
    search_results: list[SearchResult] = store.search(
        query=question_rewritten,
        top_k=top_k,
        document_id=document_id,
    )

    retrieval_time = time.perf_counter() - retrieval_start

    logger.info(
        f"Retrieved {len(search_results)} chunks in {retrieval_time:.3f}s"
    )

    # Log top results for debugging
    for i, r in enumerate(search_results[:3], 1):
        logger.debug(
            f"  Chunk {i}: score={r.score:.4f} | "
            f"doc={r.document_id} | page={r.page} | "
            f"preview='{r.content[:60]}...'"
        )

    # ── STEP 4: Build context string ───────────────────────────────────────
    chunks_for_context = [
        {
            "content":     r.content,
            "filename":    r.filename,
            "page":        r.page,
            "score":       r.score,
            "document_id": r.document_id,
        }
        for r in search_results
    ]
    context_string = build_context_string(chunks_for_context)

    # ── STEP 5: Generate answer with LLM ──────────────────────────────────
    answer, generation_time = generate_answer(
        question=question_rewritten,
        context=context_string,
        chat_history=chat_history,
    )

    total_time = time.perf_counter() - pipeline_start

    logger.info(
        f"RAG pipeline complete | "
        f"retrieval={retrieval_time:.3f}s | "
        f"generation={generation_time:.3f}s | "
        f"total={total_time:.3f}s"
    )

    return RAGResult(
        answer=answer,
        sources=search_results,
        question_original=question,
        question_rewritten=question_rewritten,
        context_used=context_string,
        model_used=settings.groq_model,
        retrieval_time=retrieval_time,
        generation_time=generation_time,
        total_time=total_time,
        num_chunks_retrieved=len(search_results),
    )


# ── Chat History Helpers ───────────────────────────────────────────────────────

def schema_history_to_dicts(
    chat_history: list[ChatMessage],
) -> list[dict]:
    """
    Convert Pydantic ChatMessage list (from API schema) to
    the plain dict format used internally by the pipeline.

    Args:
        chat_history: List of ChatMessage schema objects.

    Returns:
        List of {"role": str, "content": str} dicts.
    """
    return [
        {"role": msg.role, "content": msg.content}
        for msg in chat_history
        if msg.role in ("user", "assistant")
    ]


def build_query_response(
    request: QueryRequest,
    result: RAGResult,
) -> QueryResponse:
    """
    Map a RAGResult onto the QueryResponse Pydantic schema
    returned to the frontend.

    Args:
        request: The original QueryRequest from the API.
        result:  The RAGResult from run_rag_pipeline().

    Returns:
        QueryResponse ready to be serialized as JSON.
    """
    source_chunks = [
        SourceChunk(
            content=src.content,
            document_id=src.document_id,
            filename=src.filename,
            page=src.page,
            score=round(src.score, 4),
        )
        for src in result.sources
    ]

    return QueryResponse(
        answer=result.answer,
        sources=source_chunks,
        question=result.question_original,
        model_used=result.model_used,
        processing_time_seconds=round(result.total_time, 4),
    )