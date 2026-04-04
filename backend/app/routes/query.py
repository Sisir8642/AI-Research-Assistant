"""
Query API Routes.

POST /api/v1/query          — Ask a question (full RAG pipeline)
POST /api/v1/query/stream   — Streaming answer (token by token)
GET  /api/v1/query/search   — Raw vector search (debug/inspect)
GET  /api/v1/query/history  — Retrieve stored chat history
DELETE /api/v1/query/history — Clear chat history
"""

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import StreamingResponse

from app.config import get_settings
from app.models.schemas import (
    ChatMessage,
    QueryRequest,
    QueryResponse,
    SourceChunk,
)
from app.services.rag_pipeline import (
    build_query_response,
    run_rag_pipeline,
    schema_history_to_dicts,
)
from app.services.vector_store import get_vector_store

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter()

# ── Chat history store (file-based, replaced by DB in production) ──────────────
HISTORY_DIR = Path(settings.docs_dir).parent / "chat_history"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)


# ── History helpers ────────────────────────────────────────────────────────────

def _history_path(session_id: str) -> Path:
    safe_id = "".join(c for c in session_id if c.isalnum() or c in "-_")[:64]
    return HISTORY_DIR / f"{safe_id}.json"


def _load_history(session_id: str) -> list[dict]:
    path = _history_path(session_id)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning(f"Corrupt history for session {session_id}, resetting.")
    return []


def _save_history(session_id: str, history: list[dict]) -> None:
    path = _history_path(session_id)
    path.write_text(
        json.dumps(history, indent=2, default=str),
        encoding="utf-8",
    )


def _append_to_history(
    session_id: str,
    question: str,
    answer: str,
    max_turns: int = 20,
) -> list[dict]:
    """
    Append the latest Q&A pair to the session history.
    Trims to the last `max_turns` turns to prevent unbounded growth.
    """
    history = _load_history(session_id)

    history.append({
        "role": "user",
        "content": question,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    history.append({
        "role": "assistant",
        "content": answer,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    # Keep only last max_turns * 2 messages (each turn = 2 messages)
    if len(history) > max_turns * 2:
        history = history[-(max_turns * 2):]

    _save_history(session_id, history)
    return history


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask a question — full RAG pipeline",
)
async def query_documents(
    request: QueryRequest,
    session_id: str = Query(
        default="default",
        description="Session ID for chat memory. "
                    "Use a unique ID per user/conversation.",
    ),
):
    """
    Full RAG pipeline:
    1. Rewrite question using chat history (if any)
    2. Embed the question
    3. Retrieve top-k relevant chunks from vector store
    4. Pass context + question + history to Groq LLM
    5. Return answer with source citations

    Provide `session_id` to enable persistent memory across requests.
    Provide `document_id` to restrict search to a specific document.
    """

    # ── Merge request history + stored session history ─────────────────────
    stored_history = _load_history(session_id)

    # Request-level history takes precedence over stored (explicit override)
    if request.chat_history:
        chat_history_dicts = schema_history_to_dicts(request.chat_history)
    else:
        # Use stored session history for seamless memory
        chat_history_dicts = [
            {"role": h["role"], "content": h["content"]}
            for h in stored_history
        ]

    # ── Run pipeline ───────────────────────────────────────────────────────
    try:
        result = run_rag_pipeline(
            question=request.question,
            chat_history=chat_history_dicts,
            document_id=request.document_id,
            top_k=request.top_k,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )

    # ── Persist Q&A to session memory ─────────────────────────────────────
    _append_to_history(
        session_id=session_id,
        question=request.question,
        answer=result.answer,
    )

    return build_query_response(request, result)


@router.post(
    "/query/stream",
    summary="Ask a question — streaming response (SSE)",
    response_class=StreamingResponse,
)
async def query_documents_stream(
    request: QueryRequest,
    session_id: str = Query(default="default"),
):
    """
    Streaming version of the RAG query endpoint.
    Returns Server-Sent Events (SSE) — tokens arrive as they are generated.

    Frontend usage:
        const evtSource = new EventSource('/api/v1/query/stream?session_id=abc');
        evtSource.onmessage = (e) => console.log(JSON.parse(e.data));

    Event types:
        { "type": "token",  "data": "..." }   — LLM token
        { "type": "sources","data": [...] }   — source chunks (end)
        { "type": "done",   "data": "" }      — stream finished
        { "type": "error",  "data": "..." }   — error message
    """
    from app.services.llm import get_llm, build_context_string, rewrite_question_for_context
    from app.services.vector_store import get_vector_store
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    from app.services.llm import RAG_SYSTEM_PROMPT

    stored_history = _load_history(session_id)
    if request.chat_history:
        chat_history_dicts = schema_history_to_dicts(request.chat_history)
    else:
        chat_history_dicts = [
            {"role": h["role"], "content": h["content"]}
            for h in stored_history
        ]

    top_k = request.top_k or settings.top_k_results

    async def event_generator():
        full_answer = ""
        try:
            # Rewrite + retrieve (same as non-streaming)
            question_rewritten = rewrite_question_for_context(
                request.question, chat_history_dicts
            )

            store = get_vector_store()
            search_results = store.search(
                query=question_rewritten,
                top_k=top_k,
                document_id=request.document_id,
            )

            chunks_for_context = [
                {
                    "content": r.content,
                    "filename": r.filename,
                    "page": r.page,
                    "score": r.score,
                    "document_id": r.document_id,
                }
                for r in search_results
            ]
            context_string = build_context_string(chunks_for_context)

            # Build messages
            llm = get_llm()
            messages = [
                SystemMessage(
                    content=RAG_SYSTEM_PROMPT.format(context=context_string)
                )
            ]
            for turn in chat_history_dicts:
                if turn["role"] == "user":
                    messages.append(HumanMessage(content=turn["content"]))
                elif turn["role"] == "assistant":
                    messages.append(AIMessage(content=turn["content"]))
            messages.append(HumanMessage(content=question_rewritten))

            # Stream tokens
            async for chunk in llm.astream(messages):
                token = chunk.content
                if token:
                    full_answer += token
                    payload = json.dumps({"type": "token", "data": token})
                    yield f"data: {payload}\n\n"

            # Send sources after streaming completes
            sources = [
                {
                    "content":     r.content[:300],
                    "document_id": r.document_id,
                    "filename":    r.filename,
                    "page":        r.page,
                    "score":       round(r.score, 4),
                }
                for r in search_results
            ]
            yield f"data: {json.dumps({'type': 'sources', 'data': sources})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'data': ''})}\n\n"

            # Persist to memory
            _append_to_history(
                session_id=session_id,
                question=request.question,
                answer=full_answer,
            )

        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


@router.get(
    "/query/search",
    summary="Raw vector similarity search (debug)",
)
async def raw_search(
    q: str = Query(..., description="Search query string"),
    top_k: int = Query(default=5, ge=1, le=20),
    document_id: str | None = Query(default=None),
):
    """
    Perform a raw vector similarity search without calling the LLM.
    Useful for debugging retrieval quality before involving the LLM.

    Returns the top-k most relevant chunks with their similarity scores.
    """
    if not q.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string 'q' cannot be empty.",
        )

    try:
        store = get_vector_store()
        results = store.search(
            query=q,
            top_k=top_k,
            document_id=document_id,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}",
        )

    return {
        "query": q,
        "top_k": top_k,
        "document_id": document_id,
        "num_results": len(results),
        "results": [
            {
                "chunk_id":    r.chunk_id,
                "document_id": r.document_id,
                "filename":    r.filename,
                "page":        r.page,
                "score":       round(r.score, 4),
                "content":     r.content[:500],
            }
            for r in results
        ],
    }


@router.get(
    "/query/history",
    summary="Retrieve chat history for a session",
)
async def get_chat_history(
    session_id: str = Query(default="default"),
    limit: int = Query(default=20, ge=1, le=100),
):
    """
    Return the stored chat history for a given session.
    History is ordered oldest → newest.
    """
    history = _load_history(session_id)
    trimmed = history[-limit:] if len(history) > limit else history

    return {
        "session_id":   session_id,
        "total_turns":  len(history) // 2,
        "returned":     len(trimmed),
        "messages":     trimmed,
    }


@router.delete(
    "/query/history",
    summary="Clear chat history for a session",
)
async def clear_chat_history(
    session_id: str = Query(default="default"),
):
    """
    Delete all stored messages for a session.
    The next query will start a fresh conversation.
    """
    path = _history_path(session_id)
    if path.exists():
        path.unlink()
        logger.info(f"Cleared history for session: {session_id}")

    return {
        "success":    True,
        "session_id": session_id,
        "message":    f"Chat history cleared for session '{session_id}'.",
    }