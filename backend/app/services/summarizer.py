"""
Summarization Service.

Strategy:
  - For SHORT documents  (< 3000 chars) : single-pass LLM summarization
  - For LONG  documents  (≥ 3000 chars) : Map-Reduce summarization
      1. MAP    — summarize each chunk independently
      2. REDUCE — combine chunk summaries into one final summary

This avoids hitting the LLM context window limit on large PDFs.

All summarization is done via the Groq LLM (same client as RAG pipeline).
No extra models are needed.
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from app.config import get_settings
from app.services.llm import get_llm

logger = logging.getLogger(__name__)
settings = get_settings()

# ── Thresholds ─────────────────────────────────────────────────────────────────
SHORT_DOC_THRESHOLD = 3000       # characters — use single pass below this
MAX_CHARS_PER_MAP_CHUNK = 2500   # max chars sent to LLM per map step
MAX_CHUNKS_TO_MAP = 20           # safety cap — avoid huge API bills


# ── Prompts ────────────────────────────────────────────────────────────────────

SINGLE_PASS_SYSTEM = """You are an expert document summarizer.
Your task is to produce a clear, structured summary of the provided document.

RULES:
1. Capture the main ideas, key findings, and important details.
2. Organize the summary with these sections:
   - **Overview**: 2-3 sentence high-level description.
   - **Key Points**: Bullet list of the most important facts/findings.
   - **Conclusion**: 1-2 sentences on the document's significance or outcome.
3. Be concise — target {max_words} words total.
4. Do NOT add information not present in the document.
"""

MAP_SYSTEM = """You are summarizing one section of a larger document.
Produce a concise paragraph (3-5 sentences) capturing the key ideas in this section.
Do NOT add any headings. Only output the paragraph text.
"""

REDUCE_SYSTEM = """You are combining multiple section summaries into one \
cohesive final summary of an entire document.

RULES:
1. Remove redundancy — do not repeat the same point twice.
2. Preserve all unique key ideas from all section summaries.
3. Organize the output with:
   - **Overview**: 2-3 sentence high-level description.
   - **Key Points**: Bullet list of the most important facts/findings.
   - **Conclusion**: 1-2 sentences on the document's significance or outcome.
4. Target {max_words} words total.
"""


# ── Result Dataclass ───────────────────────────────────────────────────────────

@dataclass
class SummaryResult:
    document_id: str
    filename: str
    summary: str
    strategy: str          # "single_pass" | "map_reduce"
    num_chunks_used: int
    processing_time: float


# ── Internal Helpers ───────────────────────────────────────────────────────────

def _load_chunks_text(document_id: str) -> tuple[list[str], str]:
    """
    Load all chunk texts for a document from the JSON sidecar.

    Returns:
        (list_of_chunk_texts, filename)

    Raises:
        FileNotFoundError: If the chunks sidecar does not exist.
    """
    chunks_path = Path(settings.docs_dir) / f"{document_id}_chunks.json"

    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Chunks not found for document '{document_id}'. "
            "Was the PDF uploaded and processed?"
        )

    raw = json.loads(chunks_path.read_text(encoding="utf-8"))
    texts = [c["content"] for c in raw if c.get("content", "").strip()]
    filename = raw[0]["filename"] if raw else "unknown"
    return texts, filename


def _call_llm(system_prompt: str, user_content: str) -> str:
    """Single LLM call — returns response text."""
    llm = get_llm(temperature=0.3)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ]
    response = llm.invoke(messages)
    return response.content.strip()


def _single_pass_summarize(
    text: str,
    max_words: int,
) -> str:
    """
    Summarize a short document in a single LLM call.

    Args:
        text:      Full document text (or a truncated version).
        max_words: Approximate target word count for the summary.

    Returns:
        Summary string.
    """
    logger.info("Using single-pass summarization strategy.")
    system = SINGLE_PASS_SYSTEM.format(max_words=max_words)
    user = f"Please summarize this document:\n\n{text}"
    return _call_llm(system, user)


def _map_reduce_summarize(
    chunk_texts: list[str],
    max_words: int,
) -> tuple[str, int]:
    """
    Summarize a long document using Map-Reduce strategy.

    MAP phase  : summarize each chunk independently.
    REDUCE phase: merge all chunk summaries into the final summary.

    Args:
        chunk_texts: List of chunk content strings.
        max_words:   Target word count for the final summary.

    Returns:
        (final_summary, num_chunks_processed)
    """
    logger.info(
        f"Using map-reduce summarization | "
        f"total_chunks={len(chunk_texts)}"
    )

    # ── MAP phase ──────────────────────────────────────────────────────────
    # Group small chunks together to reduce the number of LLM calls
    grouped_chunks = _group_chunks(chunk_texts, MAX_CHARS_PER_MAP_CHUNK)

    # Cap at MAX_CHUNKS_TO_MAP to control costs
    groups_to_process = grouped_chunks[:MAX_CHUNKS_TO_MAP]
    num_skipped = len(grouped_chunks) - len(groups_to_process)

    if num_skipped > 0:
        logger.warning(
            f"Document is very large — processing first "
            f"{MAX_CHUNKS_TO_MAP} chunk groups, "
            f"skipping {num_skipped}."
        )

    chunk_summaries: list[str] = []
    for i, group_text in enumerate(groups_to_process, 1):
        logger.info(
            f"  MAP step {i}/{len(groups_to_process)} "
            f"({len(group_text)} chars) ..."
        )
        try:
            summary = _call_llm(
                MAP_SYSTEM,
                f"Summarize this section:\n\n{group_text}",
            )
            chunk_summaries.append(summary)
        except Exception as e:
            logger.warning(f"  MAP step {i} failed: {e}. Skipping.")

    if not chunk_summaries:
        raise RuntimeError(
            "All MAP steps failed. Cannot produce summary."
        )

    # ── REDUCE phase ───────────────────────────────────────────────────────
    logger.info(
        f"REDUCE phase: combining {len(chunk_summaries)} section summaries ..."
    )
    combined = "\n\n---\n\n".join(
        f"Section {i}:\n{s}"
        for i, s in enumerate(chunk_summaries, 1)
    )
    system = REDUCE_SYSTEM.format(max_words=max_words)
    user = (
        f"Combine these section summaries into one final document summary:\n\n"
        f"{combined}"
    )
    final_summary = _call_llm(system, user)

    return final_summary, len(groups_to_process)


def _group_chunks(
    chunk_texts: list[str],
    max_chars: int,
) -> list[str]:
    """
    Greedily group small chunks together so each group is ≤ max_chars.
    Reduces the number of LLM calls in the MAP phase.
    """
    groups: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    for text in chunk_texts:
        if current_len + len(text) > max_chars and current_parts:
            groups.append("\n\n".join(current_parts))
            current_parts = [text]
            current_len = len(text)
        else:
            current_parts.append(text)
            current_len += len(text)

    if current_parts:
        groups.append("\n\n".join(current_parts))

    return groups


# ── Public API ─────────────────────────────────────────────────────────────────

def summarize_document(
    document_id: str,
    max_words: int = 500,
) -> SummaryResult:
    """
    Summarize a previously uploaded document.

    Automatically chooses between single-pass and map-reduce
    based on the total character count of all chunks.

    Args:
        document_id: The document's ID (returned from upload).
        max_words:   Approximate max words in the final summary.

    Returns:
        SummaryResult with summary text and metadata.

    Raises:
        FileNotFoundError: If the document has not been uploaded.
        RuntimeError:      If the LLM fails to generate a summary.
    """
    start = time.perf_counter()

    chunk_texts, filename = _load_chunks_text(document_id)

    if not chunk_texts:
        raise RuntimeError(
            f"Document '{document_id}' has no extractable text to summarize."
        )

    full_text = "\n\n".join(chunk_texts)
    total_chars = len(full_text)

    logger.info(
        f"Summarizing '{filename}' | "
        f"document_id={document_id} | "
        f"total_chars={total_chars:,} | "
        f"num_chunks={len(chunk_texts)} | "
        f"max_words={max_words}"
    )

    # ── Choose strategy ────────────────────────────────────────────────────
    if total_chars < SHORT_DOC_THRESHOLD:
        summary = _single_pass_summarize(full_text, max_words)
        strategy = "single_pass"
        num_chunks_used = len(chunk_texts)

    else:
        summary, num_chunks_used = _map_reduce_summarize(
            chunk_texts, max_words
        )
        strategy = "map_reduce"

    processing_time = time.perf_counter() - start

    logger.info(
        f"Summarization complete | "
        f"strategy={strategy} | "
        f"time={processing_time:.3f}s | "
        f"summary_chars={len(summary)}"
    )

    return SummaryResult(
        document_id=document_id,
        filename=filename,
        summary=summary,
        strategy=strategy,
        num_chunks_used=num_chunks_used,
        processing_time=processing_time,
    )