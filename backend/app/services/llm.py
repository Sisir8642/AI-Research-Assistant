"""
LLM Service — Groq API Integration.

Responsibilities:
  - Initialize and cache the Groq LLM client (via LangChain)
  - Format prompts correctly for RAG (system + context + question)
  - Handle streaming and non-streaming responses
  - Provide a clean generate() interface to the rest of the app

Groq provides free-tier access to:
  - llama3-8b-8192    (fast, lightweight)
  - llama3-70b-8192   (powerful, slower)
  - mixtral-8x7b-32768 (large context window)

Set GROQ_MODEL in .env to switch models.
"""

import logging
import time
from functools import lru_cache
from typing import AsyncIterator

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ── Prompt Templates ───────────────────────────────────────────────────────────

RAG_SYSTEM_PROMPT = """You are an expert AI Research Assistant. \
Your job is to answer questions accurately using ONLY the context \
retrieved from the user's uploaded documents.

STRICT RULES:
1. Answer ONLY from the provided context. Do NOT use outside knowledge.
2. If the context does not contain enough information, say:
   "I could not find sufficient information in the uploaded documents \
to answer this question."
3. Always be concise, structured, and factually precise.
4. When relevant, mention which part of the document supports your answer.
5. If the question is a follow-up, use the conversation history to \
maintain continuity.

Context from documents:
─────────────────────────────────────────────
{context}
─────────────────────────────────────────────
"""

STANDALONE_QUESTION_PROMPT = """Given the conversation history and a \
follow-up question, rewrite the follow-up question to be a standalone \
question that captures all necessary context.

Conversation history:
{chat_history}

Follow-up question: {question}

Standalone question:"""


# ── LLM Client Factory ─────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_llm(temperature: float = 0.2) -> ChatGroq:
    """
    Return a cached ChatGroq instance.

    temperature=0.2 keeps answers factual and consistent.
    Raise to 0.7+ for more creative/varied responses.

    Raises:
        ValueError: If GROQ_API_KEY is missing from .env
    """
    if not settings.groq_api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. "
            "Get a free key at https://console.groq.com and add it to .env"
        )

    logger.info(f"Initializing Groq LLM: model='{settings.groq_model}'")

    llm = ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        temperature=temperature,
        max_tokens=2048,
        timeout=60,
        max_retries=2,
    )

    logger.info("Groq LLM client ready.")
    return llm


# ── Core Generation Functions ──────────────────────────────────────────────────

def generate_answer(
    question: str,
    context: str,
    chat_history: list[dict] | None = None,
) -> tuple[str, float]:
    """
    Generate a RAG answer using the Groq LLM.

    Args:
        question:     The user's (possibly rewritten) question.
        context:      Concatenated retrieved chunks (the RAG context).
        chat_history: List of {"role": "user"|"assistant", "content": "..."}
                      dicts for conversational memory.

    Returns:
        Tuple of (answer_text, elapsed_seconds).

    Raises:
        RuntimeError: On LLM API failure after retries.
    """
    chat_history = chat_history or []
    llm = get_llm()

    # ── Build message list ─────────────────────────────────────────────────
    messages = [
        SystemMessage(content=RAG_SYSTEM_PROMPT.format(context=context))
    ]

    # Inject prior conversation turns (short-term memory)
    for turn in chat_history:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    # Current question
    messages.append(HumanMessage(content=question))

    # ── Call LLM ───────────────────────────────────────────────────────────
    logger.info(
        f"Calling Groq API | model={settings.groq_model} | "
        f"history_turns={len(chat_history)} | "
        f"context_chars={len(context)}"
    )

    start = time.perf_counter()
    try:
        response = llm.invoke(messages)
        elapsed = time.perf_counter() - start
        answer = response.content.strip()

        logger.info(
            f"Groq response received in {elapsed:.3f}s | "
            f"answer_chars={len(answer)}"
        )
        return answer, elapsed

    except Exception as e:
        elapsed = time.perf_counter() - start
        logger.error(f"Groq API call failed after {elapsed:.3f}s: {e}")
        raise RuntimeError(
            f"LLM generation failed: {str(e)}. "
            "Check your GROQ_API_KEY and network connection."
        ) from e


def rewrite_question_for_context(
    question: str,
    chat_history: list[dict],
) -> str:
    """
    If there is chat history, rewrite the user's question into a
    self-contained standalone question so retrieval works correctly.

    Example:
      history: "What is BERT?"  →  "BERT is a transformer model..."
      follow-up: "How does it handle tokenization?"
      rewritten: "How does BERT handle tokenization?"

    If no history, returns the original question unchanged.
    """
    if not chat_history:
        return question

    llm = get_llm(temperature=0.0)  # deterministic rewriting

    history_text = "\n".join(
        f"{turn['role'].capitalize()}: {turn['content']}"
        for turn in chat_history[-6:]  # last 3 turns (6 messages)
    )

    prompt = STANDALONE_QUESTION_PROMPT.format(
        chat_history=history_text,
        question=question,
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        rewritten = response.content.strip()
        if rewritten and rewritten != question:
            logger.info(
                f"Question rewritten for retrieval:\n"
                f"  Original : {question}\n"
                f"  Rewritten: {rewritten}"
            )
        return rewritten
    except Exception as e:
        logger.warning(f"Question rewriting failed ({e}), using original.")
        return question


def build_context_string(chunks_with_scores: list[dict]) -> str:
    """
    Format retrieved chunks into the context block injected into the prompt.

    Each chunk is labeled with its source file and page number so the LLM
    can cite sources in its answer.

    Args:
        chunks_with_scores: List of dicts with keys:
            content, filename, page, score, document_id

    Returns:
        Formatted multi-line string ready for the system prompt.
    """
    if not chunks_with_scores:
        return "No relevant context found in the uploaded documents."

    parts: list[str] = []
    for i, chunk in enumerate(chunks_with_scores, start=1):
        filename = chunk.get("filename", "Unknown")
        page = chunk.get("page")
        content = chunk.get("content", "").strip()
        score = chunk.get("score", 0.0)

        page_str = f", page {page}" if page else ""
        header = f"[Source {i}: {filename}{page_str} | relevance: {score:.2f}]"
        parts.append(f"{header}\n{content}")

    return "\n\n".join(parts)