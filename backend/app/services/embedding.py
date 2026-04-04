"""
Embedding Service.

Loads a HuggingFace sentence-transformer model once (singleton pattern)
and exposes two functions:
  - embed_texts()   → list of vectors (for indexing chunks)
  - embed_query()   → single vector  (for querying)

Model default: sentence-transformers/all-MiniLM-L6-v2
  - 384-dimensional vectors
  - Fast, lightweight, strong semantic similarity
  - Runs fully on CPU — no GPU required
"""

import logging
import time
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ── Singleton model loader ─────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """
    Load the embedding model exactly once for the lifetime of the process.
    lru_cache ensures subsequent calls return the cached instance.

    Downloads model from HuggingFace Hub on first run (~90 MB).
    Cached locally in ~/.cache/huggingface/ afterwards.
    """
    model_name = settings.embedding_model
    logger.info(f"Loading embedding model: '{model_name}' ...")

    start = time.perf_counter()
    model = SentenceTransformer(model_name)
    elapsed = time.perf_counter() - start

    dim = model.get_sentence_embedding_dimension()
    logger.info(
        f"Embedding model loaded in {elapsed:.2f}s — "
        f"vector dimension: {dim}"
    )
    return model


# ── Public API ─────────────────────────────────────────────────────────────────

def embed_texts(
    texts: list[str],
    batch_size: int = 32,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Embed a list of text strings into dense vectors.

    Used when indexing document chunks into the vector store.

    Args:
        texts:         List of strings to embed.
        batch_size:    Number of texts processed per forward pass.
                       Increase for faster indexing if RAM allows.
        show_progress: Print a tqdm progress bar (useful for large batches).

    Returns:
        np.ndarray of shape (len(texts), embedding_dim), dtype float32.

    Raises:
        ValueError: If texts list is empty.
    """
    if not texts:
        raise ValueError("embed_texts() received an empty list.")

    model = get_embedding_model()

    logger.info(f"Embedding {len(texts)} texts (batch_size={batch_size}) ...")
    start = time.perf_counter()

    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalize → cosine sim = dot product
    )

    elapsed = time.perf_counter() - start
    logger.info(
        f"Embedded {len(texts)} texts → shape {vectors.shape} "
        f"in {elapsed:.3f}s ({len(texts)/elapsed:.1f} texts/sec)"
    )

    return vectors.astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query string.

    Used at retrieval time — wraps embed_texts() for convenience.

    Args:
        query: The user's question or search string.

    Returns:
        np.ndarray of shape (1, embedding_dim), dtype float32.

    Raises:
        ValueError: If query is empty or whitespace-only.
    """
    query = query.strip()
    if not query:
        raise ValueError("embed_query() received an empty query string.")

    vector = embed_texts([query], batch_size=1)
    return vector   # shape: (1, dim)


def get_embedding_dimension() -> int:
    """Return the vector dimension of the loaded model."""
    return get_embedding_model().get_sentence_embedding_dimension()