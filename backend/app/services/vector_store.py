"""
Vector Store Service.

Supports two backends controlled by VECTOR_DB_TYPE in .env:
  - "faiss"  → Facebook AI Similarity Search (local, fast, no server needed)
  - "chroma" → ChromaDB          (local, persistent, metadata-rich)

Public interface (same for both backends):
  - add_chunks()      → embed + index a list of TextChunks
  - search()          → top-k similarity search by query string
  - delete_document() → remove all vectors for a document_id
  - get_stats()       → index statistics
"""

import json
import logging
import os
import pickle
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from app.config import get_settings
from app.services.embedding import embed_query, embed_texts, get_embedding_dimension
from app.utils.chunking import TextChunk

logger = logging.getLogger(__name__)
settings = get_settings()


# ── Result dataclass ───────────────────────────────────────────────────────────

class SearchResult:
    """A single result returned from a similarity search."""

    def __init__( #could have used @dataclass but this is Behavior-heavy class (logic + state) and cant used here, if we use then we can loose data control
        self,
        chunk_id: str,
        document_id: str,
        filename: str,
        content: str,
        page: int | None,
        score: float,
    ):
        self.chunk_id = chunk_id
        self.document_id = document_id
        self.filename = filename
        self.content = content
        self.page = page
        self.score = score          # Higher = more similar (0–1 for cosine)

    def __repr__(self):
        return (
            f"SearchResult(doc={self.document_id!r}, "
            f"page={self.page}, score={self.score:.4f}, "
            f"content={self.content[:60]!r}...)"
        )


# ── Abstract base ──────────────────────────────────────────────────────────────

class BaseVectorStore(ABC):
    """Interface every vector store backend must implement."""

    @abstractmethod
    def add_chunks(self, chunks: list[TextChunk]) -> int:
        """Embed and index chunks. Returns number of vectors added."""

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int,
        document_id: str | None = None,
    ) -> list[SearchResult]:
        """Return top-k most similar chunks for the given query."""

    @abstractmethod
    def delete_document(self, document_id: str) -> int:
        """Remove all vectors for a document. Returns count deleted."""

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """Return store statistics (index size, num docs, etc.)."""


# ══════════════════════════════════════════════════════════════════════════════
# FAISS BACKEND
# ══════════════════════════════════════════════════════════════════════════════

class FAISSVectorStore(BaseVectorStore):
    """
    FAISS-based vector store.

    Persistence layout (under data/vector_db/):
      faiss_index.bin     — the raw FAISS index
      faiss_metadata.pkl  — list of dicts, one per vector (parallel array)

    The metadata list is a parallel array to the FAISS index:
      metadata[i] contains the chunk info for the i-th FAISS vector.

    When documents are deleted, vectors are marked as deleted in metadata
    and filtered out at query time (soft-delete).
    A rebuild is triggered when the deleted ratio exceeds 30%.
    """

    INDEX_FILE = "faiss_index.bin"
    META_FILE = "faiss_metadata.pkl"
    DELETED_REBUILD_THRESHOLD = 0.30   # rebuild if >30% vectors are deleted

    def __init__(self):
        try:
            import faiss
            self._faiss = faiss
        except ImportError:
            raise ImportError(
                "faiss-cpu is not installed. Run: pip install faiss-cpu"
            )

        self._db_dir = Path(settings.vector_db_dir)
        self._db_dir.mkdir(parents=True, exist_ok=True)

        self._dim = get_embedding_dimension()
        self._index: Any = None         # faiss.IndexFlatIP
        self._metadata: list[dict] = [] # parallel array

        self._load_or_create()

    # ── Persistence ────────────────────────────────────────────────────────

    def _load_or_create(self):
        index_path = self._db_dir / self.INDEX_FILE
        meta_path = self._db_dir / self.META_FILE

        if index_path.exists() and meta_path.exists():
            logger.info("Loading existing FAISS index from disk ...")
            self._index = self._faiss.read_index(str(index_path))
            with open(meta_path, "rb") as f:
                self._metadata = pickle.load(f)
            logger.info(
                f"FAISS index loaded: {self._index.ntotal} vectors, "
                f"{len(self._metadata)} metadata entries."
            )
        else:
            logger.info(f"Creating new FAISS IndexFlatIP (dim={self._dim})")
            # IndexFlatIP = exact inner product (= cosine sim when L2-normalized)
            self._index = self._faiss.IndexFlatIP(self._dim)
            self._metadata = []

    def _save(self):
        self._faiss.write_index(
            self._index, str(self._db_dir / self.INDEX_FILE)
        )
        with open(self._db_dir / self.META_FILE, "wb") as f:
            pickle.dump(self._metadata, f)

    def _rebuild_index(self):
        """
        Physically remove soft-deleted vectors by rebuilding the index
        from scratch using only the active metadata entries.
        """
        logger.info("Rebuilding FAISS index to purge deleted vectors ...")
        active = [m for m in self._metadata if not m.get("deleted", False)]

        new_index = self._faiss.IndexFlatIP(self._dim)
        if active:
            # Re-embed all active chunks
            texts = [m["content"] for m in active]
            vectors = embed_texts(texts)
            new_index.add(vectors)

        self._index = new_index
        self._metadata = active
        self._save()
        logger.info(f"Rebuild complete. Active vectors: {len(active)}")

    # ── Public interface ───────────────────────────────────────────────────

    def add_chunks(self, chunks: list[TextChunk]) -> int:
        if not chunks:
            return 0

        texts = [c.content for c in chunks]
        vectors = embed_texts(texts)             # shape: (n, dim)

        self._index.add(vectors)

        for chunk in chunks:
            self._metadata.append({
                "chunk_id":    chunk.chunk_id,
                "document_id": chunk.document_id,
                "filename":    chunk.filename,
                "content":     chunk.content,
                "page":        chunk.page,
                "deleted":     False,
            })

        self._save()
        logger.info(
            f"FAISS: added {len(chunks)} vectors. "
            f"Total: {self._index.ntotal}"
        )
        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int,
        document_id: str | None = None,
    ) -> list[SearchResult]:

        if self._index.ntotal == 0:
            logger.warning("FAISS index is empty — no results.")
            return []

        query_vec = embed_query(query)           # shape: (1, dim)

        # Over-fetch to account for soft-deleted entries and doc filtering
        fetch_k = min(top_k * 10, self._index.ntotal)
        scores, indices = self._index.search(query_vec, fetch_k)

        results: list[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._metadata):
                continue

            meta = self._metadata[idx]
            if meta.get("deleted", False):
                continue
            if document_id and meta["document_id"] != document_id:
                continue

            results.append(SearchResult(
                chunk_id=meta["chunk_id"],
                document_id=meta["document_id"],
                filename=meta["filename"],
                content=meta["content"],
                page=meta.get("page"),
                score=float(score),
            ))

            if len(results) >= top_k:
                break

        return results

    def delete_document(self, document_id: str) -> int:
        count = 0
        for meta in self._metadata:
            if meta["document_id"] == document_id and not meta.get("deleted"):
                meta["deleted"] = True
                count += 1

        if count:
            self._save()
            logger.info(
                f"FAISS: soft-deleted {count} vectors for doc '{document_id}'"
            )

            # Rebuild index if deleted ratio is too high
            total = len(self._metadata)
            deleted = sum(1 for m in self._metadata if m.get("deleted", False))
            if total > 0 and deleted / total > self.DELETED_REBUILD_THRESHOLD:
                self._rebuild_index()

        return count

    def get_stats(self) -> dict[str, Any]:
        active = sum(
            1 for m in self._metadata if not m.get("deleted", False)
        )
        doc_ids = {
            m["document_id"]
            for m in self._metadata
            if not m.get("deleted", False)
        }
        return {
            "backend":        "faiss",
            "total_vectors":  self._index.ntotal,
            "active_vectors": active,
            "num_documents":  len(doc_ids),
            "dimension":      self._dim,
        }


# ══════════════════════════════════════════════════════════════════════════════
# CHROMA BACKEND
# ══════════════════════════════════════════════════════════════════════════════

class ChromaVectorStore(BaseVectorStore):
    """
    ChromaDB-based vector store.

    Uses ChromaDB's persistent client — data is saved to data/vector_db/.
    ChromaDB handles its own embedding metadata natively.
    We pass pre-computed embeddings so our HuggingFace model is always used.
    """

    COLLECTION_NAME = "research_assistant"

    def __init__(self):
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError:
            raise ImportError(
                "chromadb is not installed. Run: pip install chromadb"
            )

        db_path = str(Path(settings.vector_db_dir) / "chroma")
        logger.info(f"Initializing ChromaDB at: {db_path}")

        client = chromadb.PersistentClient(
            path=db_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        self._collection = client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"ChromaDB collection '{self.COLLECTION_NAME}' ready — "
            f"{self._collection.count()} vectors."
        )

    def add_chunks(self, chunks: list[TextChunk]) -> int:
        if not chunks:
            return 0

        texts = [c.content for c in chunks]
        vectors = embed_texts(texts)

        self._collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=vectors.tolist(),
            documents=texts,
            metadatas=[
                {
                    "document_id": c.document_id,
                    "filename":    c.filename,
                    "page":        c.page or 0,
                    "chunk_index": c.chunk_index,
                }
                for c in chunks
            ],
        )

        logger.info(
            f"ChromaDB: added {len(chunks)} vectors. "
            f"Total: {self._collection.count()}"
        )
        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int,
        document_id: str | None = None,
    ) -> list[SearchResult]:

        if self._collection.count() == 0:
            logger.warning("ChromaDB collection is empty — no results.")
            return []

        query_vec = embed_query(query)

        where_filter = {"document_id": document_id} if document_id else None

        results = self._collection.query(
            query_embeddings=query_vec.tolist(),
            n_results=min(top_k, self._collection.count()),
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        search_results: list[SearchResult] = []
        ids       = results["ids"][0]
        docs      = results["documents"][0]
        metas     = results["metadatas"][0]
        distances = results["distances"][0]

        for chunk_id, content, meta, dist in zip(ids, docs, metas, distances):
            # ChromaDB cosine distance → similarity: 1 - distance
            score = 1.0 - dist
            search_results.append(SearchResult(
                chunk_id=chunk_id,
                document_id=meta["document_id"],
                filename=meta["filename"],
                content=content,
                page=meta.get("page") or None,
                score=score,
            ))

        return search_results

    def delete_document(self, document_id: str) -> int:
        existing = self._collection.get(
            where={"document_id": document_id},
            include=[],
        )
        ids_to_delete = existing["ids"]

        if ids_to_delete:
            self._collection.delete(ids=ids_to_delete)
            logger.info(
                f"ChromaDB: deleted {len(ids_to_delete)} vectors "
                f"for doc '{document_id}'"
            )

        return len(ids_to_delete)

    def get_stats(self) -> dict[str, Any]:
        return {
            "backend":        "chroma",
            "total_vectors":  self._collection.count(),
            "collection":     self.COLLECTION_NAME,
            "dimension":      get_embedding_dimension(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# FACTORY — single entry point for the rest of the app
# ══════════════════════════════════════════════════════════════════════════════

_store_instance: BaseVectorStore | None = None


def get_vector_store() -> BaseVectorStore:
    """
    Return the singleton vector store instance.
    Backend is chosen by VECTOR_DB_TYPE in .env ("faiss" | "chroma").

    All services import this function — never instantiate directly.
    """
    global _store_instance

    if _store_instance is None:
        backend = settings.vector_db_type.lower().strip()
        logger.info(f"Initializing vector store backend: '{backend}'")

        if backend == "faiss":
            _store_instance = FAISSVectorStore()
        elif backend == "chroma":
            _store_instance = ChromaVectorStore()
        else:
            raise ValueError(
                f"Unknown VECTOR_DB_TYPE='{backend}'. "
                "Choose 'faiss' or 'chroma' in your .env file."
            )

    return _store_instance


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: embed and index chunks from a saved JSON sidecar
# ══════════════════════════════════════════════════════════════════════════════

def index_document_chunks(document_id: str) -> int:
    """
    Read the chunks JSON sidecar saved during upload (Step 2)
    and add all chunks to the vector store.

    Called automatically from the upload route after saving chunks.

    Args:
        document_id: The document whose chunks JSON sidecar to index.

    Returns:
        Number of vectors indexed.

    Raises:
        FileNotFoundError: If the chunks sidecar doesn't exist.
    """
    chunks_path = Path(settings.docs_dir) / f"{document_id}_chunks.json"

    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {chunks_path}. "
            "Was the PDF uploaded and processed first?"
        )

    raw = json.loads(chunks_path.read_text(encoding="utf-8"))

    chunks = [
        TextChunk(
            chunk_id=c["chunk_id"],
            document_id=c["document_id"],
            filename=c["filename"],
            content=c["content"],
            page=c.get("page"),
            chunk_index=c["chunk_index"],
            total_chunks=c["total_chunks"],
        )
        for c in raw
    ]

    start = time.perf_counter()
    store = get_vector_store()
    n = store.add_chunks(chunks)
    elapsed = time.perf_counter() - start

    logger.info(
        f"Indexed {n} chunks for document '{document_id}' "
        f"in {elapsed:.3f}s"
    )
    return n