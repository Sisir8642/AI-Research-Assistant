"""
Pydantic schemas for all API request/response models.
These are the data contracts between frontend and backend.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# ── Upload Schemas ─────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    """Response returned after a successful PDF upload and processing."""
    success: bool
    message: str
    document_id: str
    filename: str
    num_chunks: int
    processing_time_seconds: float


# ── Query / Chat Schemas ───────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    """A single message in the conversation."""
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str
    timestamp: Optional[datetime] = None


class QueryRequest(BaseModel):
    """Request payload when the user asks a question."""
    question: str = Field(..., min_length=1, max_length=2000)
    document_id: Optional[str] = Field(
        default=None,
        description="Limit retrieval to a specific document. "
                    "If None, search across all documents.",
    )
    chat_history: list[ChatMessage] = Field(
        default_factory=list,
        description="Previous messages for context-aware answers.",
    )
    top_k: Optional[int] = Field(
        default=None,
        description="Override the default top-k retrieval count.",
    )


class SourceChunk(BaseModel):
    """A retrieved document chunk used as context for the answer."""
    content: str
    document_id: str
    filename: str
    page: Optional[int] = None
    score: Optional[float] = None


class QueryResponse(BaseModel):
    """Full response to a user question."""
    answer: str
    sources: list[SourceChunk]
    question: str
    model_used: str
    processing_time_seconds: float


# ── Summarization Schemas ──────────────────────────────────────────────────────

class SummarizeRequest(BaseModel):
    document_id: str
    max_length: Optional[int] = Field(
        default=500,
        description="Approximate max word count of the summary.",
    )


class SummarizeResponse(BaseModel):
    document_id: str
    filename: str
    summary: str
    processing_time_seconds: float


# ── Classification Schemas ─────────────────────────────────────────────────────

class ClassifyRequest(BaseModel):
    document_id: str


class ClassifyResponse(BaseModel):
    document_id: str
    filename: str
    predicted_category: str
    confidence: float
    all_scores: dict[str, float]
    processing_time_seconds: float


# ── Document Listing ───────────────────────────────────────────────────────────

class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    num_chunks: int
    uploaded_at: datetime


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]
    total: int