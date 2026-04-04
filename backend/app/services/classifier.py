"""
Document Classification Service.

Two-layer classification approach:
  Layer 1 — Zero-Shot LLM Classification (primary)
      Ask the Groq LLM to classify the document against a
      predefined taxonomy. Requires no training data.

  Layer 2 — TF-IDF + Cosine Similarity (fallback / confidence boost)
      If the LLM returns low confidence or fails, use a lightweight
      keyword-based TF-IDF classifier as a fallback.
      Also used to validate LLM output.

Categories (extensible via CATEGORIES dict below):
  - Academic / Research
  - Legal / Contracts
  - Financial / Business
  - Medical / Healthcare
  - Technical / Engineering
  - News / Journalism
  - Educational / Textbook
  - General / Other
"""

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.config import get_settings
from app.services.llm import get_llm

logger = logging.getLogger(__name__)
settings = get_settings()


# ── Category Taxonomy ──────────────────────────────────────────────────────────
# Each category maps to a list of representative keywords.
# Used for the TF-IDF fallback classifier.

CATEGORIES: dict[str, list[str]] = {
    "Academic / Research": [
        "abstract", "methodology", "hypothesis", "experiment", "findings",
        "literature review", "citation", "peer review", "dataset", "analysis",
        "conclusion", "journal", "study", "research", "theorem", "proof",
        "survey", "benchmark", "evaluation", "results",
    ],
    "Legal / Contracts": [
        "agreement", "clause", "liability", "indemnity", "jurisdiction",
        "plaintiff", "defendant", "court", "attorney", "contract",
        "terms and conditions", "warranty", "intellectual property",
        "confidentiality", "breach", "legal", "statute", "regulation",
        "arbitration", "amendment",
    ],
    "Financial / Business": [
        "revenue", "profit", "loss", "balance sheet", "cash flow",
        "investment", "shareholder", "quarterly", "fiscal year", "earnings",
        "budget", "forecast", "equity", "assets", "liabilities", "audit",
        "dividend", "market", "portfolio", "financial statement",
    ],
    "Medical / Healthcare": [
        "patient", "diagnosis", "treatment", "clinical", "symptoms",
        "medication", "therapy", "hospital", "physician", "disease",
        "healthcare", "surgery", "trial", "dosage", "prescription",
        "pathology", "prognosis", "epidemiology", "vaccine", "anatomy",
    ],
    "Technical / Engineering": [
        "algorithm", "architecture", "framework", "implementation",
        "system design", "api", "database", "infrastructure", "deployment",
        "software", "hardware", "protocol", "network", "optimization",
        "pipeline", "module", "specification", "debugging", "code", "server",
    ],
    "News / Journalism": [
        "reported", "according to", "spokesperson", "announced", "press release",
        "interview", "editor", "publication", "headline", "breaking",
        "correspondent", "journalist", "statement", "coverage", "source",
        "government", "election", "policy", "incident", "update",
    ],
    "Educational / Textbook": [
        "chapter", "lesson", "exercise", "definition", "example",
        "introduction", "summary", "objective", "quiz", "curriculum",
        "student", "teacher", "learning", "course", "textbook",
        "concept", "theory", "practice", "assignment", "lecture",
    ],
    "General / Other": [
        "information", "document", "content", "text", "page",
        "section", "part", "item", "detail", "note",
    ],
}

# ── Prompts ────────────────────────────────────────────────────────────────────

CLASSIFY_SYSTEM = """You are a document classification expert.
Your task is to classify a document into exactly ONE of the following categories:

{categories}

RULES:
1. Read the document excerpt carefully.
2. Choose the MOST appropriate category.
3. Respond ONLY with a valid JSON object — no markdown, no explanation.
4. Format:
{{
  "category": "<exact category name from the list>",
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "<one sentence explanation>"
}}
"""

CLASSIFY_USER = """Classify this document excerpt:

{text_excerpt}
"""


# ── Result Dataclass ───────────────────────────────────────────────────────────

@dataclass
class ClassificationResult:
    document_id: str
    filename: str
    predicted_category: str
    confidence: float
    all_scores: dict[str, float]
    method: str               # "llm" | "tfidf" | "llm+tfidf"
    reasoning: str
    processing_time: float


# ── TF-IDF Fallback Classifier ─────────────────────────────────────────────────

class TFIDFClassifier:
    """
    Lightweight keyword-based classifier using TF-IDF vectors.

    How it works:
      1. Build a "category document" by joining all keywords for each category.
      2. Vectorize category documents and the input text with TF-IDF.
      3. Compute cosine similarity between input and each category.
      4. Return the category with the highest similarity as prediction.

    No training required — runs entirely offline.
    """

    def __init__(self):
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),    # unigrams + bigrams
            stop_words="english",
            max_features=5000,
        )
        self._category_names = list(CATEGORIES.keys())

        # Build category corpus
        category_docs = [
            " ".join(keywords)
            for keywords in CATEGORIES.values()
        ]

        # Fit on category keyword docs
        self._vectorizer.fit(category_docs)
        self._category_vectors = self._vectorizer.transform(category_docs)

    def predict(self, text: str) -> dict[str, float]:
        """
        Predict category scores for the given text.

        Args:
            text: Document text (or excerpt) to classify.

        Returns:
            Dict mapping category_name → similarity_score (0–1).
            Scores sum to approximately 1 after softmax normalization.
        """
        text_vector = self._vectorizer.transform([text])
        similarities = cosine_similarity(
            text_vector, self._category_vectors
        )[0]

        # Softmax normalization so scores sum to ~1
        exp_sim = np.exp(similarities - similarities.max())
        normalized = exp_sim / exp_sim.sum()

        return {
            name: float(round(score, 4))
            for name, score in zip(self._category_names, normalized)
        }


# Singleton instance — created once
_tfidf_classifier: TFIDFClassifier | None = None


def _get_tfidf_classifier() -> TFIDFClassifier:
    global _tfidf_classifier
    if _tfidf_classifier is None:
        logger.info("Initializing TF-IDF classifier ...")
        _tfidf_classifier = TFIDFClassifier()
    return _tfidf_classifier


# ── LLM Classification ─────────────────────────────────────────────────────────

def _classify_with_llm(
    text_excerpt: str,
) -> tuple[str, float, str] | None:
    """
    Ask the Groq LLM to classify the document.

    Args:
        text_excerpt: First ~2000 chars of the document text.

    Returns:
        (category, confidence, reasoning) or None on failure.
    """
    category_list = "\n".join(
        f"  - {name}" for name in CATEGORIES.keys()
    )
    system = CLASSIFY_SYSTEM.format(categories=category_list)
    user = CLASSIFY_USER.format(text_excerpt=text_excerpt)

    llm = get_llm(temperature=0.0)   # deterministic output
    try:
        response = llm.invoke([
            SystemMessage(content=system),
            HumanMessage(content=user),
        ])
        raw = response.content.strip()

        # Strip markdown code fences if present
        raw = re.sub(r"```json|```", "", raw).strip()

        parsed = json.loads(raw)
        category  = parsed.get("category", "").strip()
        confidence = float(parsed.get("confidence", 0.0))
        reasoning  = parsed.get("reasoning", "")

        # Validate category is in our taxonomy
        if category not in CATEGORIES:
            # Try case-insensitive match
            for known in CATEGORIES:
                if known.lower() == category.lower():
                    category = known
                    break
            else:
                logger.warning(
                    f"LLM returned unknown category: '{category}'. "
                    "Falling back to TF-IDF."
                )
                return None

        confidence = max(0.0, min(1.0, confidence))   # clamp to [0, 1]
        return category, confidence, reasoning

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.warning(f"LLM classification parse error: {e}. Using TF-IDF.")
        return None
    except Exception as e:
        logger.error(f"LLM classification failed: {e}. Using TF-IDF.")
        return None


# ── Public API ─────────────────────────────────────────────────────────────────

def classify_document(document_id: str) -> ClassificationResult:
    """
    Classify a previously uploaded document into a category.

    Strategy:
      1. Load the document's chunk texts.
      2. Prepare a representative excerpt (first ~2000 chars).
      3. Run LLM classification.
      4. Run TF-IDF classification in parallel (always).
      5. Combine:
         - If LLM succeeds and confidence ≥ 0.6 → use LLM result.
         - If LLM fails or confidence < 0.6   → use TF-IDF result.
         - Always return all_scores from TF-IDF for the score breakdown.

    Args:
        document_id: The document's ID (returned from upload).

    Returns:
        ClassificationResult with category, confidence, and all scores.

    Raises:
        FileNotFoundError: If the document has not been uploaded.
    """
    start = time.perf_counter()

    # ── Load document chunks ───────────────────────────────────────────────
    chunks_path = Path(settings.docs_dir) / f"{document_id}_chunks.json"
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Chunks not found for document '{document_id}'. "
            "Was the PDF uploaded and processed?"
        )

    raw = json.loads(chunks_path.read_text(encoding="utf-8"))
    chunk_texts = [c["content"] for c in raw if c.get("content", "").strip()]
    filename = raw[0]["filename"] if raw else "unknown"

    if not chunk_texts:
        raise RuntimeError(
            f"Document '{document_id}' has no extractable text to classify."
        )

    # Use a representative sample: first 2000 chars of the document
    full_text = " ".join(chunk_texts)
    text_excerpt = full_text[:2000]

    logger.info(
        f"Classifying '{filename}' | "
        f"document_id={document_id} | "
        f"excerpt_chars={len(text_excerpt)}"
    )

    # ── TF-IDF classification (always runs) ───────────────────────────────
    tfidf = _get_tfidf_classifier()
    tfidf_scores = tfidf.predict(full_text[:5000])  # use more text for TF-IDF
    tfidf_category = max(tfidf_scores, key=tfidf_scores.get)
    tfidf_confidence = tfidf_scores[tfidf_category]

    logger.info(
        f"TF-IDF result: '{tfidf_category}' "
        f"(confidence={tfidf_confidence:.3f})"
    )

    # ── LLM classification ─────────────────────────────────────────────────
    llm_result = _classify_with_llm(text_excerpt)

    # ── Combine results ────────────────────────────────────────────────────
    LLM_CONFIDENCE_THRESHOLD = 0.60

    if llm_result and llm_result[1] >= LLM_CONFIDENCE_THRESHOLD:
        llm_category, llm_confidence, reasoning = llm_result
        predicted_category = llm_category
        confidence = llm_confidence
        method = "llm+tfidf"

        logger.info(
            f"LLM result accepted: '{llm_category}' "
            f"(confidence={llm_confidence:.3f})"
        )

        # Boost the LLM-chosen category in the all_scores breakdown
        # so it reflects the final decision clearly
        all_scores = dict(tfidf_scores)
        all_scores[predicted_category] = max(
            all_scores.get(predicted_category, 0.0),
            llm_confidence,
        )

    else:
        if llm_result:
            logger.info(
                f"LLM confidence too low "
                f"({llm_result[1]:.3f} < {LLM_CONFIDENCE_THRESHOLD}). "
                "Using TF-IDF result."
            )
        predicted_category = tfidf_category
        confidence = tfidf_confidence
        all_scores = tfidf_scores
        reasoning = (
            f"TF-IDF similarity-based classification "
            f"(LLM confidence was insufficient)."
        )
        method = "tfidf"

    processing_time = time.perf_counter() - start

    logger.info(
        f"Classification complete | "
        f"category='{predicted_category}' | "
        f"confidence={confidence:.3f} | "
        f"method={method} | "
        f"time={processing_time:.3f}s"
    )

    return ClassificationResult(
        document_id=document_id,
        filename=filename,
        predicted_category=predicted_category,
        confidence=confidence,
        all_scores=all_scores,
        method=method,
        reasoning=reasoning,
        processing_time=processing_time,
    )