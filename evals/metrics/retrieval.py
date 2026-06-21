"""
Retrieval evaluation metrics.

These metrics answer: did the RAG system retrieve the right chunks?
They operate on the retrieved chunk list independently of generation.

Metrics:
    recall_at_k          -- fraction of expected sources found in top-k chunks
    mean_reciprocal_rank -- how high the first relevant chunk ranks
    source_coverage      -- how many distinct expected sources are represented
    detect_missing_sources -- which expected sources were not retrieved
"""

from __future__ import annotations

import logging
from typing import List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class RetrievalMetrics(BaseModel):
    """All retrieval metrics for a single eval question run."""

    question_id: str
    recall_at_k: float = Field(ge=0.0, le=1.0)
    mrr: float = Field(ge=0.0, le=1.0, description="Mean Reciprocal Rank")
    source_coverage: float = Field(ge=0.0, le=1.0)
    missing_sources: List[str]
    chunks_retrieved: int
    top_score: float
    k: int


def _url_matches(source_pattern: str, chunk_url: str) -> bool:
    """Check if a source pattern appears as a substring of a chunk URL."""
    return source_pattern.lower() in chunk_url.lower()


def recall_at_k(
    retrieved_urls: List[str],
    expected_sources: List[str],
    k: int,
) -> float:
    """
    Fraction of expected sources that appear in the top-k retrieved chunk URLs.

    A source is considered found if any of the top-k retrieved URLs contains
    the expected source pattern as a substring.

    Args:
        retrieved_urls: Ordered list of URLs for retrieved chunks (most relevant first).
        expected_sources: List of URL substrings that must appear in retrieval.
        k: Number of top chunks to consider.

    Returns:
        Float in [0, 1]. 1.0 means all expected sources were found in top-k.
    """
    if not expected_sources:
        return 1.0

    top_k_urls = retrieved_urls[:k]
    found = 0
    for expected in expected_sources:
        if any(_url_matches(expected, url) for url in top_k_urls):
            found += 1

    return found / len(expected_sources)


def mean_reciprocal_rank(
    retrieved_urls: List[str],
    expected_sources: List[str],
) -> float:
    """
    Mean Reciprocal Rank across expected sources.

    For each expected source, finds the rank (1-indexed) of the first retrieved
    chunk whose URL matches, then averages the reciprocals.

    Returns:
        Float in [0, 1]. 0.0 means no expected sources were found.
    """
    if not expected_sources:
        return 1.0

    reciprocal_ranks: List[float] = []
    for expected in expected_sources:
        rr = 0.0
        for rank, url in enumerate(retrieved_urls, start=1):
            if _url_matches(expected, url):
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def source_coverage(
    retrieved_urls: List[str],
    expected_sources: List[str],
) -> float:
    """
    Fraction of distinct expected sources covered anywhere in retrieval.

    Unlike recall_at_k, this considers all retrieved chunks, not just top-k.
    Useful for understanding whether the corpus contains the right material
    regardless of ranking quality.
    """
    if not expected_sources:
        return 1.0

    found = sum(
        1 for expected in expected_sources
        if any(_url_matches(expected, url) for url in retrieved_urls)
    )
    return found / len(expected_sources)


def detect_missing_sources(
    retrieved_urls: List[str],
    expected_sources: List[str],
) -> List[str]:
    """
    Return the list of expected sources not found anywhere in retrieved chunks.

    These are actionable signals: if a source consistently goes missing,
    it may be absent from the corpus, chunked in a way that makes it
    unretrievable, or under-represented in the embedding space.
    """
    return [
        expected for expected in expected_sources
        if not any(_url_matches(expected, url) for url in retrieved_urls)
    ]


def compute_retrieval_metrics(
    question_id: str,
    retrieved_chunks: List[dict],
    expected_sources: List[str],
    k: int = 5,
) -> RetrievalMetrics:
    """
    Compute all retrieval metrics from a list of retrieved chunk dicts.

    Each chunk dict must have a 'url' key and a 'score' key.
    """
    urls = [c.get("url", "") for c in retrieved_chunks]
    scores = [c.get("score", 0.0) for c in retrieved_chunks]
    top_score = max(scores, default=0.0)

    return RetrievalMetrics(
        question_id=question_id,
        recall_at_k=recall_at_k(urls, expected_sources, k),
        mrr=mean_reciprocal_rank(urls, expected_sources),
        source_coverage=source_coverage(urls, expected_sources),
        missing_sources=detect_missing_sources(urls, expected_sources),
        chunks_retrieved=len(retrieved_chunks),
        top_score=top_score,
        k=k,
    )
