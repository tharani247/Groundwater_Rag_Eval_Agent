"""
RAG system adapter layer.

Separates the evaluation harness from the underlying RAG implementation.
Swap LiveRAGAdapter for any other backend without touching the runner logic.

Two adapters provided:
    LiveRAGAdapter  -- calls the real retrieve_chunks_multi_query + Gemini pipeline
    MockRAGAdapter  -- deterministic mock for unit testing without DB or API
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import List, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    url: str
    chunk_index: int
    text: str
    score: float


class RAGResponse(BaseModel):
    """Structured output from a single RAG system call."""

    question: str
    answer: str
    retrieved_chunks: List[RetrievedChunk]
    latency_seconds: float
    token_estimate: Optional[int] = None
    error: Optional[str] = None
    raw_metadata: dict = {}


class RAGAdapter(ABC):
    """Abstract base class for all RAG system adapters."""

    @abstractmethod
    def query(self, question: str, top_k: int = 8, min_score: float = 0.45) -> RAGResponse:
        ...


class LiveRAGAdapter(RAGAdapter):
    """
    Calls the real groundwater RAG pipeline.

    Requires DATABASE_URL and GEMINI_API_KEY in the environment.
    """

    def __init__(self, dsn: str, output_dim: int = 1536):
        import sys
        import os
        # Make src importable when running from project root
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from src.ingest.embed_gemini import GeminiEmbedder
        from src.ingest.retrieve_gemini import (
            retrieve_chunks_multi_query,
            pick_top_docs,
            build_prompt,
            summarize_with_gemini,
        )

        self._dsn = dsn
        self._embedder = GeminiEmbedder(output_dim=output_dim)
        self._retrieve = retrieve_chunks_multi_query
        self._pick_top = pick_top_docs
        self._build_prompt = build_prompt
        self._summarize = summarize_with_gemini

    def query(self, question: str, top_k: int = 8, min_score: float = 0.45) -> RAGResponse:
        start = time.monotonic()
        error: Optional[str] = None
        answer = ""
        raw_chunks: list = []

        try:
            raw_chunks = self._retrieve(
                dsn=self._dsn,
                embedder=self._embedder,
                question=question,
                top_k_per_query=top_k,
                min_score=min_score,
            )
            top_chunks = self._pick_top(raw_chunks, docs_k=3, chunks_per_doc=2)
            prompt = self._build_prompt(question, top_chunks)
            answer = self._summarize(prompt)
        except Exception as exc:
            error = str(exc)
            logger.error("LiveRAGAdapter error for question '%s': %s", question[:60], exc)

        latency = time.monotonic() - start

        chunks = []
        for row in raw_chunks:
            chunk_id, doc_id, title, url, chunk_index, text, score = row
            chunks.append(
                RetrievedChunk(
                    chunk_id=str(chunk_id),
                    doc_id=str(doc_id),
                    title=title or "",
                    url=url or "",
                    chunk_index=chunk_index,
                    text=text,
                    score=float(score),
                )
            )

        return RAGResponse(
            question=question,
            answer=answer,
            retrieved_chunks=chunks,
            latency_seconds=round(latency, 3),
            error=error,
        )


class MockRAGAdapter(RAGAdapter):
    """
    Deterministic mock adapter for unit testing and CI.

    Returns configurable answers and chunks without hitting a database or API.
    Pass custom_responses to override specific questions.
    """

    DEFAULT_ANSWER = (
        "Overview:\n"
        "Nebraska groundwater is regulated primarily by Natural Resources Districts (NRDs) "
        "under the Groundwater Management and Protection Act (Chapter 46).\n\n"
        "Key Points:\n"
        "- NRDs issue well permits\n"
        "- Chapter 46 provides statutory authority\n"
        "- DNR coordinates integrated management\n"
        "- Violations may result in permit suspension\n\n"
        "Who Regulates It:\n"
        "Natural Resources Districts and the Nebraska DNR.\n\n"
        "Practical Takeaway:\n"
        "Contact your local NRD before drilling or modifying a well.\n\n"
        "Limits of Retrieved Evidence:\n"
        "Specific allocation numbers are not available in the retrieved sources."
    )

    DEFAULT_CHUNKS = [
        {
            "chunk_id": "mock-001",
            "doc_id": "doc-cpnrd",
            "title": "CPNRD Rules and Regulations",
            "url": "https://www.cpnrd.org/forms-permits/",
            "chunk_index": 0,
            "text": "Well permits are required before any drilling within the district.",
            "score": 0.82,
        },
        {
            "chunk_id": "mock-002",
            "doc_id": "doc-legislature",
            "title": "Nebraska Legislature - Chapter 46",
            "url": "https://nebraskalegislature.gov/laws/browse-chapters.php",
            "chunk_index": 1,
            "text": "The Groundwater Management and Protection Act grants authority to NRDs.",
            "score": 0.77,
        },
        {
            "chunk_id": "mock-003",
            "doc_id": "doc-dnr",
            "title": "Nebraska DNR Groundwater",
            "url": "https://dnr.nebraska.gov/water-planning/state-laws-and-rules",
            "chunk_index": 2,
            "text": "Integrated management plans coordinate surface and groundwater use.",
            "score": 0.71,
        },
    ]

    def __init__(
        self,
        custom_responses: Optional[dict] = None,
        simulate_latency: float = 0.05,
        force_error: Optional[str] = None,
    ):
        self._custom = custom_responses or {}
        self._latency = simulate_latency
        self._force_error = force_error

    def query(self, question: str, top_k: int = 8, min_score: float = 0.45) -> RAGResponse:
        time.sleep(self._latency)

        if self._force_error:
            return RAGResponse(
                question=question,
                answer="",
                retrieved_chunks=[],
                latency_seconds=self._latency,
                error=self._force_error,
            )

        answer = self._custom.get(question, self.DEFAULT_ANSWER)
        chunks = [RetrievedChunk(**c) for c in self.DEFAULT_CHUNKS[:top_k]]

        return RAGResponse(
            question=question,
            answer=answer,
            retrieved_chunks=chunks,
            latency_seconds=self._latency,
        )
