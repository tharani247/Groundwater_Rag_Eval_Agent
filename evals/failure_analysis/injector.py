"""
Failure injection tests.

Simulates known failure scenarios to verify the evaluation harness handles
edge cases gracefully — the system must not crash, must log clearly,
and must classify the failure correctly.

Each InjectionScenario wraps a MockRAGAdapter with a specific fault.
The EvalRunner should produce a non-empty RunResult (with failure modes)
for every scenario — never raise an unhandled exception.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from ..runners.rag_adapter import MockRAGAdapter, RAGAdapter, RAGResponse, RetrievedChunk

logger = logging.getLogger(__name__)


class InjectionScenario(str, Enum):
    empty_retrieval = "empty_retrieval"
    slow_response = "slow_response"
    invalid_citation_format = "invalid_citation_format"
    missing_source_metadata = "missing_source_metadata"
    malformed_output = "malformed_output"
    adapter_exception = "adapter_exception"
    partial_answer = "partial_answer"


class EmptyRetrievalAdapter(RAGAdapter):
    """Simulates a retrieval step that returns zero chunks."""

    def query(self, question: str, top_k: int = 8, min_score: float = 0.45) -> RAGResponse:
        logger.debug("[injection] empty_retrieval triggered for: %s", question[:60])
        return RAGResponse(
            question=question,
            answer="I could not find any relevant information in the retrieved sources.",
            retrieved_chunks=[],
            latency_seconds=0.05,
        )


class SlowResponseAdapter(RAGAdapter):
    """Simulates a model response that takes an unusually long time."""

    def __init__(self, delay_seconds: float = 2.0):
        self._delay = delay_seconds

    def query(self, question: str, top_k: int = 8, min_score: float = 0.45) -> RAGResponse:
        logger.debug("[injection] slow_response: sleeping %.1fs", self._delay)
        time.sleep(self._delay)
        return MockRAGAdapter().query(question, top_k, min_score)


class InvalidCitationFormatAdapter(RAGAdapter):
    """
    Returns an answer that violates the citation format rules.
    The prompt instructs the model to NOT use bracketed source refs.
    This adapter injects them to test citation_support_check detection.
    """

    def query(self, question: str, top_k: int = 8, min_score: float = 0.45) -> RAGResponse:
        base = MockRAGAdapter().query(question, top_k, min_score)
        # Inject bracketed source references the prompt prohibits
        bad_answer = (
            "Overview:\n"
            "According to [Source 1] and [Source 2], Nebraska groundwater is regulated by NRDs. "
            "[Source 3] further confirms the permit requirement.\n\n"
            "Key Points:\n"
            "- NRDs issue permits [Source 1]\n"
            "- DNR coordinates management [Source 2]\n\n"
            "Who Regulates It:\n"
            "NRDs [Source 1].\n\n"
            "Practical Takeaway:\n"
            "Contact your NRD [Source 3].\n\n"
            "Limits of Retrieved Evidence:\n"
            "None noted."
        )
        return RAGResponse(
            question=question,
            answer=bad_answer,
            retrieved_chunks=base.retrieved_chunks,
            latency_seconds=base.latency_seconds,
        )


class MissingSourceMetadataAdapter(RAGAdapter):
    """
    Returns chunks where URL and title fields are empty.
    Tests whether metrics gracefully handle missing metadata.
    """

    def query(self, question: str, top_k: int = 8, min_score: float = 0.45) -> RAGResponse:
        base = MockRAGAdapter().query(question, top_k, min_score)
        stripped_chunks = [
            RetrievedChunk(
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                title="",
                url="",
                chunk_index=c.chunk_index,
                text=c.text,
                score=c.score,
            )
            for c in base.retrieved_chunks
        ]
        return RAGResponse(
            question=question,
            answer=base.answer,
            retrieved_chunks=stripped_chunks,
            latency_seconds=base.latency_seconds,
        )


class MalformedOutputAdapter(RAGAdapter):
    """
    Returns a response that is missing required answer sections.
    Tests format_failure detection.
    """

    def query(self, question: str, top_k: int = 8, min_score: float = 0.45) -> RAGResponse:
        base = MockRAGAdapter().query(question, top_k, min_score)
        # Intentionally incomplete output missing most sections
        malformed = "Nebraska has some groundwater laws. NRDs handle permits. Contact them."
        return RAGResponse(
            question=question,
            answer=malformed,
            retrieved_chunks=base.retrieved_chunks,
            latency_seconds=base.latency_seconds,
        )


class AdapterExceptionAdapter(RAGAdapter):
    """Simulates an adapter that raises an unexpected exception during query."""

    def query(self, question: str, top_k: int = 8, min_score: float = 0.45) -> RAGResponse:
        raise RuntimeError("Simulated adapter failure: database connection lost")


class PartialAnswerAdapter(RAGAdapter):
    """
    Returns an answer that covers only a subset of the required facts.
    Tests partial_answer detection.
    """

    def query(self, question: str, top_k: int = 8, min_score: float = 0.45) -> RAGResponse:
        base = MockRAGAdapter().query(question, top_k, min_score)
        partial = (
            "Overview:\n"
            "Groundwater in Nebraska has some regulatory oversight.\n\n"
            "Key Points:\n"
            "- Some permits may be required\n\n"
            "Who Regulates It:\n"
            "State agencies.\n\n"
            "Practical Takeaway:\n"
            "Check with local authorities.\n\n"
            "Limits of Retrieved Evidence:\n"
            "Much of the detail is not available in the retrieved sources."
        )
        return RAGResponse(
            question=question,
            answer=partial,
            retrieved_chunks=base.retrieved_chunks,
            latency_seconds=base.latency_seconds,
        )


class FailureInjector:
    """
    Factory for creating adapters that simulate specific failure scenarios.

    Usage:
        adapter = FailureInjector.get(InjectionScenario.empty_retrieval)
        response = adapter.query("What permits are required?")
    """

    _registry = {
        InjectionScenario.empty_retrieval: EmptyRetrievalAdapter,
        InjectionScenario.slow_response: SlowResponseAdapter,
        InjectionScenario.invalid_citation_format: InvalidCitationFormatAdapter,
        InjectionScenario.missing_source_metadata: MissingSourceMetadataAdapter,
        InjectionScenario.malformed_output: MalformedOutputAdapter,
        InjectionScenario.adapter_exception: AdapterExceptionAdapter,
        InjectionScenario.partial_answer: PartialAnswerAdapter,
    }

    @classmethod
    def get(cls, scenario: InjectionScenario, **kwargs) -> RAGAdapter:
        adapter_cls = cls._registry.get(scenario)
        if adapter_cls is None:
            raise ValueError(f"Unknown injection scenario: {scenario}")
        return adapter_cls(**kwargs)

    @classmethod
    def all_scenarios(cls) -> List[InjectionScenario]:
        return list(cls._registry.keys())
