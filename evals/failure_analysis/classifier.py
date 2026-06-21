"""
Failure mode classifier.

After the runner collects metrics, the classifier labels what went wrong.
Each failure mode is mutually non-exclusive — a single response can have
multiple failure modes (e.g., retrieval miss AND partial answer).

Failure modes:
    retrieval_miss          No retrieved chunks matched expected sources
    wrong_chunk_retrieved   Chunks retrieved but all from wrong documents
    citation_hallucination  Answer references sources not in retrieved set
    answer_not_supported    Answer makes claims not supported by any chunk
    partial_answer          Required facts are partially missing (>0 but <50% covered)
    format_failure          Answer does not follow required output structure
    timeout_or_tool_failure Adapter returned an error or timed out
    lost_in_context         Retrieval was good but relevant content ignored in generation
    unknown                 Failure detected but does not fit any specific category
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import List

logger = logging.getLogger(__name__)


class FailureMode(str, Enum):
    retrieval_miss = "retrieval_miss"
    wrong_chunk_retrieved = "wrong_chunk_retrieved"
    citation_hallucination = "citation_hallucination"
    answer_not_supported = "answer_not_supported"
    partial_answer = "partial_answer"
    format_failure = "format_failure"
    timeout_or_tool_failure = "timeout_or_tool_failure"
    lost_in_context = "lost_in_context"
    unknown = "unknown"


class FailureClassifier:
    """
    Classifies failure modes from runner outputs.

    Uses metric thresholds, not model calls — intentionally deterministic
    so classifications are reproducible and auditable.
    """

    REQUIRED_SECTIONS = [
        "Overview:",
        "Key Points:",
        "Who Regulates It:",
        "Practical Takeaway:",
        "Limits of Retrieved Evidence:",
    ]

    def classify(
        self,
        rag_response,
        retrieval_metrics,
        generation_metrics,
    ) -> List[FailureMode]:
        """
        Return all failure modes that apply to this result.

        Ordering: infrastructure failures first, then retrieval, then generation.
        """
        failures: List[FailureMode] = []

        # Infrastructure: adapter-level errors
        if rag_response.error:
            failures.append(FailureMode.timeout_or_tool_failure)
            return failures  # No further analysis if the pipeline itself failed

        # Retrieval: no chunks at all
        if retrieval_metrics.chunks_retrieved == 0:
            failures.append(FailureMode.retrieval_miss)
            return failures

        # Retrieval: chunks retrieved but none from expected sources
        if retrieval_metrics.source_coverage == 0.0 and retrieval_metrics.chunks_retrieved > 0:
            failures.append(FailureMode.wrong_chunk_retrieved)

        # Retrieval: expected sources partially missing
        elif retrieval_metrics.recall_at_k < 0.5 and retrieval_metrics.source_coverage < 1.0:
            failures.append(FailureMode.retrieval_miss)

        # Generation: answer empty or too short to be meaningful
        answer = rag_response.answer or ""
        if len(answer.strip()) < 50:
            failures.append(FailureMode.format_failure)
            return failures

        # Generation: format does not match required structure
        missing_sections = [s for s in self.REQUIRED_SECTIONS if s not in answer]
        if len(missing_sections) >= 3:
            failures.append(FailureMode.format_failure)

        # Generation: citation hallucination
        if not generation_metrics.citation_support_check:
            failures.append(FailureMode.citation_hallucination)

        # Generation: facts partially covered
        if 0.0 < generation_metrics.fact_coverage <= 0.5:
            failures.append(FailureMode.partial_answer)
        elif generation_metrics.fact_coverage == 0.0:
            failures.append(FailureMode.answer_not_supported)

        # Generation: retrieval looked good but facts still missing (context lost)
        if (
            retrieval_metrics.recall_at_k >= 0.7
            and generation_metrics.fact_coverage < 0.5
            and FailureMode.answer_not_supported not in failures
        ):
            failures.append(FailureMode.lost_in_context)

        # Unknown: something failed but we can't pinpoint it
        if not failures and (
            generation_metrics.rubric_score < 0.4
            or (generation_metrics.semantic_similarity >= 0 and generation_metrics.semantic_similarity < 0.3)
        ):
            failures.append(FailureMode.unknown)

        return failures
