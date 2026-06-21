"""Tests for failure mode classification logic."""

import pytest

from evals.failure_analysis.classifier import FailureClassifier, FailureMode
from evals.metrics.retrieval import RetrievalMetrics
from evals.metrics.generation import GenerationMetrics
from evals.runners.rag_adapter import RAGResponse, RetrievedChunk


def _make_retrieval(
    recall=1.0, mrr=1.0, coverage=1.0, chunks=3, top_score=0.82, missing=None
):
    return RetrievalMetrics(
        question_id="TEST-001",
        recall_at_k=recall,
        mrr=mrr,
        source_coverage=coverage,
        missing_sources=missing or [],
        chunks_retrieved=chunks,
        top_score=top_score,
        k=5,
    )


def _make_generation(
    fact_cov=1.0, rubric=1.0, sem_sim=0.9, citation_ok=True, missing_facts=None
):
    return GenerationMetrics(
        question_id="TEST-001",
        exact_match=False,
        semantic_similarity=sem_sim,
        fact_coverage=fact_cov,
        facts_missing=missing_facts or [],
        citation_support_check=citation_ok,
        rubric_score=rubric,
        rubric_details={},
    )


def _make_response(answer="", error=None, chunks=3):
    chunk_list = [
        RetrievedChunk(
            chunk_id=f"c{i}", doc_id=f"d{i}", title="T", url="https://cpnrd.org",
            chunk_index=i, text="Nebraska groundwater NRD regulation", score=0.8
        )
        for i in range(chunks)
    ]
    full_answer = answer or (
        "Overview:\nNRDs regulate groundwater.\n\n"
        "Key Points:\n- Chapter 46\n\n"
        "Who Regulates It:\nNRD.\n\n"
        "Practical Takeaway:\nContact NRD.\n\n"
        "Limits of Retrieved Evidence:\nSome limits apply."
    )
    return RAGResponse(
        question="test?",
        answer=full_answer if not error else "",
        retrieved_chunks=chunk_list if not error else [],
        latency_seconds=0.1,
        error=error,
    )


clf = FailureClassifier()


class TestFailureClassifier:
    def test_no_failure_on_clean_result(self):
        modes = clf.classify(_make_response(), _make_retrieval(), _make_generation())
        assert modes == []

    def test_timeout_on_error(self):
        modes = clf.classify(
            _make_response(error="DB connection lost"),
            _make_retrieval(chunks=0),
            _make_generation(),
        )
        assert FailureMode.timeout_or_tool_failure in modes

    def test_retrieval_miss_on_zero_chunks(self):
        modes = clf.classify(
            _make_response(chunks=0),
            _make_retrieval(chunks=0, recall=0.0, coverage=0.0),
            _make_generation(),
        )
        assert FailureMode.retrieval_miss in modes

    def test_wrong_chunk_on_zero_coverage(self):
        modes = clf.classify(
            _make_response(),
            _make_retrieval(coverage=0.0, recall=0.0, chunks=3),
            _make_generation(),
        )
        assert FailureMode.wrong_chunk_retrieved in modes

    def test_citation_hallucination_flagged(self):
        modes = clf.classify(
            _make_response(),
            _make_retrieval(),
            _make_generation(citation_ok=False),
        )
        assert FailureMode.citation_hallucination in modes

    def test_partial_answer_on_low_fact_coverage(self):
        modes = clf.classify(
            _make_response(),
            _make_retrieval(),
            _make_generation(fact_cov=0.3),
        )
        assert FailureMode.partial_answer in modes

    def test_answer_not_supported_on_zero_facts(self):
        modes = clf.classify(
            _make_response(),
            _make_retrieval(),
            _make_generation(fact_cov=0.0),
        )
        assert FailureMode.answer_not_supported in modes

    def test_format_failure_on_short_answer(self):
        modes = clf.classify(
            _make_response(answer="Too short."),
            _make_retrieval(),
            _make_generation(),
        )
        assert FailureMode.format_failure in modes

    def test_lost_in_context_when_retrieval_good_generation_bad(self):
        modes = clf.classify(
            _make_response(),
            _make_retrieval(recall=0.9, coverage=0.9),
            _make_generation(fact_cov=0.3),
        )
        assert FailureMode.lost_in_context in modes

    def test_multiple_failure_modes_coexist(self):
        modes = clf.classify(
            _make_response(),
            _make_retrieval(),
            _make_generation(fact_cov=0.0, citation_ok=False),
        )
        assert FailureMode.answer_not_supported in modes
        assert FailureMode.citation_hallucination in modes
