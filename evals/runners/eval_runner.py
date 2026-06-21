"""
Agentic evaluation runner.

Iterates over an EvalDataset, queries the RAG adapter for each question,
computes retrieval and generation metrics, classifies failure modes,
and writes structured JSON logs.

Design principles:
    - Each run gets a unique run_id and its own log file
    - Failures never crash the runner — they are captured and classified
    - Every intermediate result is persisted so partial runs are recoverable
    - The runner is stateless; multiple instances can run concurrently
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel

from ..datasets.schema import EvalDataset, EvalQuestion
from ..failure_analysis.classifier import FailureClassifier, FailureMode
from ..metrics.generation import compute_generation_metrics, GenerationMetrics
from ..metrics.retrieval import compute_retrieval_metrics, RetrievalMetrics
from .rag_adapter import RAGAdapter, RAGResponse

logger = logging.getLogger(__name__)

LOG_DIR = Path(__file__).parent.parent / "logs"


class RunResult(BaseModel):
    """Complete result for a single eval question."""

    question_id: str
    question: str
    category: str
    difficulty: str
    rag_response: RAGResponse
    retrieval_metrics: RetrievalMetrics
    generation_metrics: GenerationMetrics
    failure_modes: List[FailureMode]
    passed: bool
    timestamp_utc: str


class EvalRunSummary(BaseModel):
    """Aggregate summary across all questions in a run."""

    run_id: str
    dataset_name: str
    model_tag: str
    started_at_utc: str
    completed_at_utc: str
    total_questions: int
    passed: int
    failed: int
    errored: int
    overall_score: float
    avg_recall_at_k: float
    avg_mrr: float
    avg_source_coverage: float
    avg_semantic_similarity: float
    avg_fact_coverage: float
    avg_rubric_score: float
    avg_latency_seconds: float
    failure_mode_counts: dict
    results: List[RunResult]


class EvalRunner:
    """
    Runs a full evaluation pass over an EvalDataset.

    Usage:
        adapter = MockRAGAdapter()
        runner = EvalRunner(adapter, log_dir="evals/logs")
        summary = runner.run(dataset, model_tag="gemini-2.5-flash")
    """

    def __init__(
        self,
        adapter: RAGAdapter,
        log_dir: Optional[Path | str] = None,
        k: int = 5,
    ):
        self._adapter = adapter
        self._log_dir = Path(log_dir) if log_dir else LOG_DIR
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._classifier = FailureClassifier()
        self._k = k

    def run(
        self,
        dataset: EvalDataset,
        model_tag: str = "unknown",
        run_id: Optional[str] = None,
    ) -> EvalRunSummary:
        """
        Execute evaluation over every question in the dataset.

        Args:
            dataset:   The EvalDataset to evaluate against.
            model_tag: Identifier for the model/config being evaluated.
            run_id:    Optional fixed run ID (auto-generated if omitted).

        Returns:
            EvalRunSummary with all per-question results and aggregate metrics.
        """
        run_id = run_id or f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        started_at = datetime.now(timezone.utc).isoformat()

        logger.info(
            "Starting eval run '%s' — %d questions from dataset '%s'",
            run_id, len(dataset.questions), dataset.name,
        )

        results: List[RunResult] = []
        errored = 0

        for i, question in enumerate(dataset.questions, start=1):
            logger.info("[%d/%d] Evaluating %s", i, len(dataset.questions), question.id)
            result = self._evaluate_question(question)
            results.append(result)

            if result.rag_response.error:
                errored += 1

            # Persist incrementally so partial runs are recoverable
            self._write_result_log(run_id, result)

        completed_at = datetime.now(timezone.utc).isoformat()
        summary = self._build_summary(run_id, dataset, model_tag, started_at, completed_at, results, errored)

        self._write_summary_log(run_id, summary)
        self._write_latest_symlink(run_id)

        logger.info(
            "Run '%s' complete. Score: %.2f | Passed: %d/%d",
            run_id, summary.overall_score, summary.passed, summary.total_questions,
        )
        return summary

    def _evaluate_question(self, question: EvalQuestion) -> RunResult:
        """Run and score a single eval question."""
        try:
            rag_response = self._adapter.query(question.question)
        except Exception as exc:
            logger.error("Adapter exception on %s: %s", question.id, exc)
            from .rag_adapter import RAGResponse
            rag_response = RAGResponse(
                question=question.question,
                answer="",
                retrieved_chunks=[],
                latency_seconds=0.0,
                error=str(exc),
            )

        retrieved_chunk_dicts = [
            {"url": c.url, "score": c.score}
            for c in rag_response.retrieved_chunks
        ]
        retrieved_texts = [c.text for c in rag_response.retrieved_chunks]

        retrieval_metrics = compute_retrieval_metrics(
            question_id=question.id,
            retrieved_chunks=retrieved_chunk_dicts,
            expected_sources=question.expected_sources,
            k=self._k,
        )

        generation_metrics = compute_generation_metrics(
            question_id=question.id,
            generated_answer=rag_response.answer,
            expected_answer=question.expected_answer,
            required_facts=question.minimum_required_facts,
            rubric=question.rubric.model_dump(),
            retrieved_chunk_texts=retrieved_texts,
        )

        failure_modes = self._classifier.classify(
            rag_response=rag_response,
            retrieval_metrics=retrieval_metrics,
            generation_metrics=generation_metrics,
        )

        passed = (
            not failure_modes
            and retrieval_metrics.recall_at_k >= 0.5
            and generation_metrics.fact_coverage >= 0.5
            and generation_metrics.rubric_score >= 0.5
        )

        return RunResult(
            question_id=question.id,
            question=question.question,
            category=question.category.value,
            difficulty=question.difficulty.value,
            rag_response=rag_response,
            retrieval_metrics=retrieval_metrics,
            generation_metrics=generation_metrics,
            failure_modes=failure_modes,
            passed=passed,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
        )

    def _build_summary(
        self,
        run_id: str,
        dataset: EvalDataset,
        model_tag: str,
        started_at: str,
        completed_at: str,
        results: List[RunResult],
        errored: int,
    ) -> EvalRunSummary:
        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed

        def avg(values: List[float]) -> float:
            return round(sum(values) / len(values), 4) if values else 0.0

        recall_scores = [r.retrieval_metrics.recall_at_k for r in results]
        mrr_scores = [r.retrieval_metrics.mrr for r in results]
        coverage_scores = [r.retrieval_metrics.source_coverage for r in results]
        sem_sim_scores = [r.generation_metrics.semantic_similarity for r in results if r.generation_metrics.semantic_similarity >= 0]
        fact_scores = [r.generation_metrics.fact_coverage for r in results]
        rubric_scores = [r.generation_metrics.rubric_score for r in results]
        latencies = [r.rag_response.latency_seconds for r in results]

        retrieval_score = avg([avg(recall_scores), avg(mrr_scores), avg(coverage_scores)])
        generation_score = avg([avg(fact_scores), avg(rubric_scores)])
        overall_score = round((retrieval_score + generation_score) / 2, 4)

        failure_mode_counts: dict = {}
        for r in results:
            for fm in r.failure_modes:
                key = fm.value
                failure_mode_counts[key] = failure_mode_counts.get(key, 0) + 1

        return EvalRunSummary(
            run_id=run_id,
            dataset_name=dataset.name,
            model_tag=model_tag,
            started_at_utc=started_at,
            completed_at_utc=completed_at,
            total_questions=len(results),
            passed=passed,
            failed=failed,
            errored=errored,
            overall_score=overall_score,
            avg_recall_at_k=avg(recall_scores),
            avg_mrr=avg(mrr_scores),
            avg_source_coverage=avg(coverage_scores),
            avg_semantic_similarity=avg(sem_sim_scores),
            avg_fact_coverage=avg(fact_scores),
            avg_rubric_score=avg(rubric_scores),
            avg_latency_seconds=avg(latencies),
            failure_mode_counts=failure_mode_counts,
            results=results,
        )

    def _write_result_log(self, run_id: str, result: RunResult) -> None:
        run_dir = self._log_dir / run_id
        run_dir.mkdir(exist_ok=True)
        log_path = run_dir / f"{result.question_id}.json"
        with log_path.open("w", encoding="utf-8") as f:
            json.dump(result.model_dump(), f, indent=2, default=str)

    def _write_summary_log(self, run_id: str, summary: EvalRunSummary) -> None:
        run_dir = self._log_dir / run_id
        run_dir.mkdir(exist_ok=True)
        summary_path = run_dir / "summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary.model_dump(), f, indent=2, default=str)
        logger.info("Summary written to %s", summary_path)

    def _write_latest_symlink(self, run_id: str) -> None:
        """Write a 'latest' file pointing to the most recent run_id."""
        latest_path = self._log_dir / "latest"
        try:
            latest_path.write_text(run_id, encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not write 'latest' pointer: %s", exc)
