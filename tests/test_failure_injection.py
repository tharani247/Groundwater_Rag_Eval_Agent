"""
Tests for failure injection behavior.

Verifies that every injection scenario:
1. Does NOT crash the evaluation runner
2. Produces a RunResult with the expected failure mode(s)
3. Returns a properly structured RAGResponse
"""

import pytest

from evals.datasets.schema import EvalDataset, EvalQuestion, DifficultyLevel, QuestionCategory, Rubric
from evals.failure_analysis.classifier import FailureMode
from evals.failure_analysis.injector import FailureInjector, InjectionScenario
from evals.runners.eval_runner import EvalRunner
from evals.runners.rag_adapter import MockRAGAdapter


SINGLE_QUESTION = EvalDataset(
    name="Injection Test",
    version="1.0",
    description="Minimal dataset for injection testing",
    questions=[
        EvalQuestion(
            id="INJ-001",
            question="Who regulates groundwater in Nebraska?",
            expected_answer="Natural Resources Districts regulate groundwater under Chapter 46.",
            expected_sources=["nebraskalegislature.gov", "cpnrd.org"],
            minimum_required_facts=["NRDs regulate groundwater", "Chapter 46 provides authority"],
            difficulty=DifficultyLevel.easy,
            category=QuestionCategory.regulatory_authority,
            rubric=Rubric(
                mentions_regulatory_authority=True,
                cites_specific_statute_or_rule=False,
                answers_without_hallucination=True,
                provides_actionable_guidance=False,
            ),
        )
    ],
)


def _run_scenario(scenario: InjectionScenario, tmp_path):
    adapter = FailureInjector.get(scenario)
    runner = EvalRunner(adapter=adapter, log_dir=tmp_path, k=5)
    summary = runner.run(SINGLE_QUESTION, model_tag=f"injection:{scenario.value}")
    return summary


class TestFailureInjectionBehavior:
    """Every scenario must complete without raising — runner must be bulletproof."""

    def test_empty_retrieval_does_not_crash(self, tmp_path):
        summary = _run_scenario(InjectionScenario.empty_retrieval, tmp_path)
        assert summary.total_questions == 1
        assert summary.errored == 0

    def test_empty_retrieval_flags_retrieval_miss(self, tmp_path):
        summary = _run_scenario(InjectionScenario.empty_retrieval, tmp_path)
        modes = list(summary.failure_mode_counts.keys())
        assert "retrieval_miss" in modes

    def test_slow_response_does_not_crash(self, tmp_path):
        summary = _run_scenario(InjectionScenario.slow_response, tmp_path)
        assert summary.total_questions == 1
        # Slow response should still produce a valid answer
        result = summary.results[0]
        assert result.rag_response.latency_seconds >= 0

    def test_invalid_citation_format_flagged(self, tmp_path):
        summary = _run_scenario(InjectionScenario.invalid_citation_format, tmp_path)
        modes = list(summary.failure_mode_counts.keys())
        assert "citation_hallucination" in modes

    def test_missing_source_metadata_does_not_crash(self, tmp_path):
        summary = _run_scenario(InjectionScenario.missing_source_metadata, tmp_path)
        assert summary.total_questions == 1
        result = summary.results[0]
        # All URLs should be empty strings, not None
        for chunk in result.rag_response.retrieved_chunks:
            assert chunk.url == ""

    def test_malformed_output_flags_format_failure(self, tmp_path):
        summary = _run_scenario(InjectionScenario.malformed_output, tmp_path)
        modes = list(summary.failure_mode_counts.keys())
        assert "format_failure" in modes

    def test_adapter_exception_is_caught(self, tmp_path):
        """The runner must catch adapter exceptions and produce a valid summary."""
        summary = _run_scenario(InjectionScenario.adapter_exception, tmp_path)
        assert summary.total_questions == 1
        result = summary.results[0]
        assert result.rag_response.error is not None
        assert FailureMode.timeout_or_tool_failure in result.failure_modes

    def test_partial_answer_flags_partial_failure(self, tmp_path):
        summary = _run_scenario(InjectionScenario.partial_answer, tmp_path)
        modes = list(summary.failure_mode_counts.keys())
        # Partial answer produces either partial_answer or answer_not_supported
        assert "partial_answer" in modes or "answer_not_supported" in modes

    def test_all_scenarios_produce_json_logs(self, tmp_path):
        """Every scenario must produce a log file — no silent failures."""
        for scenario in FailureInjector.all_scenarios():
            _run_scenario(scenario, tmp_path / scenario.value)
            run_dirs = list((tmp_path / scenario.value).iterdir())
            assert len(run_dirs) >= 1, f"No log dir for scenario {scenario.value}"
