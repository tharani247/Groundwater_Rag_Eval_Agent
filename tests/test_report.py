"""Tests for benchmark report generation."""

import json
from pathlib import Path

import pytest

from evals.datasets.loader import load_dataset
from evals.failure_analysis.injector import FailureInjector, InjectionScenario
from evals.reports.generator import ReportGenerator
from evals.runners.eval_runner import EvalRunner
from evals.runners.rag_adapter import MockRAGAdapter


def _build_summary(tmp_path, dataset_path="evals/datasets/sample_groundtruth.json"):
    path = Path(dataset_path)
    if not path.exists():
        pytest.skip("Sample dataset not found")
    dataset = load_dataset(path)
    # Use just 3 questions for speed
    from evals.datasets.schema import EvalDataset
    mini = EvalDataset(
        name=dataset.name,
        version=dataset.version,
        description=dataset.description,
        questions=dataset.questions[:3],
    )
    runner = EvalRunner(adapter=MockRAGAdapter(), log_dir=tmp_path)
    return runner.run(mini, model_tag="mock-test")


class TestReportGenerator:
    def test_report_file_created(self, tmp_path):
        summary = _build_summary(tmp_path)
        gen = ReportGenerator(report_dir=tmp_path)
        report_path = gen.generate(summary)
        assert report_path.exists()
        assert report_path.suffix == ".md"

    def test_report_contains_required_sections(self, tmp_path):
        summary = _build_summary(tmp_path)
        gen = ReportGenerator(report_dir=tmp_path)
        report_path = gen.generate(summary)
        content = report_path.read_text(encoding="utf-8")

        required_sections = [
            "## Overall Score",
            "## Retrieval Metrics",
            "## Generation Metrics",
            "## Failure Mode Breakdown",
            "## Recommendations",
        ]
        for section in required_sections:
            assert section in content, f"Missing section: {section}"

    def test_report_contains_run_id(self, tmp_path):
        summary = _build_summary(tmp_path)
        gen = ReportGenerator(report_dir=tmp_path)
        report_path = gen.generate(summary)
        content = report_path.read_text(encoding="utf-8")
        assert summary.run_id in content

    def test_generate_from_run_id_latest(self, tmp_path):
        summary = _build_summary(tmp_path)
        gen = ReportGenerator(report_dir=tmp_path)
        report_path = gen.generate_from_run_id("latest")
        assert report_path.exists()

    def test_generate_from_missing_run_raises(self, tmp_path):
        gen = ReportGenerator(report_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            gen.generate_from_run_id("run_does_not_exist")

    def test_recommendations_not_empty(self, tmp_path):
        summary = _build_summary(tmp_path)
        gen = ReportGenerator(report_dir=tmp_path)
        report_path = gen.generate(summary)
        content = report_path.read_text(encoding="utf-8")
        # At least one numbered recommendation should exist
        assert "1. " in content
