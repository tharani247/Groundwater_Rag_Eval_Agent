"""Tests for dataset schema validation and loading."""

import json
import tempfile
from pathlib import Path

import pytest

from evals.datasets.schema import (
    DifficultyLevel,
    EvalDataset,
    EvalQuestion,
    QuestionCategory,
    Rubric,
)
from evals.datasets.loader import load_dataset, validate_dataset


SAMPLE_QUESTION = {
    "id": "TEST-001",
    "question": "Who regulates groundwater in Nebraska?",
    "expected_answer": "Natural Resources Districts regulate groundwater in Nebraska.",
    "expected_sources": ["nebraskalegislature.gov", "cpnrd.org"],
    "minimum_required_facts": ["NRDs regulate groundwater", "Chapter 46 provides authority"],
    "difficulty": "easy",
    "category": "regulatory_authority",
    "rubric": {
        "mentions_regulatory_authority": True,
        "cites_specific_statute_or_rule": False,
        "answers_without_hallucination": True,
        "provides_actionable_guidance": False,
        "acknowledges_limits": False,
    },
}


class TestEvalQuestionSchema:
    def test_valid_question_parses(self):
        q = EvalQuestion.model_validate(SAMPLE_QUESTION)
        assert q.id == "TEST-001"
        assert q.difficulty == DifficultyLevel.easy
        assert q.category == QuestionCategory.regulatory_authority

    def test_empty_question_raises(self):
        bad = {**SAMPLE_QUESTION, "question": "  "}
        with pytest.raises(ValueError, match="question must not be empty"):
            EvalQuestion.model_validate(bad)

    def test_empty_sources_raises(self):
        bad = {**SAMPLE_QUESTION, "expected_sources": []}
        with pytest.raises(ValueError, match="expected_sources"):
            EvalQuestion.model_validate(bad)

    def test_empty_facts_raises(self):
        bad = {**SAMPLE_QUESTION, "minimum_required_facts": []}
        with pytest.raises(ValueError, match="minimum_required_facts"):
            EvalQuestion.model_validate(bad)


class TestEvalDatasetSchema:
    def _make_dataset(self, questions=None):
        return {
            "name": "Test Dataset",
            "version": "1.0",
            "description": "Test",
            "questions": questions or [SAMPLE_QUESTION],
        }

    def test_valid_dataset_parses(self):
        ds = EvalDataset.model_validate(self._make_dataset())
        assert len(ds.questions) == 1

    def test_duplicate_ids_raises(self):
        with pytest.raises(ValueError, match="unique"):
            EvalDataset.model_validate(
                self._make_dataset(questions=[SAMPLE_QUESTION, SAMPLE_QUESTION])
            )


class TestDatasetLoader:
    def test_load_sample_groundtruth(self):
        path = Path("evals/datasets/sample_groundtruth.json")
        if not path.exists():
            pytest.skip("Sample dataset not found")
        dataset = load_dataset(path)
        assert len(dataset.questions) >= 15

    def test_validate_sample_groundtruth_no_errors(self):
        path = Path("evals/datasets/sample_groundtruth.json")
        if not path.exists():
            pytest.skip("Sample dataset not found")
        errors = validate_dataset(path)
        assert errors == [], f"Validation errors: {errors}"

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_dataset("/tmp/does_not_exist_groundwater.json")

    def test_load_and_write_roundtrip(self, tmp_path):
        data = {
            "name": "Roundtrip",
            "version": "1.0",
            "description": "test",
            "questions": [SAMPLE_QUESTION],
        }
        p = tmp_path / "test.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        ds = load_dataset(p)
        assert ds.name == "Roundtrip"
