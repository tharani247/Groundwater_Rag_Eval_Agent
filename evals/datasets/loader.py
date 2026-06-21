"""Dataset loading and validation helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

from .schema import EvalDataset, EvalQuestion

logger = logging.getLogger(__name__)


def load_dataset(path: str | Path) -> EvalDataset:
    """Load and validate an eval dataset from a JSON file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    dataset = EvalDataset.model_validate(raw)
    logger.info("Loaded dataset '%s' with %d questions", dataset.name, len(dataset.questions))
    return dataset


def validate_dataset(path: str | Path) -> List[str]:
    """
    Validate a dataset file without running an evaluation.
    Returns a list of validation errors (empty list means valid).
    """
    errors: List[str] = []
    try:
        dataset = load_dataset(path)
        for q in dataset.questions:
            if len(q.minimum_required_facts) < 1:
                errors.append(f"{q.id}: minimum_required_facts is empty")
            if len(q.expected_sources) < 1:
                errors.append(f"{q.id}: expected_sources is empty")
    except Exception as exc:
        errors.append(str(exc))
    return errors
