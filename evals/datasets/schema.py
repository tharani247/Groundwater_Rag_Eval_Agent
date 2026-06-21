"""
Ground truth dataset schema for the Groundwater RAG evaluation harness.

Each EvalQuestion defines a test case: what was asked, what a correct answer
looks like, which sources should be retrieved, and how to score the response.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


class DifficultyLevel(str, Enum):
    easy = "easy"
    medium = "medium"
    hard = "hard"


class QuestionCategory(str, Enum):
    permit_requirements = "permit_requirements"
    well_registration = "well_registration"
    regulatory_authority = "regulatory_authority"
    allocation_limits = "allocation_limits"
    enforcement = "enforcement"
    drought_restrictions = "drought_restrictions"
    legal_definitions = "legal_definitions"
    procedure = "procedure"
    policy_overview = "policy_overview"
    technical = "technical"


class Rubric(BaseModel):
    """Scoring rubric for generation quality. Each criterion is 0-1 normalized."""

    mentions_regulatory_authority: bool = Field(
        description="Answer names the relevant governing body (e.g., NRD, DNR)."
    )
    cites_specific_statute_or_rule: bool = Field(
        description="Answer references a specific law, chapter, or rule number."
    )
    answers_without_hallucination: bool = Field(
        description="Answer does not introduce facts absent from the retrieved sources."
    )
    provides_actionable_guidance: bool = Field(
        description="Answer tells the user what to do or what applies to them."
    )
    acknowledges_limits: bool = Field(
        default=False,
        description="Answer flags what the sources do NOT cover (optional criterion).",
    )


class EvalQuestion(BaseModel):
    """
    A single evaluation test case.

    Fields are intentionally explicit so the runner can score each dimension
    independently: retrieval quality, generation quality, and rubric compliance.
    """

    id: str = Field(description="Unique identifier, e.g. 'GW-001'.")
    question: str = Field(description="The natural language question to pose to the RAG system.")
    expected_answer: str = Field(
        description="A reference answer written from the source documents. Used for semantic similarity."
    )
    expected_sources: List[str] = Field(
        description="List of source URL substrings or doc titles that must appear in retrieved chunks."
    )
    minimum_required_facts: List[str] = Field(
        description="Atomic facts that must appear in a correct answer."
    )
    difficulty: DifficultyLevel
    category: QuestionCategory
    rubric: Rubric
    notes: Optional[str] = Field(default=None, description="Optional evaluator notes.")

    @model_validator(mode="after")
    def check_non_empty(self) -> "EvalQuestion":
        if not self.question.strip():
            raise ValueError("question must not be empty")
        if not self.expected_sources:
            raise ValueError("expected_sources must have at least one entry")
        if not self.minimum_required_facts:
            raise ValueError("minimum_required_facts must have at least one fact")
        return self


class EvalDataset(BaseModel):
    """Container for a collection of evaluation questions."""

    name: str
    version: str = "1.0"
    description: str
    questions: List[EvalQuestion]

    @model_validator(mode="after")
    def check_unique_ids(self) -> "EvalDataset":
        ids = [q.id for q in self.questions]
        if len(ids) != len(set(ids)):
            raise ValueError("All question IDs must be unique")
        return self
