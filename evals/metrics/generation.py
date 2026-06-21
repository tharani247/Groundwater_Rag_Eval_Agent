"""
Generation evaluation metrics.

These metrics answer: given what was retrieved, did the model produce a
correct, faithful, and complete answer?

Metrics:
    exact_match          -- strict string match (best for factual short answers)
    semantic_similarity  -- cosine similarity between answer and reference embedding
    fact_coverage        -- fraction of required atomic facts present in answer
    citation_support_check -- checks whether the answer claims unsupported by sources
    rubric_score         -- structured rubric-based score using the EvalQuestion rubric
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Lazy-load sentence transformers to avoid cost on import
_st_model = None


def _get_st_model():
    """Load sentence-transformers model on first use."""
    global _st_model
    if _st_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _st_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Loaded sentence-transformers model: all-MiniLM-L6-v2")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Semantic similarity will return -1.0 as sentinel."
            )
    return _st_model


class GenerationMetrics(BaseModel):
    """All generation metrics for a single eval question run."""

    question_id: str
    exact_match: bool
    semantic_similarity: float = Field(
        ge=-1.0, le=1.0,
        description="-1.0 means sentence-transformers unavailable."
    )
    fact_coverage: float = Field(ge=0.0, le=1.0)
    facts_missing: List[str]
    citation_support_check: bool = Field(
        description="True = no obvious unsupported citation claims detected."
    )
    rubric_score: float = Field(ge=0.0, le=1.0)
    rubric_details: dict


def exact_match(generated: str, expected: str) -> bool:
    """
    Strict normalized exact match.

    Normalizes by lowercasing, stripping punctuation, and collapsing whitespace.
    Best used on short factual answers, not long-form policy responses.
    """
    def normalize(s: str) -> str:
        s = s.lower().strip()
        s = re.sub(r"[^\w\s]", "", s)
        s = re.sub(r"\s+", " ", s)
        return s

    return normalize(generated) == normalize(expected)


def semantic_similarity(
    generated: str,
    reference: str,
) -> float:
    """
    Cosine similarity between sentence embeddings of generated and reference answers.

    Returns float in [-1, 1]. Values above 0.7 indicate strong semantic alignment.
    Returns -1.0 if sentence-transformers is not available (treat as sentinel).
    """
    model = _get_st_model()
    if model is None:
        return -1.0

    try:
        import numpy as np
        embeddings = model.encode([generated, reference], normalize_embeddings=True)
        score = float(np.dot(embeddings[0], embeddings[1]))
        return round(score, 4)
    except Exception as exc:
        logger.warning("semantic_similarity failed: %s", exc)
        return -1.0


def fact_coverage(
    generated_answer: str,
    required_facts: List[str],
) -> Tuple[float, List[str]]:
    """
    Fraction of required atomic facts present in the generated answer.

    Uses simple substring/keyword matching — intentionally naive so results
    are deterministic and debuggable. For production, swap with an LLM-as-judge call.

    Returns:
        (score in [0,1], list of missing facts)
    """
    if not required_facts:
        return 1.0, []

    answer_lower = generated_answer.lower()
    missing: List[str] = []

    for fact in required_facts:
        # Check if key terms from the fact appear in the answer
        fact_terms = [t.strip() for t in re.split(r"[\s,]+", fact.lower()) if len(t.strip()) > 3]
        # A fact is "present" if more than half its key terms appear
        hits = sum(1 for term in fact_terms if term in answer_lower)
        if fact_terms and hits / len(fact_terms) < 0.5:
            missing.append(fact)

    score = (len(required_facts) - len(missing)) / len(required_facts)
    return round(score, 4), missing


def citation_support_check(
    generated_answer: str,
    retrieved_chunk_texts: List[str],
) -> bool:
    """
    Placeholder faithfulness check.

    A production implementation would use an LLM to verify each claim
    in the answer is entailed by at least one retrieved chunk.

    This implementation flags two clear failure signals:
    1. The answer explicitly references a source number not provided (bracketed refs).
    2. The answer is suspiciously long relative to retrieved evidence (>3x ratio).

    Returns True (clean) or False (potential faithfulness issue).
    """
    # Check for inline citation markers that don't match our prompt format
    citation_refs = re.findall(r"\[Source\s+\d+\]|\[\d+\]|\(Source\s+\d+\)", generated_answer)
    if citation_refs:
        logger.debug("Answer contains explicit source refs — format failure: %s", citation_refs)
        return False

    # Check answer length vs retrieved context
    context_len = sum(len(t) for t in retrieved_chunk_texts)
    if context_len > 0 and len(generated_answer) > context_len * 3:
        logger.debug(
            "Answer (%d chars) is >3x retrieved context (%d chars) — possible hallucination",
            len(generated_answer), context_len
        )
        return False

    return True


def rubric_score(
    generated_answer: str,
    rubric: dict,
    retrieved_chunk_texts: Optional[List[str]] = None,
) -> Tuple[float, dict]:
    """
    Score the generated answer against a rubric.

    Each rubric criterion is checked with a heuristic and scored 0 or 1.
    The final score is the fraction of required criteria that pass.

    Args:
        generated_answer: The model's output text.
        rubric: Dict with boolean fields from the EvalQuestion rubric.
        retrieved_chunk_texts: Optional list of chunk texts for context checks.

    Returns:
        (score in [0,1], dict of criterion -> bool)
    """
    answer_lower = generated_answer.lower()
    details: dict = {}

    # Regulatory authority check
    if rubric.get("mentions_regulatory_authority", False):
        authority_terms = ["nrd", "natural resources district", "dnr", "department of natural resources",
                           "nebraska legislature", "board", "commission"]
        details["mentions_regulatory_authority"] = any(t in answer_lower for t in authority_terms)
    else:
        details["mentions_regulatory_authority"] = None  # not required

    # Statute/rule citation check
    if rubric.get("cites_specific_statute_or_rule", False):
        statute_terms = ["chapter 46", "statute", "§", "section", "act", "rule", "regulation", "r.r.s"]
        details["cites_specific_statute_or_rule"] = any(t in answer_lower for t in statute_terms)
    else:
        details["cites_specific_statute_or_rule"] = None

    # Non-hallucination: check answer does not contain bracketed fabricated references
    details["answers_without_hallucination"] = citation_support_check(
        generated_answer,
        retrieved_chunk_texts or [],
    )

    # Actionable guidance check
    if rubric.get("provides_actionable_guidance", False):
        action_terms = ["must", "should", "required", "need to", "apply", "contact", "submit",
                        "file", "register", "obtain", "landowner", "applicant", "you"]
        details["provides_actionable_guidance"] = any(t in answer_lower for t in action_terms)
    else:
        details["provides_actionable_guidance"] = None

    # Limit acknowledgment check
    if rubric.get("acknowledges_limits", False):
        limit_terms = ["not fully", "not clear", "not stated", "sources do not", "limit",
                       "may not cover", "does not address", "unclear", "not available in"]
        details["acknowledges_limits"] = any(t in answer_lower for t in limit_terms)
    else:
        details["acknowledges_limits"] = None

    # Score only required criteria (non-None)
    required = {k: v for k, v in details.items() if v is not None}
    if not required:
        return 1.0, details

    score = sum(1 for v in required.values() if v) / len(required)
    return round(score, 4), details


def compute_generation_metrics(
    question_id: str,
    generated_answer: str,
    expected_answer: str,
    required_facts: List[str],
    rubric: dict,
    retrieved_chunk_texts: Optional[List[str]] = None,
) -> GenerationMetrics:
    """Compute all generation metrics for a single question run."""
    fc_score, missing_facts = fact_coverage(generated_answer, required_facts)
    rs_score, rs_details = rubric_score(generated_answer, rubric, retrieved_chunk_texts)

    return GenerationMetrics(
        question_id=question_id,
        exact_match=exact_match(generated_answer, expected_answer),
        semantic_similarity=semantic_similarity(generated_answer, expected_answer),
        fact_coverage=fc_score,
        facts_missing=missing_facts,
        citation_support_check=citation_support_check(generated_answer, retrieved_chunk_texts or []),
        rubric_score=rs_score,
        rubric_details=rs_details,
    )
