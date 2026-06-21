"""Tests for retrieval and generation metric calculations."""

import pytest

from evals.metrics.retrieval import (
    recall_at_k,
    mean_reciprocal_rank,
    source_coverage,
    detect_missing_sources,
    compute_retrieval_metrics,
)
from evals.metrics.generation import (
    exact_match,
    fact_coverage,
    citation_support_check,
    rubric_score,
    compute_generation_metrics,
)


URLS = [
    "https://www.cpnrd.org/forms-permits/",
    "https://nebraskalegislature.gov/laws/browse-chapters.php",
    "https://dnr.nebraska.gov/water-planning/state-laws-and-rules",
    "https://www.cpnrd.org/water-resources/wells/",
]


class TestRetrievalMetrics:
    def test_recall_all_found(self):
        assert recall_at_k(URLS, ["cpnrd.org", "nebraskalegislature.gov"], k=5) == 1.0

    def test_recall_none_found(self):
        assert recall_at_k(URLS, ["epa.gov", "usgs.gov"], k=5) == 0.0

    def test_recall_partial(self):
        score = recall_at_k(URLS, ["cpnrd.org", "epa.gov"], k=5)
        assert 0.0 < score < 1.0

    def test_recall_respects_k(self):
        # Only first URL is cpnrd.org; with k=0 nothing is in window
        score = recall_at_k(URLS, ["cpnrd.org"], k=0)
        assert score == 0.0

    def test_recall_empty_expected(self):
        assert recall_at_k(URLS, [], k=5) == 1.0

    def test_mrr_first_hit(self):
        mrr = mean_reciprocal_rank(URLS, ["cpnrd.org"])
        assert mrr == pytest.approx(1.0)  # cpnrd.org is URL[0]

    def test_mrr_second_hit(self):
        mrr = mean_reciprocal_rank(URLS, ["nebraskalegislature.gov"])
        assert mrr == pytest.approx(0.5)  # position 2

    def test_mrr_no_hit(self):
        mrr = mean_reciprocal_rank(URLS, ["epa.gov"])
        assert mrr == 0.0

    def test_source_coverage_full(self):
        assert source_coverage(URLS, ["cpnrd.org", "dnr.nebraska.gov"]) == 1.0

    def test_source_coverage_none(self):
        assert source_coverage(URLS, ["epa.gov"]) == 0.0

    def test_detect_missing(self):
        missing = detect_missing_sources(URLS, ["cpnrd.org", "epa.gov"])
        assert missing == ["epa.gov"]

    def test_compute_retrieval_metrics(self):
        chunks = [{"url": u, "score": 0.8 - i * 0.05} for i, u in enumerate(URLS)]
        m = compute_retrieval_metrics("TEST-001", chunks, ["cpnrd.org"], k=5)
        assert m.recall_at_k == 1.0
        assert m.mrr == 1.0
        assert m.chunks_retrieved == 4


class TestGenerationMetrics:
    GOOD_ANSWER = (
        "Natural Resources Districts (NRDs) regulate groundwater in Nebraska "
        "under Chapter 46 of the Nebraska Statutes. The Groundwater Management "
        "and Protection Act grants authority to NRDs to issue permits and enforce "
        "groundwater rules. The Nebraska DNR coordinates integrated management."
    )

    def test_exact_match_true(self):
        assert exact_match("hello world", "hello world") is True

    def test_exact_match_normalized(self):
        assert exact_match("Hello World!", "hello world") is True

    def test_exact_match_false(self):
        assert exact_match("hello world", "goodbye world") is False

    def test_fact_coverage_all_present(self):
        facts = ["NRDs regulate groundwater", "Chapter 46 provides authority"]
        score, missing = fact_coverage(self.GOOD_ANSWER, facts)
        assert score == 1.0
        assert missing == []

    def test_fact_coverage_none_present(self):
        score, missing = fact_coverage("The sky is blue.", ["NRDs regulate groundwater"])
        assert score == 0.0
        assert len(missing) == 1

    def test_fact_coverage_empty_facts(self):
        score, missing = fact_coverage("anything", [])
        assert score == 1.0
        assert missing == []

    def test_citation_support_clean(self):
        assert citation_support_check(self.GOOD_ANSWER, ["Nebraska groundwater is regulated by Natural Resources Districts under the Groundwater Management and Protection Act Chapter 46."]) is True

    def test_citation_support_detects_bracket_refs(self):
        bad = "According to [Source 1], NRDs regulate groundwater."
        assert citation_support_check(bad, ["chunk"]) is False

    def test_rubric_score_authority_present(self):
        score, details = rubric_score(
            self.GOOD_ANSWER,
            {"mentions_regulatory_authority": True, "cites_specific_statute_or_rule": True,
             "answers_without_hallucination": True, "provides_actionable_guidance": False,
             "acknowledges_limits": False},
        )
        assert score > 0.0
        assert details["mentions_regulatory_authority"] is True

    def test_compute_generation_metrics(self):
        m = compute_generation_metrics(
            question_id="TEST-001",
            generated_answer=self.GOOD_ANSWER,
            expected_answer=self.GOOD_ANSWER,
            required_facts=["NRDs regulate groundwater"],
            rubric={"mentions_regulatory_authority": True, "cites_specific_statute_or_rule": False,
                    "answers_without_hallucination": True, "provides_actionable_guidance": False,
                    "acknowledges_limits": False},
        )
        assert m.exact_match is True
        assert m.fact_coverage > 0.0
