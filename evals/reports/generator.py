"""
Benchmark report generator.

Reads a run summary JSON and writes a structured Markdown report.
The report is designed to be readable by both engineers and non-technical
stakeholders — every metric has a plain-English interpretation.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..runners.eval_runner import EvalRunSummary, RunResult

logger = logging.getLogger(__name__)

REPORT_DIR = Path(__file__).parent.parent / "logs"


def _score_label(score: float) -> str:
    if score >= 0.85:
        return "Excellent"
    if score >= 0.70:
        return "Good"
    if score >= 0.50:
        return "Fair"
    return "Needs Work"


def _fmt(score: float) -> str:
    return f"{score:.2%}"


class ReportGenerator:
    """
    Generates Markdown benchmark reports from EvalRunSummary objects.

    Usage:
        summary = EvalRunner(...).run(dataset)
        report_path = ReportGenerator().generate(summary)
    """

    def __init__(self, report_dir: Optional[Path | str] = None):
        self._dir = Path(report_dir) if report_dir else REPORT_DIR

    def generate(self, summary: EvalRunSummary, output_path: Optional[Path | str] = None) -> Path:
        """Write a Markdown report and return the path."""
        if output_path is None:
            run_dir = self._dir / summary.run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            output_path = run_dir / "benchmark_report.md"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        md = self._build_report(summary)
        output_path.write_text(md, encoding="utf-8")
        logger.info("Report written to %s", output_path)
        return output_path

    def generate_from_run_id(self, run_id: str) -> Path:
        """Load a summary JSON by run_id and generate a report."""
        if run_id == "latest":
            latest_path = self._dir / "latest"
            if not latest_path.exists():
                raise FileNotFoundError("No 'latest' run found. Run an evaluation first.")
            run_id = latest_path.read_text(encoding="utf-8").strip()

        summary_path = self._dir / run_id / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Summary not found for run_id '{run_id}': {summary_path}")

        with summary_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        summary = EvalRunSummary.model_validate(raw)
        return self.generate(summary)

    def _build_report(self, s: EvalRunSummary) -> str:
        lines = []

        # Header
        lines += [
            f"# Groundwater RAG Evaluation Report",
            f"",
            f"**Run ID:** `{s.run_id}`  ",
            f"**Model:** `{s.model_tag}`  ",
            f"**Dataset:** {s.dataset_name}  ",
            f"**Started:** {s.started_at_utc}  ",
            f"**Completed:** {s.completed_at_utc}  ",
            f"",
        ]

        # Overall Score
        retrieval_score = (s.avg_recall_at_k + s.avg_mrr + s.avg_source_coverage) / 3
        generation_score = (s.avg_fact_coverage + s.avg_rubric_score) / 2
        overall = s.overall_score

        lines += [
            "## Overall Score",
            "",
            f"| Dimension | Score | Rating |",
            f"|-----------|-------|--------|",
            f"| **Overall** | {_fmt(overall)} | {_score_label(overall)} |",
            f"| Retrieval | {_fmt(retrieval_score)} | {_score_label(retrieval_score)} |",
            f"| Generation | {_fmt(generation_score)} | {_score_label(generation_score)} |",
            "",
            f"**{s.passed}/{s.total_questions}** questions passed end-to-end.  ",
            f"**{s.failed}** failed. **{s.errored}** errored (infrastructure failures).",
            "",
        ]

        # Retrieval Metrics
        lines += [
            "## Retrieval Metrics",
            "",
            "These metrics measure whether the system retrieved the right chunks",
            "before the model ever saw them. Poor retrieval cannot be compensated",
            "by a better model — garbage in, garbage out.",
            "",
            f"| Metric | Score | Interpretation |",
            f"|--------|-------|----------------|",
            f"| Recall@{s.avg_recall_at_k:.0%} (k=5) | {_fmt(s.avg_recall_at_k)} | Fraction of expected sources found in top 5 chunks |",
            f"| MRR | {_fmt(s.avg_mrr)} | How high the first relevant chunk ranked on average |",
            f"| Source Coverage | {_fmt(s.avg_source_coverage)} | Fraction of expected sources found anywhere in retrieval |",
            f"| Avg Top Score | N/A | See per-question logs |",
            "",
        ]

        # Generation Metrics
        lines += [
            "## Generation Metrics",
            "",
            "These metrics measure whether the model produced a correct, faithful,",
            "and complete answer given what was retrieved.",
            "",
            f"| Metric | Score | Interpretation |",
            f"|--------|-------|----------------|",
        ]
        sem_sim = s.avg_semantic_similarity
        sem_label = f"{_fmt(sem_sim)}" if sem_sim >= 0 else "N/A (model unavailable)"
        lines += [
            f"| Semantic Similarity | {sem_label} | Cosine sim to reference answer |",
            f"| Fact Coverage | {_fmt(s.avg_fact_coverage)} | Fraction of required facts present in answer |",
            f"| Rubric Score | {_fmt(s.avg_rubric_score)} | Structured rubric pass rate |",
            f"| Avg Latency | {s.avg_latency_seconds:.2f}s | Wall-clock time per query |",
            "",
        ]

        # Failure Mode Breakdown
        lines += [
            "## Failure Mode Breakdown",
            "",
            "| Failure Mode | Count | % of Questions |",
            "|--------------|-------|----------------|",
        ]
        total = s.total_questions or 1
        for mode, count in sorted(s.failure_mode_counts.items(), key=lambda x: -x[1]):
            pct = count / total * 100
            lines.append(f"| `{mode}` | {count} | {pct:.1f}% |")

        if not s.failure_mode_counts:
            lines.append("| — | 0 | 0% |")

        lines += [""]

        # Failure mode explanations
        explanations = {
            "retrieval_miss": "Expected source documents were not retrieved. Check corpus coverage and embedding quality.",
            "wrong_chunk_retrieved": "Chunks were retrieved but from irrelevant documents. Review similarity thresholds and chunk boundaries.",
            "citation_hallucination": "Answer contains inline citations not matching prompt format, or answer is suspiciously longer than retrieved context.",
            "answer_not_supported": "Answer makes claims with zero coverage of required facts. May indicate hallucination.",
            "partial_answer": "Answer partially covers required facts but misses key ones. Often a generation length or prompt issue.",
            "format_failure": "Answer does not follow the required five-section output structure.",
            "timeout_or_tool_failure": "The pipeline errored at infrastructure level. Check API keys, DB connection, and timeouts.",
            "lost_in_context": "Retrieval was good but the model ignored relevant chunks. Review prompt construction.",
            "unknown": "Rubric or similarity score was low but no specific failure pattern matched. Inspect manually.",
        }
        if s.failure_mode_counts:
            lines += ["### Failure Mode Descriptions", ""]
            for mode in s.failure_mode_counts:
                if mode in explanations:
                    lines.append(f"**`{mode}`** — {explanations[mode]}")
                    lines.append("")

        # Per-category breakdown
        from collections import defaultdict
        category_scores: dict = defaultdict(list)
        for r in s.results:
            combined = (r.retrieval_metrics.recall_at_k + r.generation_metrics.fact_coverage) / 2
            category_scores[r.category].append(combined)

        lines += [
            "## Score by Category",
            "",
            "| Category | Avg Score | Questions |",
            "|----------|-----------|-----------|",
        ]
        for cat, scores in sorted(category_scores.items(), key=lambda x: sum(x[1]) / len(x[1])):
            avg = sum(scores) / len(scores)
            lines.append(f"| {cat} | {_fmt(avg)} | {len(scores)} |")
        lines += [""]

        # Worst performers
        worst = sorted(s.results, key=lambda r: r.generation_metrics.fact_coverage)[:5]
        lines += [
            "## Worst Performing Questions",
            "",
            "These questions had the lowest fact coverage and are the highest-priority",
            "candidates for prompt tuning, corpus expansion, or retrieval threshold adjustment.",
            "",
        ]
        for r in worst:
            modes = ", ".join(f.value for f in r.failure_modes) or "none"
            lines += [
                f"### {r.question_id} ({r.category} / {r.difficulty})",
                f"**Question:** {r.question}",
                "",
                f"- Fact coverage: {_fmt(r.generation_metrics.fact_coverage)}",
                f"- Recall@k: {_fmt(r.retrieval_metrics.recall_at_k)}",
                f"- Rubric score: {_fmt(r.generation_metrics.rubric_score)}",
                f"- Failure modes: `{modes}`",
                "",
            ]
            if r.generation_metrics.facts_missing:
                lines.append("**Missing facts:**")
                for fact in r.generation_metrics.facts_missing:
                    lines.append(f"  - {fact}")
                lines.append("")
            if r.retrieval_metrics.missing_sources:
                lines.append("**Missing sources:**")
                for src in r.retrieval_metrics.missing_sources:
                    lines.append(f"  - `{src}`")
                lines.append("")

        # Recommendations
        lines += ["## Recommendations", ""]
        recs = self._build_recommendations(s)
        for i, rec in enumerate(recs, start=1):
            lines.append(f"{i}. {rec}")
        lines += [""]

        # Footer
        lines += [
            "---",
            f"*Generated by the Groundwater RAG Evaluation Agent. Run ID: `{s.run_id}`*",
        ]

        return "\n".join(lines)

    def _build_recommendations(self, s: EvalRunSummary) -> list[str]:
        recs = []
        fm = s.failure_mode_counts

        if s.avg_recall_at_k < 0.6:
            recs.append(
                "Retrieval recall is below 0.60. Consider lowering the `min_score` threshold, "
                "expanding multi-query variants, or re-chunking documents at finer granularity."
            )
        if s.avg_mrr < 0.5:
            recs.append(
                "MRR is low — relevant chunks are not ranking highly. "
                "Review the embedding model choice and whether the query expansion covers the right terminology."
            )
        if s.avg_fact_coverage < 0.6:
            recs.append(
                "Fact coverage is below 0.60. The model may be producing generic answers. "
                "Tighten the prompt to require explicit reference to retrieved content."
            )
        if fm.get("citation_hallucination", 0) > 0:
            recs.append(
                f"{fm['citation_hallucination']} answer(s) had citation format issues. "
                "The prompt rule prohibiting [Source N] references may need stronger enforcement "
                "or a post-processing filter."
            )
        if fm.get("format_failure", 0) > 0:
            recs.append(
                f"{fm['format_failure']} answer(s) failed the required five-section format check. "
                "Add an answer completeness check to the pipeline (similar to the existing "
                "`answer_looks_complete()` function in retrieve_gemini.py)."
            )
        if fm.get("timeout_or_tool_failure", 0) > 0:
            recs.append(
                "Infrastructure failures occurred. Check API key validity, DB connection stability, "
                "and add retry logic with exponential backoff for Gemini API calls."
            )
        if fm.get("lost_in_context", 0) > 0:
            recs.append(
                "Some questions had good retrieval but poor generation — the model is losing track "
                "of retrieved content. Reduce chunks per prompt, or experiment with re-ranking "
                "to ensure the most relevant chunk is first."
            )
        if not recs:
            recs.append(
                "All major metrics are in good shape. Focus next on expanding the ground truth "
                "dataset to cover edge cases in drought restrictions and allocation limits, "
                "then benchmark with a harder model (e.g., gemini-2.5-pro)."
            )
        return recs
