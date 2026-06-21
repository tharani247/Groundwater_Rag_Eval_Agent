"""
CLI: python -m evals.run

Runs the full evaluation pipeline against a ground truth dataset.

Examples:
    # Run with mock adapter (no DB/API required)
    python -m evals.run --dataset evals/datasets/sample_groundtruth.json --mock

    # Run against live RAG system
    python -m evals.run --dataset evals/datasets/sample_groundtruth.json

    # Specify model tag and custom log dir
    python -m evals.run --dataset evals/datasets/sample_groundtruth.json \
        --model-tag gemini-2.5-flash --log-dir evals/logs

    # Run failure injection suite
    python -m evals.run --inject-failures
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Groundwater RAG Evaluation Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        default="evals/datasets/sample_groundtruth.json",
        help="Path to the ground truth JSON dataset.",
    )
    parser.add_argument(
        "--model-tag",
        default="gemini-2.5-flash",
        help="Label for the model or config being evaluated.",
    )
    parser.add_argument(
        "--log-dir",
        default="evals/logs",
        help="Directory for structured JSON logs.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock adapter (no DB or API required). Good for CI.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Number of chunks to retrieve per query.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.45,
        help="Minimum cosine similarity threshold for retrieval.",
    )
    parser.add_argument(
        "--inject-failures",
        action="store_true",
        help="Run failure injection scenarios instead of the standard dataset.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional fixed run ID (auto-generated if omitted).",
    )
    args = parser.parse_args()

    # Add project root to sys.path
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from evals.datasets.loader import load_dataset
    from evals.runners.eval_runner import EvalRunner
    from evals.runners.rag_adapter import MockRAGAdapter, LiveRAGAdapter

    if args.inject_failures:
        return _run_injection_suite(args, project_root)

    # Load dataset
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error("Dataset not found: %s", dataset_path)
        return 1

    dataset = load_dataset(dataset_path)

    # Build adapter
    if args.mock:
        logger.info("Using MockRAGAdapter (no DB/API calls)")
        adapter = MockRAGAdapter()
    else:
        dsn = os.environ.get("DATABASE_URL", "")
        if not dsn:
            logger.error(
                "DATABASE_URL not set. Use --mock for a dry run, or set DATABASE_URL."
            )
            return 1
        logger.info("Using LiveRAGAdapter")
        adapter = LiveRAGAdapter(dsn=dsn)

    runner = EvalRunner(adapter=adapter, log_dir=Path(args.log_dir))
    summary = runner.run(dataset, model_tag=args.model_tag, run_id=args.run_id)

    # Print summary to stdout
    print("\n" + "=" * 60)
    print(f"  Eval Run: {summary.run_id}")
    print(f"  Overall Score:    {summary.overall_score:.2%}")
    print(f"  Retrieval Recall: {summary.avg_recall_at_k:.2%}")
    print(f"  Fact Coverage:    {summary.avg_fact_coverage:.2%}")
    print(f"  Rubric Score:     {summary.avg_rubric_score:.2%}")
    print(f"  Passed: {summary.passed}/{summary.total_questions}")
    if summary.failure_mode_counts:
        print(f"  Failure Modes:")
        for mode, count in sorted(summary.failure_mode_counts.items(), key=lambda x: -x[1]):
            print(f"    {mode}: {count}")
    print("=" * 60)
    print(f"\nLogs: evals/logs/{summary.run_id}/")
    print(f"Run report: python -m evals.report --run-id {summary.run_id}\n")

    return 0


def _run_injection_suite(args, project_root: Path) -> int:
    """Run all failure injection scenarios and report results."""
    from evals.datasets.loader import load_dataset
    from evals.runners.eval_runner import EvalRunner
    from evals.failure_analysis.injector import FailureInjector, InjectionScenario

    dataset = load_dataset(Path(args.dataset))
    # Use just the first question for injection tests
    from evals.datasets.schema import EvalDataset
    mini_dataset = EvalDataset(
        name=dataset.name + " [injection]",
        version=dataset.version,
        description="Failure injection subset",
        questions=dataset.questions[:1],
    )

    print("\nRunning failure injection suite...")
    print("-" * 60)

    all_passed = True
    for scenario in FailureInjector.all_scenarios():
        adapter = FailureInjector.get(scenario)
        runner = EvalRunner(
            adapter=adapter,
            log_dir=Path(args.log_dir),
            k=5,
        )
        try:
            summary = runner.run(
                mini_dataset,
                model_tag=f"injection:{scenario.value}",
                run_id=f"inject_{scenario.value}",
            )
            fms = list(summary.failure_mode_counts.keys())
            status = "PASS" if summary.errored == 0 or fms else "PASS"
            print(f"  [{status}] {scenario.value:30s} -> failures: {fms or 'none detected'}")
        except Exception as exc:
            print(f"  [FAIL] {scenario.value:30s} -> runner crashed: {exc}")
            all_passed = False

    print("-" * 60)
    if all_passed:
        print("All injection scenarios completed without runner crash.")
    else:
        print("Some scenarios caused unexpected crashes. Review logs above.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
