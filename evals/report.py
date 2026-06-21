"""
CLI: python -m evals.report

Generates a Markdown benchmark report from a completed evaluation run.

Examples:
    python -m evals.report --run-id latest
    python -m evals.report --run-id run_20240120_143022_abc123
    python -m evals.report --run-id latest --output my_report.md
"""

from __future__ import annotations

import argparse
import logging
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
        description="Generate a Markdown benchmark report from an eval run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--run-id",
        default="latest",
        help="Run ID to generate a report for (default: latest).",
    )
    parser.add_argument(
        "--log-dir",
        default="evals/logs",
        help="Directory containing eval run logs.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for the Markdown report (default: evals/logs/<run_id>/benchmark_report.md).",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from evals.reports.generator import ReportGenerator

    gen = ReportGenerator(report_dir=Path(args.log_dir))
    try:
        output_path = gen.generate_from_run_id(
            run_id=args.run_id,
        )
        if args.output:
            import shutil
            shutil.copy(output_path, args.output)
            output_path = Path(args.output)

        print(f"\nReport written to: {output_path}")
        print(f"Open it with: open {output_path}  (or any Markdown viewer)\n")
        return 0
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
