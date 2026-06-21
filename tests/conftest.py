"""
Pytest configuration for the Groundwater RAG Evaluation test suite.

Adds the project root to sys.path so evals and src packages resolve correctly.
"""

import sys
from pathlib import Path

# Project root is one level above the tests/ directory
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
