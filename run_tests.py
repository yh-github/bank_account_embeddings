#!/usr/bin/env python3
"""Test runner for hierarchical embeddings package.

Usage:
    python run_tests.py
    
Or with pytest directly (recommended):
    python -m pytest tests/ -v
"""

import subprocess
import sys


def main() -> int:
    """Run the test suite using pytest."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        cwd=None,  # Run from current directory
    )
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
