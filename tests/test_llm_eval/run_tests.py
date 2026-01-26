#!/usr/bin/env python3
"""Test runner for ollama_lib tests.

Usage:
    python run_tests.py              # Run all tests
    python run_tests.py -v           # Run with verbose output
    python run_tests.py -k thinking  # Run only tests matching 'thinking'
"""
import sys
import os
import pytest

# Add the project root to Python path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, project_root)


def main():
    """Run the tests with pytest."""
    # Default arguments
    args = [
        os.path.dirname(__file__),  # Test directory
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "--strict-markers",  # Strict marker handling
    ]

    # Add any command line arguments
    if len(sys.argv) > 1:
        args.extend(sys.argv[1:])

    # Run tests
    exit_code = pytest.main(args)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
