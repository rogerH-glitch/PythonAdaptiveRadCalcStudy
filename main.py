#!/usr/bin/env python3
"""
Main entry point for the radiation view factor validation tool.

This script provides the command-line interface for calculating
view factors between rectangular surfaces using multiple methods.
"""

from src.cli import main  # single source of truth

if __name__ == "__main__":
    raise SystemExit(main())