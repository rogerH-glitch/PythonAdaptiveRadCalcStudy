#!/usr/bin/env python3
"""
Main entry point for the radiation view factor validation tool.

This script provides the command-line interface for calculating
view factors between rectangular surfaces using multiple methods.
"""

import sys
from src.cli import main

if __name__ == "__main__":
    # If no arguments provided, show usage and exit with code 2
    if len(sys.argv) == 1:
        print("Usage: python main.py [OPTIONS]")
        print("")
        print("Local Python tool for radiation view factor validation")
        print("")
        print("Examples:")
        print("  python main.py --emitter 5.1 2.1 --setback 1.0 --method adaptive")
        print("  python main.py --emitter 20.02 1.05 --setback 0.81 --method all")
        print("  python main.py --config config.yaml --output results.csv")
        print("")
        print("For detailed help:")
        print("  python main.py --help")
        sys.exit(2)
    
    # Call the main CLI function
    main()