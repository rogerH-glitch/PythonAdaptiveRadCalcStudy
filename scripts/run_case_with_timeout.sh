#!/bin/bash
# scripts/run_case_with_timeout.sh
# Unix shell wrapper for eng/run_with_timeout.py
# Usage: ./scripts/run_case_with_timeout.sh --timeout 5.0 --payload '{"method":"adaptive",...}'

set -e

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TIMEOUT_RUNNER="$PROJECT_ROOT/eng/run_with_timeout.py"

# Check if the timeout runner exists
if [ ! -f "$TIMEOUT_RUNNER" ]; then
    echo "Error: Timeout runner not found at: $TIMEOUT_RUNNER" >&2
    exit 1
fi

# Check if python is available
if ! command -v python &> /dev/null; then
    echo "Error: python command not found" >&2
    exit 1
fi

# Run the timeout wrapper with all passed arguments
python "$TIMEOUT_RUNNER" "$@"
