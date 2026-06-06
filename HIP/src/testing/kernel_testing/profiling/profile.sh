#!/bin/bash

# Usage: ./profile.sh <yaml_file> <heat_test_args...>
# Example: ./profile.sh yaml_files/basic.yaml synthetic timing config_files/stride.txt 10000 20000

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 <yaml_file> <heat_test_args...>"
    echo "Example: $0 yaml_files/basic.yaml synthetic timing config_files/stride.txt 10000 20000"
    exit 1
fi

YAML_FILE="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTING_DIR="$(dirname "$SCRIPT_DIR")"
KERNEL_TEST="$TESTING_DIR/kernel_test"

# Resolve yaml_file to absolute path from the caller's working directory
if [[ "$YAML_FILE" != /* ]]; then
    YAML_FILE="$(pwd)/profiling/yaml_files/$YAML_FILE"
fi

# Run from testing/ so relative paths like config_files/stride.txt resolve correctly
cd "$TESTING_DIR"

echo $TESTING_DIR

rocprofv3 -i "$YAML_FILE" -- "$KERNEL_TEST" "$@"

cd "$SCRIPT_DIR"
python3 csv_cleanup/extract_counters.py \
    --passes "$TESTING_DIR"/pass_*/out_counter_collection.csv \
    --yaml "$YAML_FILE"

rm -rf "$TESTING_DIR"/pass_*/
rm -rf "$TESTING_DIR"/.rocprofv3/

echo "Results: $SCRIPT_DIR/output.csv"
