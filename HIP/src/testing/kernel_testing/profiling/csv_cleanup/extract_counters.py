#!/usr/bin/env python3
"""Extract counter values from rocprofv3 pass CSVs into output.csv."""

import csv
import re
import sys
import argparse
from collections import defaultdict


def parse_counters_from_yaml(yaml_path):
    """Extract all counter names from pmc lists in the yaml file, preserving order."""
    counters = []
    seen = set()
    with open(yaml_path) as f:
        content = f.read()
    for match in re.finditer(r'pmc\s*:\s*\[([^\]]+)\]', content):
        for name in re.findall(r'"([^"]+)"', match.group(1)):
            if name not in seen:
                counters.append(name)
                seen.add(name)
    return counters


def extract(pass_csvs, counters):
    """Read pass CSVs and return {corr_id: {counter: value}} mapping."""
    counter_set = set(counters)
    dispatches = defaultdict(dict)
    for path in pass_csvs:
        try:
            with open(path, newline="") as f:
                for row in csv.DictReader(f):
                    corr_id = row.get("Correlation_Id")
                    counter_name = row.get("Counter_Name")
                    if corr_id and counter_name in counter_set:
                        dispatches[corr_id][counter_name] = row.get("Counter_Value", "0")
        except FileNotFoundError:
            print(f"Warning: {path} not found.", file=sys.stderr)
    return dispatches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--passes", nargs="+", required=True, help="Paths to pass counter CSVs")
    parser.add_argument("--yaml", required=True, help="Path to rocprofv3 yaml file")
    args = parser.parse_args()

    counters = parse_counters_from_yaml(args.yaml)
    if not counters:
        print("Error: no counters found in yaml.", file=sys.stderr)
        sys.exit(1)

    dispatches = extract(args.passes, counters)
    if not dispatches:
        print("Error: no counter data found in pass CSVs.", file=sys.stderr)
        sys.exit(1)

    output_path = "output.csv"
    headers = ["Correlation_Id"] + counters

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for corr_id in sorted(dispatches, key=lambda x: int(x)):
            row = {"Correlation_Id": corr_id}
            row.update({c: dispatches[corr_id].get(c, "0") for c in counters})
            writer.writerow(row)

    print(f"Written {len(dispatches)} row(s) to {output_path}")


if __name__ == "__main__":
    main()
