"""CLI: compare two CSVs column by column and print a drift report.

Usage:
    python run.py reference.csv current.csv [--bins 10]
"""
from __future__ import annotations

import argparse

import pandas as pd

from drift import drift_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Numeric data-drift report between two CSVs")
    parser.add_argument("reference_csv")
    parser.add_argument("current_csv")
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()

    reference = pd.read_csv(args.reference_csv)
    current = pd.read_csv(args.current_csv)
    print(drift_report(reference, current, bins=args.bins).to_string())


if __name__ == "__main__":
    main()
