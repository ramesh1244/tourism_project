"""Validate and summarize the tourism package dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "tourism.csv"

EXPECTED_COLUMNS = [
    "CustomerID",
    "ProdTaken",
    "Age",
    "TypeofContact",
    "CityTier",
    "DurationOfPitch",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "ProductPitched",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome",
]


def validate_columns(df: pd.DataFrame) -> None:
    optional_index_columns = {col for col in df.columns if col.startswith("Unnamed:")}
    actual_columns = set(df.columns) - optional_index_columns
    expected_columns = set(EXPECTED_COLUMNS)

    missing_columns = sorted(expected_columns - actual_columns)
    extra_columns = sorted(actual_columns - expected_columns)

    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Dataset validation failed. Missing columns: {missing}")

    if extra_columns:
        extra = ", ".join(extra_columns)
        print(f"Warning: extra columns found and retained for inspection: {extra}")


def print_summary(df: pd.DataFrame, data_path: Path) -> None:
    print("Dataset registration summary")
    print(f"Path: {data_path}")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]:,}")

    if "ProdTaken" in df.columns:
        print("\nTarget distribution:")
        print(df["ProdTaken"].value_counts(dropna=False).sort_index().to_string())

    missing_summary = df.isna().sum()
    missing_summary = missing_summary[missing_summary > 0].sort_values(ascending=False)
    print("\nMissing values:")
    if missing_summary.empty:
        print("No missing values found.")
    else:
        print(missing_summary.to_string())

    print("\nColumn list:")
    for column in df.columns:
        print(f"- {column}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Register the tourism dataset.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to tourism.csv inside the repository.",
    )
    args = parser.parse_args()

    if not args.data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {args.data_path}")

    df = pd.read_csv(args.data_path)
    validate_columns(df)
    print_summary(df, args.data_path)
    print("\nDataset registration completed successfully.")


if __name__ == "__main__":
    main()
