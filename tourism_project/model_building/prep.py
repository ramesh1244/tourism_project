"""Clean the tourism dataset and create reproducible train/test splits."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "tourism.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "data"
TARGET_COLUMN = "ProdTaken"
RANDOM_STATE = 42


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    unnamed_columns = [column for column in df.columns if column.startswith("Unnamed:")]
    df = df.drop(columns=unnamed_columns, errors="ignore")
    df = df.drop(columns=["CustomerID"], errors="ignore")

    text_columns = df.select_dtypes(include=["object"]).columns
    for column in text_columns:
        df[column] = df[column].astype(str).str.strip()

    replacements = {
        "Gender": {"Fe Male": "Female"},
        "Occupation": {"Free Lancer": "Freelancer"},
        "MaritalStatus": {"Unmarried": "Single"},
        "TypeofContact": {"Self Inquiry": "Self Enquiry"},
    }
    for column, mapping in replacements.items():
        if column in df.columns:
            df[column] = df[column].replace(mapping)

    df = df.drop_duplicates()
    df = df.dropna(subset=[TARGET_COLUMN])
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    return df


def write_splits(train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(output_dir / "train.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    train_df.drop(columns=[TARGET_COLUMN]).to_csv(output_dir / "Xtrain.csv", index=False)
    test_df.drop(columns=[TARGET_COLUMN]).to_csv(output_dir / "Xtest.csv", index=False)
    train_df[[TARGET_COLUMN]].to_csv(output_dir / "ytrain.csv", index=False)
    test_df[[TARGET_COLUMN]].to_csv(output_dir / "ytest.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare train/test data.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    raw_df = pd.read_csv(args.data_path)
    clean_df = clean_data(raw_df)

    train_df, test_df = train_test_split(
        clean_df,
        test_size=args.test_size,
        random_state=RANDOM_STATE,
        stratify=clean_df[TARGET_COLUMN],
    )
    write_splits(train_df, test_df, args.output_dir)

    print("Data preparation completed successfully.")
    print(f"Cleaned rows: {clean_df.shape[0]:,}")
    print(f"Training rows: {train_df.shape[0]:,}")
    print(f"Testing rows: {test_df.shape[0]:,}")
    print(f"Saved splits to: {args.output_dir}")


if __name__ == "__main__":
    main()
