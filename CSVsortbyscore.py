"""Utility functions for sorting graded abstract CSV files by score."""

from pathlib import Path
import argparse
import csv


def sort_csv_by_score(csv_file_path: Path, output_file_path: Path) -> None:
    """Sort *csv_file_path* by the ``SCORE`` column and write to *output_file_path*.

    The parent directory of *output_file_path* is created automatically if it
    does not exist.
    """

    with csv_file_path.open(mode="r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    sorted_rows = sorted(rows, key=lambda row: float(row["SCORE"]), reverse=True)

    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    with output_file_path.open(mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(sorted_rows)

    print(f"Sorted CSV has been saved to {output_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sort a graded CSV by score.")
    parser.add_argument("--input", type=Path, required=True, help="Input CSV file")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV file")

    args = parser.parse_args()
    sort_csv_by_score(args.input, args.output)

