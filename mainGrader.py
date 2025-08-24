"""Main entry point for grading and sorting scientific abstracts.

This module ties together the grading and sorting utilities. The paths for
input and output files are fully parameterised so the script can be reused in
different projects.  If the expected directory structure does not exist it is
created automatically.

Example directory layout expected by default::

    Articles/PM/Scored_articles/
    Articles/PM/Graded_articles/
    Articles/PM/Graded_sorted_articles/

Adjust the ``--database`` and ``--base-dir`` parameters to match your setup.
"""

from pathlib import Path
import argparse

import CSVgrader
import CSVsortbyscore


def main(
    current_iteration: int,
    database: str,
    delimiter: str,
    base_dir: Path,
    config_path: Path,
) -> None:
    """Run the grading pipeline for a given iteration.

    Parameters
    ----------
    current_iteration:
        Index of the current iteration. The graded output will use
        ``current_iteration + 1``.
    database:
        Name of the database subdirectory (e.g. ``"PM"`` or ``"EMBASE"``).
    delimiter:
        CSV delimiter used by the input and output files.
    base_dir:
        Root directory containing the ``Articles`` folder.  This directory and
        its children are created if they do not exist.
    config_path:
        Path to the configuration JSON file used by :mod:`CSVgrader`.
    """

    input_path = (
        base_dir
        / database
        / "Scored_articles"
        / f"scored_graded_articles{current_iteration}.csv"
    )

    output_path = (
        base_dir
        / database
        / "Graded_articles"
        / f"graded_articles{current_iteration + 1}.csv"
    )

    output_path_sorted = (
        base_dir
        / database
        / "Graded_sorted_articles"
        / f"graded_sorted_articles{current_iteration + 1}.csv"
    )

    print(f"Reading input file: {input_path}")
    print(f"Setting output file: {output_path}")

    # Ensure output directories exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path_sorted.parent.mkdir(parents=True, exist_ok=True)

    # Grade CSV
    CSVgrader.score_articles(
        input_file_path=input_path,
        output_file_path=output_path,
        delimiter=delimiter,
        config_path=config_path,
    )

    # Sort CSV
    CSVsortbyscore.sort_csv_by_score(output_path, output_path_sorted)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grade and sort abstracts.")
    parser.add_argument(
        "--current-iteration",
        type=int,
        default=1,
        help="Current grading iteration (default: 1).",
    )
    parser.add_argument(
        "--database",
        type=str,
        default="PM",
        help="Database subdirectory name (default: 'PM').",
    )
    parser.add_argument(
        "--delimiter",
        type=str,
        default=",",
        help="CSV delimiter used in the files (default: ',').",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("Articles"),
        help="Base directory that contains the article folders.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.json"),
        help="Path to the configuration JSON file.",
    )

    args = parser.parse_args()
    main(
        current_iteration=args.current_iteration,
        database=args.database,
        delimiter=args.delimiter,
        base_dir=args.base_dir,
        config_path=args.config,
    )

