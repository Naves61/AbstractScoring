"""Utilities for scoring scientific abstracts.

The module provides functions to preprocess text, calculate scores based on a
weighted keyword dictionary and semantic similarity, and to write the results to
CSV files.  Paths are handled using :mod:`pathlib` so directories are created
automatically if they do not exist.

The module expects a configuration JSON file with the following minimal
structure::

    {
        "target_description": "text describing the target topic",
        "weights": {
            "semantic": 0.5,
            "keyword": 0.4,
            "cohort": 0.1
        },
        "keyword_weights": {"\\bexample\\b": 5},
        "cohort_bonus": 10,
        "case_penalty": -10
    }

Install the required spaCy model before running the script::

    pip install scispacy
    python -m spacy download en_core_sci_sm
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import argparse
import json
import re

import numpy as np
import pandas as pd
import spacy
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# -----------------------
# Model initialisation
# -----------------------
nlp = spacy.load("en_core_sci_sm")
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")


def load_config(path: Path) -> Dict[str, Any]:
    """Load configuration from *path*.

    Parameters
    ----------
    path:
        Location of the JSON configuration file.
    """

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def preprocess(text: str) -> str:
    """Lower-case, remove punctuation and lemmatise *text* using spaCy."""

    if pd.isna(text):
        return ""
    text = re.sub(r"[^\w\s-]", "", text.lower())
    doc = nlp(text)
    return " ".join(token.lemma_ for token in doc if not token.is_stop)


def calculate_simple_score(text: str, config: Dict[str, Any]) -> float:
    """Calculate score using keyword matches only."""

    if pd.isna(text) or not text.strip():
        return 0.0

    cleaned_text = preprocess(text)
    if not cleaned_text.strip():
        return 0.0

    keyword_score = sum(
        weight * len(re.findall(pattern, text, flags=re.IGNORECASE))
        for pattern, weight in config["keyword_weights"].items()
    )

    final_score = config["weights"]["keyword"] * keyword_score
    return float(np.clip(final_score, 0, 100))


def calculate_score(text: str, config: Dict[str, Any], target_embedding: np.ndarray) -> float:
    """Calculate the full score for *text* given *config* and *target_embedding*."""

    if pd.isna(text) or not text.strip():
        return 0.0

    cleaned_text = preprocess(text)
    if not cleaned_text.strip():
        return 0.0

    text_embedding = sentence_model.encode([cleaned_text])[0]
    semantic_score = 1 - cosine(target_embedding, text_embedding)

    keyword_score = sum(
        weight * len(re.findall(pattern, text, flags=re.IGNORECASE))
        for pattern, weight in config["keyword_weights"].items()
    )

    cohort_score = (
        config["cohort_bonus"]
        if re.search(r"\b(n=\d+|patients\s+\d+|\d+\s+cases)\b", text)
        else 0
    )
    penalty = (
        config["case_penalty"]
        if re.search(r"\b(case report|single case)\b", text, flags=re.IGNORECASE)
        else 0
    )

    final_score = (
        config["weights"]["semantic"] * semantic_score * 100
        + config["weights"]["keyword"] * keyword_score
        + config["weights"]["cohort"] * cohort_score
        + penalty
    )

    return float(np.clip(final_score, 0, 100))


def TQDMScoreCalc(
    df: pd.DataFrame,
    config: Dict[str, Any],
    target_embedding: np.ndarray,
    abstract_column: str = "ABSTRACT",
) -> pd.Series:
    """Calculate scores for all abstracts in *df* with a progress bar."""

    tqdm.pandas(desc="Grading abstracts")
    return df[abstract_column].progress_apply(
        lambda x: calculate_score(x, config=config, target_embedding=target_embedding)
    )


def score_articles(
    input_file_path: Path,
    output_file_path: Path,
    delimiter: str,
    config_path: Path = Path("config.json"),
    abstract_column: str = "ABSTRACT",
) -> None:
    """Score articles from *input_file_path* and write results to *output_file_path*.

    The parent directory of *output_file_path* is created if it does not yet
    exist.  The configuration file determines the keyword weights and other
    scoring parameters.
    """

    config = load_config(config_path)
    target_embedding = sentence_model.encode([config["target_description"]])[0]

    df = pd.read_csv(input_file_path, delimiter=delimiter)
    df["SCORE"] = TQDMScoreCalc(
        df, config=config, target_embedding=target_embedding, abstract_column=abstract_column
    )
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file_path, index=False)
    print("Processing completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score abstracts in a CSV file.")
    parser.add_argument("--input", type=Path, required=True, help="Input CSV file")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV file")
    parser.add_argument(
        "--config", type=Path, default=Path("config.json"), help="Configuration JSON file"
    )
    parser.add_argument(
        "--delimiter", type=str, default=",", help="CSV delimiter (default: ',')"
    )
    parser.add_argument(
        "--abstract-column",
        type=str,
        default="ABSTRACT",
        help="Name of the column containing abstracts",
    )

    args = parser.parse_args()
    score_articles(
        input_file_path=args.input,
        output_file_path=args.output,
        delimiter=args.delimiter,
        config_path=args.config,
        abstract_column=args.abstract_column,
    )

