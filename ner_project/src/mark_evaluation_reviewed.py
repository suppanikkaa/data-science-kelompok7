"""
Record human sign-off for the evaluation set.

Run this only after a person has checked every excerpt and span in
``data/evaluation_examples.json``.  The script refuses to mark the set as
human-reviewed without a reviewer name and date, then validates every offset
and overlap before writing the file.

Example:
    python ner_project/src/mark_evaluation_reviewed.py --reviewer "Nama Reviewer"
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from evaluation import validate_eval_examples

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PATH = os.path.join(PROJECT_DIR, "data", "evaluation_examples.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", default=DEFAULT_PATH, help="Evaluation JSON file")
    parser.add_argument("--reviewer", required=True, help="Name of the human reviewer")
    parser.add_argument(
        "--date",
        default=dt.date.today().isoformat(),
        help="Review date in YYYY-MM-DD format (default: today)",
    )
    args = parser.parse_args()

    reviewer = args.reviewer.strip()
    if not reviewer:
        raise SystemExit("Reviewer name must not be empty")

    with open(args.examples, "r", encoding="utf-8-sig") as file_handle:
        examples = json.load(file_handle)

    if not isinstance(examples, list) or not examples:
        raise SystemExit("Evaluation file must contain a non-empty JSON list")

    for example in examples:
        example["review_status"] = "human-reviewed"
        example["reviewer"] = reviewer
        example["reviewed_at"] = args.date

    validate_eval_examples(examples)
    with open(args.examples, "w", encoding="utf-8") as file_handle:
        json.dump(examples, file_handle, indent=2, ensure_ascii=False)

    print(
        f"Marked {len(examples)} examples as human-reviewed by {reviewer} "
        f"on {args.date}."
    )
    print("Now rerun: python ner_project/src/evaluate_model.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
