"""
Print the evaluation set for human review.

This is intentionally read-only.  After checking the output, a human reviewer
runs ``mark_evaluation_reviewed.py`` to record sign-off.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from evaluation import combined_text, load_eval_examples

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PATH = os.path.join(PROJECT_DIR, "data", "evaluation_examples.json")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", default=DEFAULT_PATH, help="Evaluation JSON file")
    parser.add_argument(
        "--context",
        type=int,
        default=35,
        help="Characters of context around each span (default: 35)",
    )
    args = parser.parse_args()

    examples = load_eval_examples(args.examples)
    print(f"{len(examples)} examples / "
          f"{sum(len(ex['gold_entities']) for ex in examples)} gold spans")
    print(f"Review statuses: "
          f"{', '.join(sorted({ex.get('review_status', 'unknown') for ex in examples}))}")
    print()

    for example in examples:
        print("=" * 100)
        print(f"{example['id']} | {example.get('review_status')} | "
              f"original_id={example.get('original_id')} | "
              f"source_row={example.get('source_row')} | "
              f"source={example.get('source_name')}")
        print(f"TITLE: {example['title']}")
        print(f"BODY:  {example['body']}")
        print(f"NOTES: {example.get('notes', '')}")
        print("GOLD:")
        if not example["gold_entities"]:
            print("  (none - negative control)")
        text = combined_text(example)
        for entity in example["gold_entities"]:
            start = entity["start"]
            end = entity["end"]
            left = max(0, start - args.context)
            right = min(len(text), end + args.context)
            prefix = "..." if left else ""
            suffix = "..." if right < len(text) else ""
            context = f"{prefix}{text[left:right]}{suffix}".replace("\n", " ")
            print(f"  {entity['label']:<7} {entity['text']!r} [{start}:{end}] | {context}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
