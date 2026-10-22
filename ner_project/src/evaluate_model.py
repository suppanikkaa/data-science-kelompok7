"""
Reproducible evaluation entry point.

Runs the baseline spaCy model and the fine-tuned model on the examples in
``data/evaluation_examples.json`` without retraining.  The per-example review
status is printed and recorded; a pending set is labelled as such, never as
human-reviewed.

Examples
--------
    python src/evaluate_model.py
    python src/evaluate_model.py --out data/evaluation_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import spacy

sys.path.insert(0, os.path.dirname(__file__))
from evaluation import EVAL_LABELS, evaluate, load_eval_examples

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
DEFAULT_EXAMPLES = os.path.join(DATA_DIR, "evaluation_examples.json")
DEFAULT_MODEL = os.path.join(DATA_DIR, "trained_model")
DEFAULT_OUT = os.path.join(DATA_DIR, "evaluation_report.json")


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _relative(path: str) -> str:
    """Keep generated reports portable by storing project-relative paths."""
    try:
        return os.path.relpath(os.path.abspath(path), PROJECT_DIR)
    except ValueError:
        return os.path.abspath(path)


def _print_metrics(name: str, metrics: dict) -> None:
    overall = metrics["overall"]
    print(
        f"{name:<10} P={_fmt(overall['precision'])}  "
        f"R={_fmt(overall['recall'])}  F1={_fmt(overall['f1'])}  "
        f"(tp={metrics['counts']['tp']}, fp={metrics['counts']['fp']}, "
        f"fn={metrics['counts']['fn']})"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", default=DEFAULT_EXAMPLES, help="Gold JSON file")
    parser.add_argument(
        "--trained-model",
        default=DEFAULT_MODEL,
        help="Path to the fine-tuned spaCy model directory",
    )
    parser.add_argument(
        "--baseline",
        default="en_core_web_sm",
        help="Baseline spaCy model name or path",
    )
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output JSON report")
    args = parser.parse_args()

    examples = load_eval_examples(args.examples)
    print(
        f"Loaded {len(examples)} examples / "
        f"{sum(len(ex['gold_entities']) for ex in examples)} gold spans"
    )
    print(f"Labels: {', '.join(sorted(EVAL_LABELS))}")
    print(f"Matching: exact-span+label")
    review_statuses = sorted({ex.get("review_status", "unknown") for ex in examples})
    gold_spans = sum(len(ex["gold_entities"]) for ex in examples)
    print(f"Review status: {', '.join(review_statuses)}")
    if review_statuses != ["human-reviewed"]:
        print(
            "WARNING: this evaluation set is not fully human-reviewed; "
            "treat the metrics as candidate results.\n"
        )
    else:
        print()

    baseline_nlp = spacy.load(args.baseline)
    baseline_metrics = evaluate(baseline_nlp, examples)

    print("Baseline model")
    _print_metrics(args.baseline, baseline_metrics)

    trained_metrics = None
    if os.path.isdir(args.trained_model):
        trained_nlp = spacy.load(args.trained_model)
        trained_metrics = evaluate(trained_nlp, examples)
        print(f"\nFine-tuned model ({args.trained_model})")
        _print_metrics("trained", trained_metrics)
    else:
        print(f"\nTrained model not found at {args.trained_model}; skipping.")

    print("\nPer-label metrics (fine-tuned model)")
    if trained_metrics:
        for label, values in trained_metrics["per_label"].items():
            print(
                f"  {label:<7} P={_fmt(values['precision'])}  "
                f"R={_fmt(values['recall'])}  F1={_fmt(values['f1'])}  "
                f"(support={values['support']})"
            )

    report = {
        "examples_file": _relative(args.examples),
        "matching": "exact-span+label",
        "labels_evaluated": sorted(EVAL_LABELS),
        "evaluation_examples": {
            "count": len(examples),
            "gold_spans": gold_spans,
            "review_statuses": review_statuses,
            "all_human_reviewed": review_statuses == ["human-reviewed"],
        },
        "baseline_model": args.baseline,
        "trained_model": _relative(args.trained_model),
        "baseline": baseline_metrics,
        "trained": trained_metrics,
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"\nReport written to {os.path.abspath(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
