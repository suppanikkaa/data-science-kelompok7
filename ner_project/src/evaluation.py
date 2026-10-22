"""
Evaluation utilities for the NER service.

The evaluation set lives in ``data/evaluation_examples.json``.  Every gold
entity is a span-level annotation over the *combined* text.  The review status
is stored per example; a ``human-reviewed`` status requires a reviewer name and
review date, so the file cannot accidentally claim human review before it has
happened:

    combined = title + "\\n\\n" + body

Matching is exact-span + exact-label.  Predictions whose label is outside the
documented evaluation label set (for example CARDINAL, PERCENT or ORDINAL) are
ignored so that the metrics only cover the categories the project reports.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Sequence, Set

# Labels covered by the evaluation set and by the user-facing label mapping.
EVAL_LABELS: Set[str] = {
    "PERSON",
    "ORG",
    "GPE",
    "LOC",
    "DATE",
    "MONEY",
    "EVENT",
    "NORP",
}

MATCHING = "exact-span+label"


def combined_text(example: Dict[str, Any]) -> str:
    """Return the exact text the NER model analyses for one example."""
    return f"{example['title'].strip()}\n\n{example['body'].strip()}"


def load_eval_examples(path: str) -> List[Dict[str, Any]]:
    """Load and validate the evaluation examples and their provenance."""
    with open(path, "r", encoding="utf-8-sig") as fh:
        examples = json.load(fh)
    if not isinstance(examples, list):
        raise ValueError(f"Evaluation file must contain a JSON list: {path}")
    validate_eval_examples(examples)
    return examples


def validate_eval_examples(examples: Sequence[Dict[str, Any]]) -> None:
    """
    Validate gold spans before any metric is computed.

    Raises ``ValueError`` with a precise message when a span does not point at
    its own text, overlaps another span, or uses an unsupported label.  This
    prevents the silent offset bug that made earlier metrics meaningless.
    """
    if not examples:
        raise ValueError("Evaluation set is empty")

    seen_example_ids = set()
    for ex in examples:
        ex_id = ex.get("id", "<missing id>")
        if ex_id in seen_example_ids:
            raise ValueError(f"Duplicate evaluation example id: {ex_id!r}")
        seen_example_ids.add(ex_id)
        for field in ("title", "body", "gold_entities"):
            if field not in ex:
                raise ValueError(f"{ex_id}: missing required field '{field}'")
        if not isinstance(ex["title"], str) or not ex["title"].strip():
            raise ValueError(f"{ex_id}: title must be a non-empty string")
        if not isinstance(ex["body"], str) or not ex["body"].strip():
            raise ValueError(f"{ex_id}: body must be a non-empty string")
        if not isinstance(ex["gold_entities"], list):
            raise ValueError(
                f"{ex_id}: gold_entities must be a list "
                "(it may be empty for a negative control)"
            )

        # Repository examples carry provenance and review metadata.  If a file
        # claims human review, it must also name the reviewer and the date.
        if "review_status" in ex:
            if not ex.get("original_id"):
                raise ValueError(f"{ex_id}: missing original_id provenance")
            if ex.get("source_row") is None:
                raise ValueError(f"{ex_id}: missing source_row provenance")
            if ex["review_status"] == "human-reviewed":
                if not ex.get("reviewer") or not ex.get("reviewed_at"):
                    raise ValueError(
                        f"{ex_id}: human-reviewed status requires reviewer and reviewed_at"
                    )

        text = combined_text(ex)
        seen = set()
        spans = []
        for ent in ex["gold_entities"]:
            for field in ("text", "label", "start", "end"):
                if field not in ent:
                    raise ValueError(f"{ex_id}: gold entity missing '{field}': {ent!r}")
            ent_text = ent["text"]
            label = ent["label"]
            start = ent["start"]
            end = ent["end"]

            if not isinstance(ent_text, str) or not ent_text:
                raise ValueError(f"{ex_id}: entity text must be a non-empty string")
            if label not in EVAL_LABELS:
                raise ValueError(
                    f"{ex_id}: unsupported label {label!r}; expected one of "
                    f"{sorted(EVAL_LABELS)}"
                )
            if not isinstance(start, int) or not isinstance(end, int):
                raise ValueError(f"{ex_id}: offsets must be integers: {ent_text!r}")
            if start < 0 or end <= start or end > len(text):
                raise ValueError(
                    f"{ex_id}: invalid offsets ({start}, {end}) for {ent_text!r} "
                    f"in text of length {len(text)}"
                )
            if text[start:end] != ent_text:
                raise ValueError(
                    f"{ex_id}: offset mismatch for {ent_text!r}: "
                    f"text[{start}:{end}] == {text[start:end]!r}"
                )
            key = (start, end, label)
            if key in seen:
                raise ValueError(f"{ex_id}: duplicate gold span {key} for {ent_text!r}")
            seen.add(key)
            spans.append((start, end, ent_text))

        spans.sort()
        for (prev_start, prev_end, prev_text), (start, end, text_value) in zip(
            spans, spans[1:]
        ):
            if start < prev_end:
                raise ValueError(
                    f"{ex_id}: overlapping gold spans {prev_text!r} "
                    f"({prev_start}, {prev_end}) and {text_value!r} ({start}, {end})"
                )


def _context(text: str, start: int, end: int, window: int = 45) -> str:
    """Return a short human-readable context around a span."""
    left = max(0, start - window)
    right = min(len(text), end + window)
    prefix = "..." if left > 0 else ""
    suffix = "..." if right < len(text) else ""
    return f"{prefix}{text[left:right]}{suffix}"


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def evaluate(
    nlp: Any,
    eval_examples: Iterable[Dict[str, Any]],
    labels: Iterable[str] = EVAL_LABELS,
) -> Dict[str, Any]:
    """
    Compute exact-span + label precision, recall and F1.

    ``nlp`` only needs to be callable and return an object with ``.ents`` whose
    items expose ``start_char``, ``end_char``, ``label_`` and ``text``.  That
    keeps the evaluator testable with a lightweight fake model.
    """
    examples = list(eval_examples)
    validate_eval_examples(examples)

    label_set = set(labels)
    tp = fp = fn = 0
    per_label: Dict[str, Dict[str, int]] = {}
    errors: List[Dict[str, Any]] = []

    for ex in examples:
        text = combined_text(ex)
        doc = nlp(text)
        predicted = {
            (ent.start_char, ent.end_char, ent.label_, ent.text)
            for ent in doc.ents
            if ent.label_ in label_set
        }
        gold = {
            (ent["start"], ent["end"], ent["label"], ent["text"])
            for ent in ex["gold_entities"]
        }

        true_pos = predicted & gold
        false_pos = predicted - gold
        false_neg = gold - predicted

        tp += len(true_pos)
        fp += len(false_pos)
        fn += len(false_neg)

        for _, _, label, _ in true_pos:
            per_label.setdefault(label, {"tp": 0, "fp": 0, "fn": 0})["tp"] += 1
        for start, end, label, ent_text in false_pos:
            per_label.setdefault(label, {"tp": 0, "fp": 0, "fn": 0})["fp"] += 1
            errors.append(
                {
                    "type": "FP",
                    "article": ex["id"],
                    "text": ent_text,
                    "label": label,
                    "start": start,
                    "end": end,
                    "context": _context(text, start, end),
                }
            )
        for start, end, label, ent_text in false_neg:
            per_label.setdefault(label, {"tp": 0, "fp": 0, "fn": 0})["fn"] += 1
            errors.append(
                {
                    "type": "FN",
                    "article": ex["id"],
                    "text": ent_text,
                    "label": label,
                    "start": start,
                    "end": end,
                    "context": _context(text, start, end),
                }
            )

    per_label_metrics: Dict[str, Dict[str, Any]] = {}
    for label, counts in sorted(per_label.items()):
        metrics = _prf(counts["tp"], counts["fp"], counts["fn"])
        per_label_metrics[label] = {
            **metrics,
            "tp": counts["tp"],
            "fp": counts["fp"],
            "fn": counts["fn"],
            "support": counts["tp"] + counts["fn"],
        }

    macro_precision = _safe_div(
        sum(m["precision"] for m in per_label_metrics.values()), len(per_label_metrics)
    )
    macro_recall = _safe_div(
        sum(m["recall"] for m in per_label_metrics.values()), len(per_label_metrics)
    )
    macro_f1 = _safe_div(
        sum(m["f1"] for m in per_label_metrics.values()), len(per_label_metrics)
    )

    errors.sort(key=lambda item: (item["article"], item["start"], item["type"]))

    return {
        "matching": MATCHING,
        "labels_evaluated": sorted(label_set),
        "examples": len(examples),
        "gold_spans": tp + fn,
        "predicted_spans": tp + fp,
        "overall": _prf(tp, fp, fn),
        "macro": {
            "precision": round(macro_precision, 4),
            "recall": round(macro_recall, 4),
            "f1": round(macro_f1, 4),
        },
        "per_label": per_label_metrics,
        "counts": {"tp": tp, "fp": fp, "fn": fn},
        "errors": errors,
        "sample_errors": errors[:20],
    }
