"""
Regression tests for the evaluation contract.

These tests guard against the failure mode that produced invalid metrics in
the first version of the project: gold offsets that do not point at their own
entity text.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_loader import find_near_duplicate_clusters
from evaluation import (
    EVAL_LABELS,
    combined_text,
    evaluate,
    load_eval_examples,
    validate_eval_examples,
)

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_DIR, "data")


class FakeEnt:
    def __init__(self, text, start, end, label):
        self.text = text
        self.start_char = start
        self.end_char = end
        self.label_ = label


class FakeDoc:
    def __init__(self, ents):
        self.ents = ents


class FakeNLP:
    """Minimal callable that mimics the parts of spaCy used by evaluate()."""

    def __init__(self, mapping):
        self.mapping = mapping

    def __call__(self, text):
        return FakeDoc(
            [FakeEnt(text_value, start, end, label) for text_value, start, end, label in self.mapping.get(text, [])]
        )


def make_example(title, body, entities):
    text = f"{title}\n\n{body}"
    gold = []
    for value, label in entities:
        start = text.index(value)
        gold.append(
            {
                "text": value,
                "label": label,
                "start": start,
                "end": start + len(value),
            }
        )
    return {"id": "test", "title": title, "body": body, "gold_entities": gold}


def fake_nlp_for(example, entities):
    text = combined_text(example)
    ents = []
    for value, label in entities:
        start = text.index(value)
        ents.append((value, start, start + len(value), label))
    return FakeNLP({text: ents})


def test_repository_evaluation_examples_are_valid():
    path = os.path.join(DATA_DIR, "evaluation_examples.json")
    examples = load_eval_examples(path)
    assert len(examples) == 12
    assert sum(len(ex["gold_entities"]) for ex in examples) == 122

    labels = {
        ent["label"] for ex in examples for ent in ex["gold_entities"]
    }
    assert labels == EVAL_LABELS

    for ex in examples:
        text = combined_text(ex)
        for ent in ex["gold_entities"]:
            assert text[ent["start"] : ent["end"]] == ent["text"]

    # The test set must be traceable to the held-out split, not invented text.
    for ex in examples:
        assert ex["original_id"]
        assert isinstance(ex["source_row"], int) and ex["source_row"] >= 1
        assert ex["split"] == "test"
        assert ex["near_duplicate_cluster"]
        assert ex["review_status"] in {"pending-human-review", "human-reviewed"}
        if ex["review_status"] == "human-reviewed":
            assert ex["reviewer"]
            assert ex["reviewed_at"]

    negative_controls = [ex for ex in examples if not ex["gold_entities"]]
    assert len(negative_controls) == 1


def test_annotations_jsonl_is_text_free_and_has_provenance():
    path = os.path.join(DATA_DIR, "annotations.jsonl")
    assert os.path.exists(path), "Run the training pipeline to generate annotations.jsonl"

    forbidden_fields = {"title", "body", "combined_text", "raw_text", "text"}
    records = []
    with open(path, "r", encoding="utf-8") as file_handle:
        for line in file_handle:
            if not line.strip():
                continue
            record = json.loads(line)
            records.append(record)
            assert not (forbidden_fields & set(record))
            assert record["original_id"]
            assert isinstance(record["source_row"], int)
            assert record["split"] in {"train", "validation", "test"}
            assert isinstance(record["spans"], list)
            assert record["label_source"] == "weak_baseline_en_core_web_sm"
            assert record["human_reviewed"] is False
            assert len(record["text_sha256"]) == 64

    assert len(records) == 577
    split_counts = {
        split: sum(1 for record in records if record["split"] == split)
        for split in ("train", "validation", "test")
    }
    assert split_counts == {"train": 404, "validation": 86, "test": 87}


def test_near_duplicate_clusters_link_syndicated_copies():
    body = "Maybank Group reported a record profit in Kuala Lumpur. " * 10
    articles = [
        {"language": "en", "title": "Maybank posts record profit", "body": body, "original_id": "a"},
        {"language": "en", "title": "Maybank posts record profit", "body": body, "original_id": "b"},
        {"language": "en", "title": "CIMB appoints a new chief executive", "body": "CIMB Group named a new chief executive in Kuala Lumpur.", "original_id": "c"},
    ]
    clusters, pairs = find_near_duplicate_clusters(articles)
    assert len(clusters) == 2
    assert len(pairs) == 1
    assert articles[0]["near_duplicate_cluster"] == articles[1]["near_duplicate_cluster"]
    assert articles[2]["near_duplicate_cluster"] != articles[0]["near_duplicate_cluster"]


def test_cluster_aware_split_never_splits_a_cluster():
    from annotate_and_train import split_articles

    articles = []
    for cluster_index in range(30):
        for member_index in range(2):
            articles.append(
                {
                    "original_id": f"c{cluster_index:03d}-{member_index}",
                    "near_duplicate_cluster": f"cluster_{cluster_index:03d}",
                    "spans": [],
                }
            )

    train, validation, test = split_articles(articles, seed=42)
    split_clusters = [
        {article["near_duplicate_cluster"] for article in split}
        for split in (train, validation, test)
    ]
    assert not (split_clusters[0] & split_clusters[1])
    assert not (split_clusters[0] & split_clusters[2])
    assert not (split_clusters[1] & split_clusters[2])
    assert len(train) + len(validation) + len(test) == len(articles)


def test_validate_rejects_offset_mismatch():
    example = {
        "id": "bad",
        "title": "Maybank",
        "body": "reported profit",
        "gold_entities": [
            {"text": "Maybank", "label": "ORG", "start": 1, "end": 8}
        ],
    }
    with pytest.raises(ValueError, match="offset mismatch"):
        validate_eval_examples([example])


def test_validate_rejects_overlapping_spans():
    example = {
        "id": "overlap",
        "title": "Maybank Group",
        "body": "reported profit",
        "gold_entities": [
            {"text": "Maybank", "label": "ORG", "start": 0, "end": 7},
            {"text": "Maybank Group", "label": "ORG", "start": 0, "end": 13},
        ],
    }
    with pytest.raises(ValueError, match="overlapping gold spans"):
        validate_eval_examples([example])


def test_validate_rejects_unsupported_label():
    example = {
        "id": "label",
        "title": "Profit rose 5 percent",
        "body": "The bank reported growth.",
        "gold_entities": [
            {"text": "5", "label": "CARDINAL", "start": 12, "end": 13}
        ],
    }
    with pytest.raises(ValueError, match="unsupported label"):
        validate_eval_examples([example])


def test_validate_requires_reviewer_for_human_reviewed_claim():
    example = {
        "id": "claim",
        "original_id": "article-1",
        "source_row": 1,
        "review_status": "human-reviewed",
        "reviewer": None,
        "reviewed_at": None,
        "title": "Maybank posts profit",
        "body": "Maybank reported growth.",
        "gold_entities": [],
    }
    with pytest.raises(ValueError, match="reviewer and reviewed_at"):
        validate_eval_examples([example])


def test_validate_rejects_duplicate_example_ids():
    example = {
        "id": "same",
        "title": "Maybank posts profit",
        "body": "Maybank reported growth.",
        "gold_entities": [],
    }
    with pytest.raises(ValueError, match="Duplicate evaluation example id"):
        validate_eval_examples([example, dict(example)])


def test_perfect_predictions_score_one():
    example = make_example(
        "Maybank posts profit",
        "Maybank reported growth in Malaysia.",
        [("Maybank", "ORG"), ("Malaysia", "GPE")],
    )
    metrics = evaluate(fake_nlp_for(example, [("Maybank", "ORG"), ("Malaysia", "GPE")]), [example])
    assert metrics["overall"] == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert metrics["counts"] == {"tp": 2, "fp": 0, "fn": 0}


def test_false_positive_and_false_negative_are_counted():
    example = make_example(
        "Maybank posts profit",
        "Maybank reported growth in Malaysia.",
        [("Maybank", "ORG"), ("Malaysia", "GPE")],
    )
    text = combined_text(example)
    reported_start = text.index("reported")
    fake = FakeNLP(
        {
            text: [
                ("Maybank", 0, len("Maybank"), "ORG"),
                ("reported", reported_start, reported_start + len("reported"), "ORG"),
            ]
        }
    )
    metrics = evaluate(fake, [example])
    assert metrics["counts"] == {"tp": 1, "fp": 1, "fn": 1}
    assert metrics["overall"] == {"precision": 0.5, "recall": 0.5, "f1": 0.5}


def test_out_of_scope_predictions_are_ignored():
    example = make_example(
        "Maybank posts profit",
        "Maybank reported growth of 5 percent in 2026.",
        [("Maybank", "ORG"), ("2026", "DATE")],
    )
    text = combined_text(example)
    percent_start = text.index("5")
    year_start = text.index("2026")
    fake = FakeNLP(
        {
            text: [
                ("Maybank", 0, len("Maybank"), "ORG"),
                ("2026", year_start, year_start + 4, "DATE"),
                ("5", percent_start, percent_start + 1, "CARDINAL"),
            ]
        }
    )
    metrics = evaluate(fake, [example])
    assert metrics["predicted_spans"] == 2
    assert metrics["counts"] == {"tp": 2, "fp": 0, "fn": 0}
    assert "CARDINAL" not in metrics["per_label"]


def test_negative_control_allows_empty_gold():
    example = {
        "id": "negative",
        "title": "Staff town hall rescheduled",
        "body": "The staff town hall has been moved to a larger room.",
        "gold_entities": [],
    }
    text = combined_text(example)
    room_start = text.index("room")
    fake = FakeNLP(
        {text: [("room", room_start, room_start + len("room"), "ORG")]}
    )
    metrics = evaluate(fake, [example])
    assert metrics["gold_spans"] == 0
    assert metrics["counts"] == {"tp": 0, "fp": 1, "fn": 0}


def test_per_label_metrics_and_errors_include_context():
    example = make_example(
        "Maybank posts profit",
        "Maybank reported growth in Malaysia.",
        [("Maybank", "ORG"), ("Malaysia", "GPE")],
    )
    text = combined_text(example)
    malaysia_start = text.index("Malaysia")
    fake = FakeNLP(
        {
            text: [
                ("Maybank", 0, len("Maybank"), "ORG"),
                ("Malaysia", malaysia_start, malaysia_start + len("Malaysia"), "LOC"),
            ]
        }
    )
    metrics = evaluate(fake, [example])
    assert metrics["counts"] == {"tp": 1, "fp": 1, "fn": 1}
    assert metrics["per_label"]["ORG"]["tp"] == 1
    assert metrics["per_label"]["GPE"]["fn"] == 1
    assert metrics["per_label"]["LOC"]["fp"] == 1
    assert metrics["errors"]
    assert all("context" in error for error in metrics["errors"])
    assert {error["type"] for error in metrics["errors"]} == {"FP", "FN"}
