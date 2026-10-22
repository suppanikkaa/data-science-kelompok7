"""
Annotation, data splitting, validation-based model selection, and training.

Pipeline
--------
1. Load English articles from the provided CSV files.
2. Remove exact duplicates and group near-duplicates (see ``data_loader.py``).
3. Auto-annotate with the baseline spaCy model (weak supervision).
4. Split by near-duplicate cluster / article identity, 70/15/15, seed 42.
5. Fine-tune ``en_core_web_sm`` with two documented configurations.
6. Evaluate validation after every epoch and keep the best epoch per config.
7. Select the final configuration by validation exact-span F1.
8. Evaluate the untouched human-reviewed test set exactly once.
9. Save the selected model and a text-free ``annotations.jsonl``.

The human-reviewed test set lives in ``data/evaluation_examples.json``.  It is
never used for training or model selection.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import spacy
from spacy.training import Example

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_all
from evaluation import EVAL_LABELS, evaluate, load_eval_examples

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEED = 42

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODEL_OUT = os.path.join(DATA_DIR, "trained_model")
ANNOT_PATH = os.path.join(DATA_DIR, "annotations.jsonl")
EVAL_PATH = os.path.join(DATA_DIR, "evaluation_examples.json")

# Two documented configurations.  Validation picks both the best epoch and the
# winning configuration.  We deliberately keep this small so a CPU-only
# reproduction stays practical.
TRAINING_CONFIGS = [
    {"name": "dropout_0.35_batch_8", "dropout": 0.35, "batch_size": 8, "epochs": 3},
    {"name": "dropout_0.25_batch_8", "dropout": 0.25, "batch_size": 8, "epochs": 3},
]

os.makedirs(DATA_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Loading and weak annotation
# ---------------------------------------------------------------------------

def load_articles() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    logger.info("Loading articles from CSV files...")
    articles, report = load_all()
    logger.info(
        "Loaded %s unique articles from %s raw rows | duplicates removed: %s "
        "(id=%s, content=%s)",
        report["total_rows"],
        report["raw_rows"],
        report["total_duplicates"],
        report["total_id_duplicates"],
        report["total_content_duplicates"],
    )
    logger.info("Language distribution: %s", report["language_distribution"])
    logger.info(
        "Missing titles: %s | Missing bodies: %s | Non-empty entities column: %s",
        report["missing_title"],
        report["missing_body"],
        report["entities_nonempty"],
    )
    logger.info(
        "Near-duplicate audit: %s clusters, %s multi-document clusters, largest=%s",
        report["near_duplicate_audit"]["clusters"],
        report["near_duplicate_audit"]["multi_document_clusters"],
        report["near_duplicate_audit"]["largest_cluster_size"],
    )
    english_articles = [article for article in articles if article["language"] == "en"]
    logger.info("English articles used for NER: %s", len(english_articles))
    return english_articles, report


def auto_annotate(articles: List[Dict[str, Any]], nlp) -> List[Dict[str, Any]]:
    """
    Run the baseline spaCy model to produce weak labels.

    These are *not* gold labels.  They are the training signal for this
    prototype and are recorded as ``label_source = weak_baseline``.
    """
    logger.info("Running auto-annotation with baseline model...")
    annotated = []
    for article in articles:
        combined = f"{article['title'].strip()}\n\n{article['body'].strip()}"
        doc = nlp(combined)
        spans = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
        annotated.append(
            {
                **article,
                "combined_text": combined,
                "spans": spans,
                "human_reviewed": False,
                "label_source": "weak_baseline",
            }
        )
    logger.info("Auto-annotated %s articles", len(annotated))
    return annotated


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def split_articles(
    articles: List[Dict[str, Any]], seed: int = SEED
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split by near-duplicate cluster, which is stronger than splitting by row.

    Exact duplicates were already removed by the loader.  Remaining
    near-duplicates receive the same ``near_duplicate_cluster`` and therefore
    always land in the same split.  The greedy assignment below keeps document
    counts close to 70/15/15 while never splitting a cluster.
    """
    clusters: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for article in articles:
        key = article.get("near_duplicate_cluster")
        if not key:
            key = f"single:{article.get('original_id') or article.get('content_hash')}"
        clusters[key].append(article)

    cluster_ids = list(clusters)
    random.Random(seed).shuffle(cluster_ids)

    total_documents = len(articles)
    targets = {"train": 0.70, "validation": 0.15, "test": 0.15}
    counts = {split: 0 for split in targets}
    assigned: Dict[str, List[Dict[str, Any]]] = {split: [] for split in targets}

    for cluster_id in cluster_ids:
        cluster = clusters[cluster_id]
        split = max(
            targets,
            key=lambda candidate: targets[candidate] * total_documents - counts[candidate],
        )
        assigned[split].extend(cluster)
        counts[split] += len(cluster)

    for split in assigned:
        assigned[split].sort(key=lambda article: article.get("original_id", ""))

    logger.info(
        "Split by near-duplicate cluster (seed=%s): train=%s, validation=%s, test=%s",
        seed,
        len(assigned["train"]),
        len(assigned["validation"]),
        len(assigned["test"]),
    )
    return assigned["train"], assigned["validation"], assigned["test"]


def label_distribution(articles: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count weak-label spans per entity label for one split."""
    counts: Dict[str, int] = {}
    for article in articles:
        for _, _, label in article["spans"]:
            counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def split_summary(articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Document, cluster, span and label counts for one split."""
    cluster_ids = {
        article.get("near_duplicate_cluster")
        for article in articles
        if article.get("near_duplicate_cluster")
    }
    return {
        "documents": len(articles),
        "near_duplicate_clusters": len(cluster_ids),
        "spans": sum(len(article["spans"]) for article in articles),
        "labels": label_distribution(articles),
    }


# ---------------------------------------------------------------------------
# Training and validation-based selection
# ---------------------------------------------------------------------------

def _make_examples(nlp, batch: List[Dict[str, Any]]) -> List[Example]:
    examples = []
    for article in batch:
        text = article["combined_text"]
        reference_doc = nlp.make_doc(text)
        entities = []
        for start, end, label in article["spans"]:
            span = reference_doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is not None:
                entities.append(span)
        reference_doc.ents = spacy.util.filter_spans(entities)
        predicted_doc = nlp.make_doc(text)
        examples.append(Example(predicted_doc, reference_doc))
    return examples


def _compact_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Keep aggregate validation metrics; raw-text error contexts are omitted."""
    return {
        "examples": metrics["examples"],
        "gold_spans": metrics["gold_spans"],
        "predicted_spans": metrics["predicted_spans"],
        "overall": metrics["overall"],
        "macro": metrics["macro"],
        "per_label": metrics["per_label"],
        "counts": metrics["counts"],
        "errors_total": len(metrics["errors"]),
        "errors_note": (
            "Validation error contexts are omitted from the committed report "
            "because validation examples are raw dataset articles."
        ),
    }


def train_config(
    train_data: List[Dict[str, Any]],
    val_examples: List[Dict[str, Any]],
    config: Dict[str, Any],
    seed: int,
) -> Tuple[Any, List[Dict[str, Any]], Dict[str, Any], float]:
    """
    Fine-tune one configuration and keep its best validation epoch.

    Validation is evaluated after every epoch.  The best epoch is selected by
    exact-span+label F1.  The test set is not touched in this function.
    """
    logger.info("Training configuration: %s", config["name"])
    nlp = spacy.load("en_core_web_sm")
    optimizer = nlp.resume_training()
    rng = random.Random(seed)

    history: List[Dict[str, Any]] = []
    best_epoch = 0
    best_f1 = -1.0
    best_metrics: Dict[str, Any] | None = None
    best_model_bytes: bytes | None = None
    start = time.time()

    for epoch in range(1, config["epochs"] + 1):
        rng.shuffle(train_data)
        losses: Dict[str, float] = {}
        batches = spacy.util.minibatch(train_data, size=config["batch_size"])
        for batch in batches:
            examples = _make_examples(nlp, batch)
            if examples:
                nlp.update(
                    examples,
                    sgd=optimizer,
                    drop=config["dropout"],
                    losses=losses,
                )

        epoch_loss = float(losses.get("ner", 0.0))
        validation_metrics = evaluate(nlp, val_examples)
        validation_f1 = validation_metrics["overall"]["f1"]
        record = {
            "epoch": epoch,
            "ner_loss": epoch_loss,
            "validation": _compact_metrics(validation_metrics),
        }
        history.append(record)
        logger.info(
            "  epoch %s/%s | NER loss %.4f | validation P=%.4f R=%.4f F1=%.4f",
            epoch,
            config["epochs"],
            epoch_loss,
            validation_metrics["overall"]["precision"],
            validation_metrics["overall"]["recall"],
            validation_f1,
        )

        if validation_f1 > best_f1:
            best_f1 = validation_f1
            best_epoch = epoch
            best_metrics = _compact_metrics(validation_metrics)
            best_model_bytes = nlp.to_bytes()

    if best_model_bytes is None or best_metrics is None:
        raise RuntimeError(f"No validation checkpoint was produced for {config['name']}")

    nlp.from_bytes(best_model_bytes)
    elapsed = time.time() - start
    logger.info(
        "Configuration %s selected epoch %s with validation F1 %.4f (%.1fs)",
        config["name"],
        best_epoch,
        best_f1,
        elapsed,
    )
    best = {
        "epoch": best_epoch,
        "validation": best_metrics,
        "model_bytes": best_model_bytes,
    }
    return nlp, history, best, elapsed


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _validation_examples(val_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build an evaluation structure from weak-labelled validation articles."""
    examples = []
    for index, article in enumerate(val_data):
        if not article["spans"]:
            continue
        title = article.get("title", "").strip()
        body = article.get("body", "").strip()
        if not title or not body:
            # Fall back to the combined field for synthetic test fixtures.
            parts = article["combined_text"].split("\n\n", 1)
            title = parts[0]
            body = parts[1] if len(parts) > 1 else ""
        gold_entities = [
            {
                "text": article["combined_text"][start:end],
                "label": label,
                "start": start,
                "end": end,
            }
            for start, end, label in article["spans"]
            if label in EVAL_LABELS
        ]
        if not gold_entities:
            continue
        examples.append(
            {
                "id": article.get("original_id") or f"validation_{index:03d}",
                "original_id": article.get("original_id", ""),
                "source_row": article.get("source_row"),
                "title": title,
                "body": body,
                "gold_entities": gold_entities,
                "review_status": "weak-validation",
            }
        )
    return examples


def _data_profile(data_report: Dict[str, Any]) -> Dict[str, Any]:
    """Summarise rows, duplicates, missing values, languages and near-duplicates."""
    return {
        "raw_rows": data_report["raw_rows"],
        "unique_rows": data_report["total_rows"],
        "duplicates_removed": data_report["total_duplicates"],
        "id_duplicates_removed": data_report["total_id_duplicates"],
        "content_duplicates_removed": data_report["total_content_duplicates"],
        "missing_title": data_report["missing_title"],
        "missing_body": data_report["missing_body"],
        "entities_nonempty": data_report["entities_nonempty"],
        "language_distribution": data_report["language_distribution"],
        "files": data_report["files"],
        "near_duplicate_audit": data_report["near_duplicate_audit"],
    }


def _class_imbalance(train_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts = label_distribution(train_data)
    total = sum(counts.values())
    shares = {
        label: round(count / total, 4) if total else 0.0
        for label, count in counts.items()
    }
    return {
        "note": "Weak labels are dominated by ORG, PERSON and DATE; EVENT and LOC are very rare.",
        "train_span_counts": counts,
        "train_span_share": shares,
    }


def _evaluation_provenance(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    statuses = sorted({example.get("review_status", "unknown") for example in examples})
    return {
        "examples": len(examples),
        "gold_spans": sum(len(example["gold_entities"]) for example in examples),
        "review_statuses": statuses,
        "all_human_reviewed": statuses == ["human-reviewed"],
        "original_ids_present": all(bool(example.get("original_id")) for example in examples),
        "source_rows_present": all(example.get("source_row") is not None for example in examples),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    articles, data_report = load_articles()

    logger.info("Loading baseline spaCy model...")
    baseline_nlp = spacy.load("en_core_web_sm")
    baseline_version = baseline_nlp.meta.get("version", "unknown")

    annotated = auto_annotate(articles, baseline_nlp)
    train_data, val_data, test_data = split_articles(annotated)

    # Text-free weak annotations.  No title/body is written here.
    split_by_id = {}
    for split_name, split_articles_list in (
        ("train", train_data),
        ("validation", val_data),
        ("test", test_data),
    ):
        for article in split_articles_list:
            split_by_id[article.get("original_id")] = split_name

    with open(ANNOT_PATH, "w", encoding="utf-8") as file_handle:
        for article in annotated:
            combined = article["combined_text"]
            record = {
                "original_id": article.get("original_id", ""),
                "source_row": article.get("source_row"),
                "source_name": article.get("source_name", ""),
                "date_published": article.get("date_published", ""),
                "clipping": article.get("clipping", ""),
                "language": article.get("language", ""),
                "near_duplicate_cluster": article.get("near_duplicate_cluster"),
                "split": split_by_id.get(article.get("original_id"), "unknown"),
                "text_sha256": hashlib.sha256(combined.encode("utf-8")).hexdigest(),
                "spans": article["spans"],
                "label_source": "weak_baseline_en_core_web_sm",
                "human_reviewed": False,
            }
            file_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Text-free weak annotations saved to %s", ANNOT_PATH)

    eval_examples = load_eval_examples(EVAL_PATH)
    provenance = _evaluation_provenance(eval_examples)
    logger.info(
        "Loaded %s evaluation examples / %s gold spans | review status: %s",
        provenance["examples"],
        provenance["gold_spans"],
        provenance["review_statuses"],
    )

    logger.info("=== BASELINE EVALUATION (HELD-OUT EVALUATION SET) ===")
    baseline_metrics = evaluate(baseline_nlp, eval_examples)
    logger.info("Baseline overall: %s", baseline_metrics["overall"])

    val_examples = _validation_examples(val_data)
    logger.info(
        "Validation set: %s examples / %s weak gold spans",
        len(val_examples),
        sum(len(example["gold_entities"]) for example in val_examples),
    )

    logger.info("=== VALIDATION-BASED CONFIGURATION SEARCH ===")
    config_results = []
    for config_index, config in enumerate(TRAINING_CONFIGS):
        trained_nlp, history, best, elapsed = train_config(
            train_data,
            val_examples,
            config,
            seed=SEED + config_index,
        )
        config_results.append(
            {
                "name": config["name"],
                "settings": {
                    "base_model": "en_core_web_sm",
                    "epochs": config["epochs"],
                    "dropout": config["dropout"],
                    "batch_size": config["batch_size"],
                    "seed": SEED + config_index,
                },
                "history": history,
                "best_epoch": best["epoch"],
                "best_validation": best["validation"],
                "training_seconds": round(elapsed, 2),
                "_trained_nlp": trained_nlp,
            }
        )

    selected = max(
        config_results,
        key=lambda result: result["best_validation"]["overall"]["f1"],
    )
    trained_nlp = selected["_trained_nlp"]
    selected_validation_f1 = selected["best_validation"]["overall"]["f1"]
    logger.info(
        "Selected configuration %s at epoch %s by validation F1 %.4f",
        selected["name"],
        selected["best_epoch"],
        selected_validation_f1,
    )

    logger.info("=== FINAL TEST EVALUATION (UNTOUCHED EVALUATION SET) ===")
    test_metrics = evaluate(trained_nlp, eval_examples)
    logger.info("Trained overall: %s", test_metrics["overall"])
    logger.info("Per-label: %s", test_metrics["per_label"])

    logger.info("Saving selected model to %s", MODEL_OUT)
    if os.path.isdir(MODEL_OUT):
        shutil.rmtree(MODEL_OUT)
    trained_nlp.to_disk(MODEL_OUT)

    public_config_results = []
    for result in config_results:
        clean_result = {key: value for key, value in result.items() if not key.startswith("_")}
        public_config_results.append(clean_result)

    report = {
        "seed": SEED,
        "annotation": {
            "training_labels": "weak supervision from en_core_web_sm (not human-reviewed)",
            "evaluation_labels": "span annotations in data/evaluation_examples.json",
            "matching": "exact-span+label",
            "labels_evaluated": sorted(EVAL_LABELS),
            "test_used_for_tuning": False,
            "entities_column": (
                "The provided entities column was empty for all 1,291 raw rows; "
                "it was inspected and not treated as gold."
            ),
            "evaluation_set_provenance": provenance,
        },
        "data_profile": _data_profile(data_report),
        "data_split": {
            "train": split_summary(train_data),
            "validation": split_summary(val_data),
            "test": split_summary(test_data),
        },
        "class_imbalance": _class_imbalance(train_data),
        "validation_tuning": {
            "selection_metric": "validation exact-span+label F1",
            "configs": public_config_results,
            "selected_config": selected["name"],
            "selected_epoch": selected["best_epoch"],
            "selected_validation_f1": selected_validation_f1,
            "test_used_for_selection": False,
        },
        "model_settings": {
            "base_model": "en_core_web_sm",
            "base_model_version": baseline_version,
            "spacy_version": spacy.__version__,
            "selected_config": selected["name"],
            "selected_epoch": selected["best_epoch"],
            "epochs": selected["settings"]["epochs"],
            "dropout": selected["settings"]["dropout"],
            "batch_size": selected["settings"]["batch_size"],
            "seed": selected["settings"]["seed"],
            "split_unit": "near-duplicate cluster / article identity",
            "split_ratio": "70/15/15",
        },
        "training_seconds": selected["training_seconds"],
        "training_seconds_total": round(
            sum(result["training_seconds"] for result in public_config_results), 2
        ),
        "losses_per_epoch": [record["ner_loss"] for record in selected["history"]],
        "validation_history": selected["history"],
        "evaluation_examples": {
            "count": len(eval_examples),
            "gold_spans": sum(len(example["gold_entities"]) for example in eval_examples),
            "matching": "exact-span+label",
            "labels": sorted(EVAL_LABELS),
            "provenance": provenance,
        },
        "baseline_metrics": baseline_metrics,
        "validation_metrics": selected["best_validation"],
        "test_metrics": test_metrics,
        "model_saved_to": os.path.relpath(MODEL_OUT, PROJECT_DIR).replace("\\", "/"),
    }

    report_path = os.path.join(DATA_DIR, "training_report.json")
    with open(report_path, "w", encoding="utf-8") as file_handle:
        json.dump(report, file_handle, indent=2, ensure_ascii=False)
    logger.info("Full report saved to %s", report_path)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Baseline F1 (test):       {baseline_metrics['overall']['f1']:.3f}")
    print(f"Trained  F1 (test):       {test_metrics['overall']['f1']:.3f}")
    print(f"Validation F1 (selected): {selected_validation_f1:.3f}")
    print(f"Selected configuration:   {selected['name']} @ epoch {selected['best_epoch']}")
    print(f"Model saved:              {MODEL_OUT}")
    print(f"Weak annotations:         {ANNOT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
