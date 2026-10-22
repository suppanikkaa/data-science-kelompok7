"""
Data Loader for Maybank Online Media CSV files.
Reads all 8 clipping CSV files defined in clipping-information.txt.
Delimiter: semicolon (;). Encoding: UTF-8-BOM.

The loader performs three provenance steps required by the study case:

1. Exact duplicate removal by ``original_id`` and by normalised article text.
2. A near-duplicate audit for English articles (title + body-prefix similarity).
3. Retention of ``original_id`` and a 1-based ``source_row`` for every article.

Raw article text is never committed.  ``source_row`` refers to the data row in
the original CSV (the header is row 1, so the first data row is row 2 in a
spreadsheet view; the field stores the 1-based data-row number).
"""

from __future__ import annotations

import csv
import difflib
import hashlib
import itertools
import logging
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple

csv.field_size_limit(10_000_000)  # some articles are very long

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "drive-download-20261016T000615Z-1-001")

CLIPPING_FILES = {
    "Hong Leong Bank":   "Maybank_Malaysia_OnlineMedia_07-Sep-2026_07-Oct-2026_9xtqAsPitG.csv",
    "CIMB Group Malaysia": "Maybank_Malaysia_OnlineMedia_07-Sep-2026_07-Oct-2026_bRQ9afwSIP.csv",
    "Public Bank Berhad": "Maybank_Malaysia_OnlineMedia_07-Sep-2026_07-Oct-2026_4sxH8Dz0hr.csv",
    "RHB Bank":          "Maybank_Malaysia_OnlineMedia_07-Sep-2026_07-Oct-2026_58yzyiBQ9U.csv",
    "MDV":               "Maybank_Malaysia_OnlineMedia_07-Sep-2026_07-Oct-2026_WdjZl0bRfZ.csv",
    "Maybank CSR":       "Maybank_Malaysia_OnlineMedia_07-Sep-2026_07-Oct-2026_i3BJEnT77K.csv",
    "Maybank Foundation": "Maybank_Malaysia_OnlineMedia_07-Sep-2026_07-Oct-2026_-nwub4cROX.csv",
    "Maybank Group":     "Maybank_Malaysia_OnlineMedia_07-Sep-2026_07-Oct-2026_hMxhx9QLTq.csv",
}

REQUIRED_COLS = {"title", "body", "language", "original_id", "source_name", "date_published", "entities"}


def _clean(value: str) -> str:
    """Strip the exported leading quote artifact from a CSV field."""
    v = (value or "").strip()
    if v.startswith("'"):
        v = v[1:]
    return v


def _normalise_for_dedup(text: str) -> str:
    """Normalise whitespace and case for exact-content deduplication."""
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _content_key(title: str, body: str) -> str:
    payload = _normalise_for_dedup(f"{title}\n{body}")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalise_for_similarity(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return difflib.SequenceMatcher(None, left, right, autojunk=False).ratio()


def find_near_duplicate_clusters(
    articles: List[Dict[str, Any]],
    *,
    title_threshold: float = 0.92,
    body_threshold: float = 0.90,
    body_chars: int = 2000,
) -> Tuple[List[List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """
    Group English articles that are near-duplicates.

    A pair is linked when its normalised titles are at least
    ``title_threshold`` similar and the first ``body_chars`` characters of the
    normalised bodies are at least ``body_threshold`` similar.  Linked articles
    form a cluster, and every cluster is assigned to one split later.  This is
    the documented alternative to deleting near-duplicates outright and keeps
    syndicated copies from leaking across train/validation/test.
    """
    items = [a for a in articles if a.get("language") == "en"]
    parent = list(range(len(items)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    pairs: List[Dict[str, Any]] = []
    for left_index, right_index in itertools.combinations(range(len(items)), 2):
        left = items[left_index]
        right = items[right_index]
        left_title = _normalise_for_similarity(left.get("title", ""))
        right_title = _normalise_for_similarity(right.get("title", ""))
        if not left_title or not right_title:
            continue

        # Cheap blocking filter before the more expensive sequence ratio.
        left_tokens = set(re.findall(r"[a-z0-9]+", left_title))
        right_tokens = set(re.findall(r"[a-z0-9]+", right_title))
        token_union = left_tokens | right_tokens
        if token_union and len(left_tokens & right_tokens) / len(token_union) < 0.50:
            continue

        title_similarity = _similarity(left_title, right_title)
        if title_similarity < title_threshold:
            continue
        body_similarity = _similarity(
            _normalise_for_similarity(left.get("body", ""))[:body_chars],
            _normalise_for_similarity(right.get("body", ""))[:body_chars],
        )
        if body_similarity < body_threshold:
            continue
        union(left_index, right_index)
        pairs.append(
            {
                "left_original_id": left.get("original_id", ""),
                "right_original_id": right.get("original_id", ""),
                "title_similarity": round(title_similarity, 4),
                "body_prefix_similarity": round(body_similarity, 4),
            }
        )

    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for index, article in enumerate(items):
        grouped[find(index)].append(article)

    clusters = sorted(
        grouped.values(),
        key=lambda group: min((article.get("original_id") or "") for article in group),
    )
    for cluster_index, group in enumerate(clusters, start=1):
        cluster_id = f"cluster_{cluster_index:04d}"
        for article in group:
            article["near_duplicate_cluster"] = cluster_id

    for article in articles:
        if article.get("language") != "en":
            article["near_duplicate_cluster"] = None

    return clusters, pairs


def load_all() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Load all clipping CSV files.

    Returns:
        articles: list of dicts with title, body, language, original_id,
                  source_row, source_name, date_published, url, entities,
                  clipping, content_hash, near_duplicate_cluster.
        report:   data-quality and near-duplicate audit report.
    """
    articles: List[Dict[str, Any]] = []
    report: Dict[str, Any] = {
        "files": {},
        "raw_rows": 0,
        "total_rows": 0,
        "total_duplicates": 0,
        "total_id_duplicates": 0,
        "total_content_duplicates": 0,
        "missing_title": 0,
        "missing_body": 0,
        "entities_nonempty": 0,
        "language_distribution": {},
    }

    seen_ids = set()
    seen_content = set()

    for clipping_name, filename in CLIPPING_FILES.items():
        path = os.path.join(DATA_DIR, filename)
        file_report: Dict[str, Any] = {
            "raw_rows": 0,
            "rows": 0,
            "id_duplicates": 0,
            "content_duplicates": 0,
            "missing_title": 0,
            "missing_body": 0,
            "entities_nonempty": 0,
            "languages": {},
        }

        if not os.path.exists(path):
            logger.warning("File not found: %s", path)
            report["files"][clipping_name] = {"error": "file not found"}
            continue

        with open(path, "r", encoding="utf-8-sig") as file_handle:
            reader = csv.DictReader(file_handle, delimiter=";")
            for source_row, row in enumerate(reader, start=1):
                file_report["raw_rows"] += 1
                report["raw_rows"] += 1

                # Normalise column names (strip BOM, whitespace, quote artifact)
                row = {
                    key.strip().lstrip("\ufeff").strip("'"): _clean(value)
                    for key, value in row.items()
                }

                title = row.get("title", "")
                body = row.get("body", "")
                language = row.get("language", "")
                original_id = row.get("original_id", "")
                entities = row.get("entities", "")

                # 1) Exact duplicate by original article identifier.
                if original_id and original_id in seen_ids:
                    file_report["id_duplicates"] += 1
                    report["total_id_duplicates"] += 1
                    report["total_duplicates"] += 1
                    continue

                # 2) Exact duplicate by normalised title + body (different IDs).
                content_hash = _content_key(title, body)
                if content_hash in seen_content:
                    file_report["content_duplicates"] += 1
                    report["total_content_duplicates"] += 1
                    report["total_duplicates"] += 1
                    continue

                if original_id:
                    seen_ids.add(original_id)
                seen_content.add(content_hash)

                if not title:
                    file_report["missing_title"] += 1
                    report["missing_title"] += 1
                if not body:
                    file_report["missing_body"] += 1
                    report["missing_body"] += 1
                if entities:
                    file_report["entities_nonempty"] += 1
                    report["entities_nonempty"] += 1

                file_report["rows"] += 1
                file_report["languages"][language] = file_report["languages"].get(language, 0) + 1
                report["language_distribution"][language] = (
                    report["language_distribution"].get(language, 0) + 1
                )

                articles.append(
                    {
                        "title": title,
                        "body": body,
                        "language": language,
                        "original_id": original_id,
                        "source_row": source_row,
                        "source_name": row.get("source_name", ""),
                        "date_published": row.get("date_published", ""),
                        "url": row.get("url", ""),
                        "entities": entities,
                        "clipping": clipping_name,
                        "content_hash": content_hash,
                    }
                )

        report["files"][clipping_name] = file_report
        report["total_rows"] += file_report["rows"]
        logger.info("Loaded %s: %s unique rows from %s raw rows", clipping_name, file_report["rows"], file_report["raw_rows"])

    clusters, pairs = find_near_duplicate_clusters(articles)
    multi_document_clusters = [group for group in clusters if len(group) > 1]
    report["near_duplicate_audit"] = {
        "method": (
            "English articles linked when normalised title similarity >= 0.92 "
            "and first 2000 characters of normalised body similarity >= 0.90"
        ),
        "thresholds": {
            "title_similarity": 0.92,
            "body_prefix_similarity": 0.90,
            "body_characters": 2000,
        },
        "english_documents": sum(1 for article in articles if article.get("language") == "en"),
        "clusters": len(clusters),
        "multi_document_clusters": len(multi_document_clusters),
        "documents_in_multi_document_clusters": sum(len(group) for group in multi_document_clusters),
        "largest_cluster_size": max((len(group) for group in clusters), default=0),
        "linked_pairs": len(pairs),
        "examples": sorted(
            pairs,
            key=lambda pair: (
                pair["title_similarity"],
                pair["body_prefix_similarity"],
            ),
            reverse=True,
        )[:10],
    }

    return articles, report


def load_english_only() -> List[Dict[str, Any]]:
    """Return only English articles (language == 'en')."""
    articles, _ = load_all()
    return [article for article in articles if article["language"] == "en"]


if __name__ == "__main__":
    import json

    arts, rep = load_all()
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    print(f"\nTotal articles loaded: {len(arts)}")
