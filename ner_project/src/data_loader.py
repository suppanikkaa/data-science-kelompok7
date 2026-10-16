"""
Data Loader for Maybank Online Media CSV files.
Reads all 8 clipping CSV files defined in clipping-information.txt.
Delimiter: semicolon (;). Encoding: UTF-8-BOM.
"""

import csv
import os
import hashlib
import logging
from typing import Dict, List, Tuple

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
    """Strip spaCy leading quote artifact from exported fields."""
    v = value.strip()
    if v.startswith("'"):
        v = v[1:]
    return v


def load_all() -> Tuple[List[Dict], Dict]:
    """
    Load all clipping CSV files.
    Returns:
        articles: list of dicts with keys: title, body, language, original_id,
                  source_name, date_published, entities, clipping
        report:   data quality report dict
    """
    articles = []
    report = {
        "files": {},
        "total_rows": 0,
        "total_duplicates": 0,
        "missing_title": 0,
        "missing_body": 0,
        "language_distribution": {},
    }

    seen_ids = set()

    for clipping_name, filename in CLIPPING_FILES.items():
        path = os.path.join(DATA_DIR, filename)
        file_report = {
            "rows": 0,
            "duplicates": 0,
            "missing_title": 0,
            "missing_body": 0,
            "languages": {},
        }

        if not os.path.exists(path):
            logger.warning(f"File not found: {path}")
            report["files"][clipping_name] = {"error": "file not found"}
            continue

        with open(path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                # Normalize column names (strip BOM, whitespace)
                row = {k.strip().lstrip("\ufeff").strip("'"): _clean(v) for k, v in row.items()}

                title = row.get("title", "")
                body = row.get("body", "")
                lang = row.get("language", "")
                orig_id = row.get("original_id", "")

                # Deduplication by original_id; fallback to content hash
                if orig_id:
                    dup_key = orig_id
                else:
                    dup_key = hashlib.md5((title + body).encode()).hexdigest()

                if dup_key in seen_ids:
                    file_report["duplicates"] += 1
                    report["total_duplicates"] += 1
                    continue
                seen_ids.add(dup_key)

                if not title:
                    file_report["missing_title"] += 1
                    report["missing_title"] += 1
                if not body:
                    file_report["missing_body"] += 1
                    report["missing_body"] += 1

                file_report["rows"] += 1
                file_report["languages"][lang] = file_report["languages"].get(lang, 0) + 1
                report["language_distribution"][lang] = report["language_distribution"].get(lang, 0) + 1

                articles.append({
                    "title": title,
                    "body": body,
                    "language": lang,
                    "original_id": orig_id,
                    "source_name": row.get("source_name", ""),
                    "date_published": row.get("date_published", ""),
                    "entities": row.get("entities", ""),
                    "clipping": clipping_name,
                })

        report["files"][clipping_name] = file_report
        report["total_rows"] += file_report["rows"]
        logger.info(f"Loaded {clipping_name}: {file_report['rows']} rows")

    return articles, report


def load_english_only() -> List[Dict]:
    """Return only English articles (language == 'en')."""
    articles, _ = load_all()
    return [a for a in articles if a["language"] == "en"]


if __name__ == "__main__":
    import json
    arts, rep = load_all()
    print(json.dumps(rep, indent=2))
    print(f"\nTotal articles loaded: {len(arts)}")
