"""
Build the held-out evaluation set from verbatim CSV excerpts.

The excerpts are selected from the deterministic test split (seed 42) and keep
``original_id`` plus ``source_row``.  Gold labels are a project-team draft for
this revision; ``review_status`` deliberately starts as ``pending-human-review``
so the file cannot claim human review before a person has checked it.

Run:
    python ner_project/src/build_evaluation_set.py

The output is data/evaluation_examples.json.  The script never copies full
articles: each body is a short prefix/except from the source row.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(__file__))
from annotate_and_train import split_articles
from data_loader import load_all
from evaluation import validate_eval_examples

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(PROJECT_DIR, "data", "evaluation_examples.json")

REVIEW_STATUS = "pending-human-review"


def nth_occurrence(text: str, needle: str, occurrence: int) -> int:
    """Return the zero-based start of the nth (0-based) occurrence of needle."""
    start = 0
    for index in range(occurrence + 1):
        position = text.find(needle, start)
        if position == -1:
            raise ValueError(
                f"Could not find occurrence {occurrence} of {needle!r} in excerpt"
            )
        start = position + 1
    return position


TARGETS: List[Dict[str, Any]] = [
    {
        "original_id": "31fb6fd395f5c4ff54ee6256aa1233f3458a492150a19296bf18e069f420174c",
        "body_end": "received on Sept 1.",
        "difficulty": "medium",
        "notes": "Parenthesised acronym, legal suffixes, repeated Maybank mentions, dates with abbreviations.",
        "gold": [
            ("Maybank", "ORG", 0),
            ("Maybank Ageas", "ORG", 0),
            ("KUALA LUMPUR", "GPE", 0),
            ("Malayan Banking Bhd", "ORG", 0),
            ("Maybank", "ORG", 2),
            ("Maybank Ageas Holdings Bhd", "ORG", 0),
            ("Etiqa", "ORG", 0),
            ("Malaysia", "GPE", 0),
            ("Singapore", "GPE", 0),
            ("Maybank", "ORG", 4),
            ("Ageas Insurance International NV", "ORG", 0),
            ("Sept 8", "DATE", 0),
            ("Bank Negara Malaysia", "ORG", 0),
            ("Sept 1", "DATE", 0),
        ],
    },
    {
        "original_id": "93de5ae7354705e4419481955201e53e08ae3ba72d9fd928b1772a403e37589b",
        "body_end": "according to the bank.",
        "difficulty": "medium",
        "notes": "Possessive apostrophes, honorifics, repeated company and person names, awards event.",
        "gold": [
            ("Maybank", "ORG", 0),
            ("Hong Leong", "ORG", 0),
            ("Bursa Malaysia Awards", "EVENT", 0),
            ("Bursa Malaysia", "ORG", 1),
            ("Maybank", "ORG", 1),
            ("Dato’ Sri Khairussaleh Ramli", "PERSON", 0),
            ("Hong Leong Bank", "ORG", 0),
            ("Malkit Singh Maan", "PERSON", 0),
            ("Bursa Malaysia IR Awards 2026", "EVENT", 0),
            ("4 September", "DATE", 0),
            ("Maybank", "ORG", 2),
            ("Khairussaleh", "PERSON", 1),
            ("Malkit", "PERSON", 1),
            ("Hong Leong Bank", "ORG", 1),
        ],
    },
    {
        "original_id": "a26427d94be0bb94fd01acff7c92b4971f7a923cc89284be1708af31583c9713",
        "body_end": "it said in a note.",
        "difficulty": "hard",
        "notes": "Abbreviated date/quarter, repeated year, percentage expressions left out of scope.",
        "gold": [
            ("CIMB Securities", "ORG", 0),
            ("2026", "DATE", 0),
            ("KUALA LUMPUR", "GPE", 0),
            ("CIMB Securities Research", "ORG", 0),
            ("Malaysia", "GPE", 0),
            ("2026", "DATE", 1),
            ("July", "DATE", 0),
            ("third quarter (Q3) 2026", "DATE", 0),
        ],
    },
    {
        "original_id": "e733790dec267d18a6c488e3447f6df08be145c89f6942bc31f07889faf7e615",
        "body_end": "RM1.10 billion and RM650 million, respectively.",
        "difficulty": "medium",
        "notes": "Money abbreviations and several numeric amounts; parenthesised company short name.",
        "gold": [
            ("Maybank", "ORG", 0),
            ("RM2.5bil", "MONEY", 0),
            ("KLSE Screener", "ORG", 0),
            ("KUALA LUMPUR", "GPE", 0),
            ("Malayan Banking Bhd", "ORG", 0),
            ("Maybank", "ORG", 1),
            ("RM2.5 billion", "MONEY", 0),
            ("Bursa Malaysia", "ORG", 0),
            ("RM750 million", "MONEY", 0),
            ("RM1.10 billion", "MONEY", 0),
            ("RM650 million", "MONEY", 0),
        ],
    },
    {
        "original_id": "91f7d72a0c4fdc0090ce2dcad7b2a2c766613fa5dca99da86777640df2a45d19",
        "body_end": "opportunities to generate income.",
        "difficulty": "hard",
        "notes": "Abbreviation BNM, possessive SURI, comma-formatted money, repeated organisation names.",
        "gold": [
            ("HLB Islamic", "ORG", 0),
            ("RM138,600", "MONEY", 0),
            ("SURI", "ORG", 0),
            ("KLSE Screener", "ORG", 0),
            ("KUALA LUMPUR", "GPE", 0),
            ("HLB Islamic", "ORG", 1),
            ("RM138,600", "MONEY", 1),
            ("SURI Inspirasi", "ORG", 0),
            ("Bank Negara Malaysia", "ORG", 0),
            ("BNM", "ORG", 0),
            ("SURI", "ORG", 2),
            ("RM14,000", "MONEY", 0),
        ],
    },
    {
        "original_id": "6bc249bb970713cb08f3af7c899347d071c98e5536d353ef8f374067a9fcd716",
        "body_end": "seeking a greater say in their own futures.",
        "difficulty": "hard",
        "notes": "Historical date, country abbreviation, region, nationality adjectives, repeated Bandung.",
        "gold": [
            ("Bandung", "GPE", 0),
            ("1955", "DATE", 0),
            ("US", "GPE", 0),
            ("Donald Trump", "PERSON", 0),
            ("West Asia", "LOC", 0),
            ("1955", "DATE", 1),
            ("Asian", "NORP", 0),
            ("African", "NORP", 0),
            ("Bandung", "GPE", 1),
            ("Indonesia", "GPE", 0),
        ],
    },
    {
        "original_id": "04e4bfb2beb86d2ecea5d01dafd34eea9ff02c2a0398c1b610c894c581b2a3f1",
        "body_end": "create a sense of shared national excitement.",
        "difficulty": "hard",
        "notes": "Title abbreviations, honorific plus person, institute and city, repeated Malaysia.",
        "gold": [
            ("F1", "EVENT", 0),
            ("World Cup", "EVENT", 0),
            ("Anwar", "PERSON", 0),
            ("Malaysia", "GPE", 0),
            ("KUALA LUMPUR", "GPE", 0),
            ("Oct 4", "DATE", 0),
            ("Datuk Seri Anwar Ibrahim", "PERSON", 0),
            ("Malaysia", "GPE", 1),
            ("Global Peace Institute", "ORG", 0),
            ("London", "GPE", 0),
            ("Prof Dr Mohd Mizan Mohammad Aslam", "PERSON", 0),
            ("Malaysia", "GPE", 2),
        ],
    },
    {
        "original_id": "5798999ae3e007d01e1da023040ca5ee90cd254cc03f296053b5e04bcbdc65ae",
        "body_end": "the Group’s ROAR30 sustainability ambitions.",
        "difficulty": "hard",
        "notes": "Event name with parenthetical acronym, building location, repeated Maybank, source organisation.",
        "gold": [
            ("BERNAMA", "ORG", 0),
            ("Maybank Group Sustainability Week 2026", "EVENT", 0),
            ("KUALA LUMPUR", "GPE", 0),
            ("Sept 25", "DATE", 0),
            ("Bernama", "ORG", 0),
            ("Maybank", "ORG", 1),
            ("Maybank Group Sustainability Week (MGSW) 2026", "EVENT", 0),
            ("21 to 23 September 2026", "DATE", 0),
            ("Menara Merdeka Maybank", "LOC", 0),
            ("MGSW 2026", "EVENT", 0),
            ("Maybank", "ORG", 4),
        ],
    },
    {
        "original_id": "aab2f8717a8977c22b2226a76705b78ea021cd9b1a117249cd500cde939be944",
        "body_end": "address complex business challenges and drive innovation.",
        "difficulty": "medium",
        "notes": "Repeated organisation and award names, nationality adjective, region.",
        "gold": [
            ("Red Hat", "ORG", 0),
            ("Hong Leong Bank", "ORG", 0),
            ("APAC Innovation Awards 2026", "EVENT", 0),
            ("Red Hat", "ORG", 1),
            ("Hong Leong Bank", "ORG", 1),
            ("Malaysian", "NORP", 0),
            ("Red Hat", "ORG", 3),
            ("APAC Innovation Awards 2026", "EVENT", 1),
            ("Asia Pacific", "LOC", 0),
            ("Red Hat", "ORG", 4),
        ],
    },
    {
        "original_id": "d07aee93f368db8b2bc52eac0d9670e5e7ba7dcc5769918cbecb901bd024945e",
        "body_end": "competitiveness on the continental stage.",
        "difficulty": "hard",
        "notes": "Person surname repeated, organisation with space, nationality adjective, two countries.",
        "gold": [
            ("Giggs", "PERSON", 0),
            ("Asean", "ORG", 0),
            ("KUALA LUMPUR", "GPE", 0),
            ("Manchester United", "ORG", 0),
            ("Ryan Giggs", "PERSON", 0),
            ("Southeast Asian", "NORP", 0),
            ("Giggs", "PERSON", 2),
            ("Premier League", "ORG", 0),
            ("Malaysia", "GPE", 0),
            ("Indonesia", "GPE", 0),
        ],
    },
    {
        "original_id": "7879fbe93d029e5036782d2e5b655d1a7b5784f34bea1a835bc733bf15826c6f",
        "body_end": "the entitlement date falls on Oct 8 at 5pm.",
        "difficulty": "medium",
        "notes": "Byline person, money in sen, bank name variant, several abbreviated dates.",
        "gold": [
            ("Hong Leong Bank", "ORG", 0),
            ("80 sen", "MONEY", 0),
            ("ANN ZAINUDDIN", "PERSON", 0),
            ("HONG Leong Bank Bhd", "ORG", 0),
            ("80 sen", "MONEY", 1),
            ("June 30, 2026", "DATE", 0),
            ("Bursa Malaysia", "ORG", 0),
            ("Oct 21, 2026", "DATE", 0),
            ("Oct 7", "DATE", 0),
            ("Oct 8", "DATE", 0),
        ],
    },
    {
        "original_id": "0af14a1c897afe00203b25d81b887c8c251b7ddce9afb000b42dc8e39422cd14",
        "body_start": "The research house also flagged the potential impact of higher bond yields on trading income and mark-to-market valuations.",
        "body_end": "mark-to-market valuations.",
        "difficulty": "medium",
        "notes": "Negative control: generic market commentary with no named entities in the excerpt.",
        "gold": [],
    },
]


def build(
    review_status: str = REVIEW_STATUS,
    reviewer: str | None = None,
    reviewed_at: str | None = None,
) -> List[Dict[str, Any]]:
    articles, _ = load_all()
    english = [article for article in articles if article["language"] == "en"]
    _, _, test_articles = split_articles(english)
    test_by_id = {article["original_id"]: article for article in test_articles}

    examples: List[Dict[str, Any]] = []
    for index, target in enumerate(TARGETS, start=1):
        original_id = target["original_id"]
        if original_id not in test_by_id:
            raise ValueError(f"Selected article {original_id} is not in the test split")
        article = test_by_id[original_id]

        body = article["body"]
        start_text = target.get("body_start")
        start = body.find(start_text) if start_text else 0
        if start == -1:
            raise ValueError(f"body_start not found in {original_id}")
        end = body.find(target["body_end"], start)
        if end == -1:
            raise ValueError(f"body_end not found in {original_id}")
        end += len(target["body_end"])
        body_excerpt = body[start:end].strip()

        title = article["title"].strip()
        combined = f"{title}\n\n{body_excerpt}"
        gold_entities = []
        for text_value, label, occurrence in target["gold"]:
            position = nth_occurrence(combined, text_value, occurrence)
            gold_entities.append(
                {
                    "text": text_value,
                    "label": label,
                    "start": position,
                    "end": position + len(text_value),
                }
            )

        example = {
            "id": f"eval_{index:03d}",
            "original_id": original_id,
            "source_row": article["source_row"],
            "source_name": article["source_name"],
            "date_published": article["date_published"],
            "clipping": article["clipping"],
            "split": "test",
            "near_duplicate_cluster": article["near_duplicate_cluster"],
            "title": title,
            "body": body_excerpt,
            "excerpt_provenance": {
                "body_start_char": start,
                "body_end_char": end,
                "selection": "verbatim excerpt from a deterministic held-out test-split article",
            },
            "gold_entities": gold_entities,
            "notes": target["notes"],
            "difficulty": target["difficulty"],
            "review_status": review_status,
            "annotator": "AI-assisted draft from CSV",
            "reviewer": reviewer,
            "reviewed_at": reviewed_at,
        }
        examples.append(example)

    validate_eval_examples(examples)
    return examples


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--review-status",
        default=REVIEW_STATUS,
        choices=["pending-human-review", "human-reviewed"],
        help="Review status to write (default: pending-human-review)",
    )
    parser.add_argument("--reviewer", default=None, help="Human reviewer name")
    parser.add_argument(
        "--reviewed-at",
        default=None,
        help="Review date (YYYY-MM-DD); required with human-reviewed status",
    )
    args = parser.parse_args()
    if args.review_status == "human-reviewed" and (
        not args.reviewer or not args.reviewed_at
    ):
        parser.error("--reviewer and --reviewed-at are required for human-reviewed")

    examples = build(args.review_status, args.reviewer, args.reviewed_at)
    with open(OUT_PATH, "w", encoding="utf-8") as file_handle:
        json.dump(examples, file_handle, indent=2, ensure_ascii=False)
    print(f"Wrote {len(examples)} examples / {sum(len(ex['gold_entities']) for ex in examples)} gold spans to {OUT_PATH}")
    print(f"Review status: {args.review_status}")
    for example in examples:
        labels = sorted({entity["label"] for entity in example["gold_entities"]})
        print(f"  {example['id']} | {example['difficulty']:<6} | {len(example['gold_entities']):>2} spans | {', '.join(labels) or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
