# Annotation Log

Project: Maybank Malaysia Online Media NER  
Draft date: 2026-10-22; sign-off: 2026-10-23  
Guidelines: [`annotation_guidelines.md`](annotation_guidelines.md)

## Status

**Current status: HUMAN-REVIEWED (reviewer: adam, 2026-10-23).**  
The 12 evaluation excerpts and 122 gold spans were signed off on 2026-10-23.
The JSON records `review_status = "human-reviewed"`, `reviewer = "adam"`, and
`reviewed_at = "2026-10-23"`. The final test metrics were recomputed after
sign-off and are unchanged from the candidate run.

## Dataset provenance

| Item | Value |
|---|---|
| Source | Eight Maybank Malaysia online-media clipping CSVs |
| Raw rows | 1,291 |
| Exact duplicates removed by `original_id` | 386 |
| Exact duplicates removed by normalised `title + body` | 169 |
| Unique articles after exact deduplication | 736 |
| English articles used for NER | 577 |
| Other languages | ms 117, zh 28, id 12, vi 1, es 1 |
| Empty `entities` column in raw data | 1,291 / 1,291 |
| Split | 70/15/15 by near-duplicate cluster / article identity, seed 42 |
| Train / validation / test documents | 404 / 86 / 87 |

Near-duplicate audit (English): 522 clusters, 45 multi-document clusters, 100
documents in multi-document clusters, largest cluster 3, 65 linked pairs. The
thresholds are title similarity ≥ 0.92 and first-2,000-character body-prefix
similarity ≥ 0.90. Clusters are kept in one split; no cluster crosses the
train/validation/test boundary.

## Evaluation set

The set is in `data/evaluation_examples.json` and is built reproducibly by
`src/build_evaluation_set.py` from the deterministic held-out test split.

| Item | Value |
|---|---|
| Examples | 12 |
| Gold spans | 122 |
| Entity-rich examples | 11 |
| Negative-control excerpts | 1 (`eval_012`) |
| All eight labels represented? | Yes |
| Review status | `human-reviewed` (adam, 2026-10-23) |
| Annotator | AI-assisted draft from CSV |

Label counts:

| Label | Spans |
|---|---:|
| ORG | 48 |
| GPE | 20 |
| DATE | 16 |
| PERSON | 12 |
| MONEY | 10 |
| EVENT | 9 |
| NORP | 4 |
| LOC | 3 |

| ID | Original ID | Source | Difficulty | Spans | Purpose |
|---|---|---|---|---:|---|
| `eval_001` | `31fb6f…174c` | nst.com.my | medium | 14 | legal suffixes, parenthesised acronym, repeated ORG |
| `eval_002` | `93de5a…589b` | fintechnews.my | medium | 14 | honorifics, possessives, awards event |
| `eval_003` | `a26427…9713` | nst.com.my | hard | 8 | quarter/year boundary, PERCENT out of scope |
| `eval_004` | `e73379…e615` | klsescreener.com | medium | 11 | money abbreviations, several amounts |
| `eval_005` | `91f7d7…5d19` | klsescreener.com | hard | 12 | BNM acronym, SURI possessive, comma money |
| `eval_006` | `6bc249…d716` | aliran.com | hard | 10 | historical date, US, region, NORP |
| `eval_007` | `04e4bf…a3f1` | malaymail.com | hard | 12 | F1/World Cup events, honorific person, repeated Malaysia |
| `eval_008` | `579899…65ae` | bernama.com | hard | 11 | long event name with acronym, building LOC |
| `eval_009` | `aab2f8…e944` | 1side0.net | medium | 10 | repeated ORG/EVENT, nationality adjective, region |
| `eval_010` | `d07aee…945e` | nst.com.my | hard | 10 | surname repetition, Asean ORG, two GPEs |
| `eval_011` | `7879fb…6c6f` | luminews.my | medium | 10 | byline PERSON, money in sen, date abbreviations |
| `eval_012` | `0af14a…cd14` | businesstoday.com.my | medium | 0 | negative-control excerpt |

(The table shortens original IDs for readability; the JSON stores the full
identifier.)

## What was corrected from baseline suggestions

The baseline spaCy model was used only to propose candidates. The following
classes of corrections were applied before validation:

- Titles were annotated separately from body mentions; the baseline sometimes
  missed title-only mentions.
- `Maybank Ageas` and `Maybank Ageas Holdings Bhd` were labelled ORG, not
  PERSON.
- `Hong Leong` in the title was labelled ORG as a shorthand for Hong Leong Bank.
- Event names that contain an organisation name were labelled as one EVENT span
  rather than overlapping ORG + EVENT spans.
- Separate ORG mentions outside event spans were kept as ORG.
- Possessive suffixes (`’s`) were excluded from spans.
- `Bursa Malaysia IR Awards 2026`, `Maybank Group Sustainability Week (MGSW)
  2026`, and `APAC Innovation Awards 2026` were treated as EVENT spans.
- `Menara Merdeka Maybank` was labelled LOC, not ORG.
- `West Asia` and `Asia Pacific` were labelled LOC; `Asean` was labelled ORG
  and `Southeast Asian` NORP.
- `80 sen` and `RM…` amounts were labelled MONEY; `per cent` / `%` figures were
  left out of scope.
- `ANN ZAINUDDIN` (byline) was labelled PERSON.
- The negative control `eval_012` was chosen as a real entity-free sentence
  from a test-split article; its title also has no gold entities.

## Validation and review notes

Mechanical checks pass: every `text[start:end]` matches, spans are
non-overlapping, labels are in the eight-label set, and every example has
`original_id` and `source_row`. The reviewer signed off the current draft on
2026-10-23. The notes below remain useful for a future revision; they do not
change the signed-off spans.

1. Is `Hong Leong` in the title acceptable as ORG shorthand?
2. Should `iTEKAD` and `ROAR30` be ORG, or remain out of scope?
3. Is `third quarter (Q3) 2026` the desired DATE boundary, or should it be
   split into `third quarter` + `2026`?
4. Is `West Asia`/`Asia Pacific` better as LOC or GPE?
5. Is `Asean` ORG or NORP in the title phrase `Asean nations`?
6. Is the byline `ANN ZAINUDDIN` in scope as PERSON?
7. Is `eval_012` acceptable as a negative-control excerpt even though the full
   source article contains entities?

## Sign-off

| Role | Name | Date | Decision |
|---|---|---|---|
| Annotator draft | AI-assisted project draft | 2026-10-22 | 12 excerpts / 122 spans proposed |
| Reviewer / handoff | adam | 2026-10-23 | 12 excerpts / 122 spans signed off |

The sign-off was recorded with `mark_evaluation_reviewed.py`, and
`python ner_project/src/evaluate_model.py` was rerun after sign-off. For a
future revision, update every example's `review_status`, `reviewer`, and
`reviewed_at` fields, rerun the evaluator, and regenerate the metrics in the
README.
