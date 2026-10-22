# Annotation Guidelines

Project: Maybank Malaysia Online Media NER  
Version: 1.0 (2026-10-22)  
Annotation unit: short article or verbatim article excerpt  
Evaluation matching: exact span + exact label

These guidelines define how the project team labels the eight entity types used
for evaluation. They exist because the provided CSV `entities` column is empty
for all 1,291 raw rows and because the training labels are weak labels produced
by `en_core_web_sm`, not gold annotations.

## 1. What is annotated

Each evaluation example is a **short excerpt from a held-out test-split CSV
article**. The title is copied verbatim. The body is a contiguous verbatim
substring (usually the first two to six sentences). The model always sees:

```text
title.strip() + "\n\n" + body.strip()
```

Gold offsets are character offsets into that combined string, not into the
original CSV row. Every example therefore stores:

- `original_id`: the source article identifier;
- `source_row`: the 1-based data row in the source CSV;
- `source_name`, `date_published`, `clipping`;
- `split` and `near_duplicate_cluster`;
- `excerpt_provenance.body_start_char` and `body_end_char` in the original body;
- `review_status`, `annotator`, `reviewer`, `reviewed_at`.

The excerpt itself is included in `data/evaluation_examples.json` because it is
short and is needed to reproduce offsets. Full articles and the raw CSVs are
never committed.

## 2. Label set

Only these eight labels are annotated and scored:

| Label | Definition | Typical examples |
|---|---|---|
| `PERSON` | Named people, including a contiguous honorific/title | `Datuk Seri Anwar Ibrahim`, `Ryan Giggs` |
| `ORG` | Named companies, banks, agencies, institutes, associations, political parties, sports clubs | `Maybank`, `Bank Negara Malaysia`, `Manchester United` |
| `GPE` | Countries, states, cities, and other geopolitical units | `Malaysia`, `Kuala Lumpur`, `Indonesia` |
| `LOC` | Non-GPE locations: regions, buildings, landmarks, bodies of water, mountains | `Menara Merdeka Maybank`, `West Asia`, `Asia Pacific` |
| `DATE` | Explicit dates, months, years, quarters, and date-like periods | `4 September`, `Q3 2026`, `21 to 23 September 2026` |
| `MONEY` | Monetary amounts with a currency marker or currency word | `RM2.5 billion`, `RM138,600`, `80 sen` |
| `EVENT` | Named events, awards, tournaments, conferences, campaigns | `Bursa Malaysia IR Awards 2026`, `World Cup`, `F1` |
| `NORP` | Nationalities, ethnic/religious groups, and nationality adjectives | `Malaysian`, `Asian`, `Southeast Asian` |

Out of scope for gold and for scoring: `CARDINAL`, `PERCENT`, `ORDINAL`,
`TIME`, `PRODUCT`, `LAW`, `WORK_OF_ART`, `FAC`, and other spaCy labels. The
service may still emit them; `evaluation.py` ignores those predictions.

## 3. Boundary rules

1. **Exact surface form.** The gold `text` must equal
   `combined[start:end]` exactly. Do not normalise case, punctuation, or
   whitespace.
2. **No nested or overlapping spans.** If an event name contains an
   organisation name, label the longest valid span with the event label. Do not
   also label the organisation inside that occurrence. If the organisation also
   appears separately elsewhere in the excerpt, annotate that separate mention.
3. **Include legal suffixes and acronyms.** `Malayan Banking Bhd`,
   `Hong Leong Bank Bhd`, and `Maybank Ageas Holdings Bhd` are complete ORG
   spans. Parentheses around an acronym are not part of the entity:
   `Malayan Banking Bhd (Maybank)` yields `Malayan Banking Bhd` and `Maybank`,
   not `(Maybank)`.
4. **Honorifics and titles.** Include a contiguous honorific/title immediately
   before a personal name: `Datuk Seri Anwar Ibrahim`,
   `Dato’ Sri Khairussaleh Ramli`, `Prof Dr Mohd Mizan Mohammad Aslam`.
   Do not include a generic role separated from the name by punctuation or a
   clause, such as `the President and Group CEO,`.
5. **Repeated entities.** Annotate every non-overlapping mention, including
   mentions in the title. Do not merge repeated mentions into one span.
6. **Possessives.** The possessive suffix is outside the span:
   `Maybank’s` → `Maybank` (ORG) + `’s`; `SURI’s` → `SURI` (ORG) + `’s`.
7. **Title shorthand.** A title shorthand may be annotated when the context is
   unambiguous. For example, `Hong Leong` in a title about Hong Leong Bank is
   ORG. This is recorded as a project decision; a reviewer may tighten it.
8. **Dates.** Annotate explicit dates and date-like periods. Vague relative
   expressions such as `recent months` or `today` are not annotated in the
   current set unless they are the only date reference and the reviewer agrees.
9. **Money.** `per cent` / `%` figures are not MONEY. Amounts such as
   `RM2.5 billion`, `RM138,600`, and `80 sen` are MONEY.
10. **Negative controls.** A negative-control excerpt has an empty
    `gold_entities` list. It is still a real CSV excerpt. If the model predicts
    a named entity in it, that prediction is a false positive.

## 4. Ambiguities resolved in this revision

These decisions are visible in the 12-example set and should be reviewed:

| Mention | Label | Reason |
|---|---|---|
| `Maybank Ageas` / `Maybank Ageas Holdings Bhd` | `ORG` | Insurance holding company, not a person or event |
| `Hong Leong` (title shorthand) | `ORG` | Refers to Hong Leong Bank in this context |
| `Bursa Malaysia Awards` | `EVENT` | Award event in the title; body gives the full name |
| `Bursa Malaysia` (separate mention) | `ORG` | The exchange, when not part of an event span |
| `F1`, `World Cup` | `EVENT` | Named sporting competitions |
| `Asean` | `ORG` | Regional organisation; `Southeast Asian` is the adjective (NORP) |
| `West Asia`, `Asia Pacific` | `LOC` | Regions, not sovereign states |
| `Menara Merdeka Maybank` | `LOC` | Building/landmark |
| `SURI` / `SURI Inspirasi` | `ORG` | Named social enterprise |
| `ANN ZAINUDDIN` | `PERSON` | Byline author |
| `80 sen` | `MONEY` | Monetary dividend amount |
| `third quarter (Q3) 2026` | `DATE` | Date-like period including the parenthetical abbreviation |

`iTEKAD`, `ROAR30`, and similar programme/strategy names are left out of the
current gold set because the eight-label scheme has no PROGRAM class.
A reviewer may decide to map them to `ORG`; that decision must be recorded in
`annotation_log.md` and applied consistently.

## 5. Review workflow

1. **Annotator draft.** The annotator selects held-out test excerpts, proposes
   spans, runs `validate_eval_examples()`, and records `review_status =
   "pending-human-review"` with `reviewer = null` and `reviewed_at = null`.
2. **Reviewer check.** A second person checks every excerpt and every span:
   text coverage, label, boundaries, overlaps, title+body offsets, and the
   negative control.
3. **Resolution.** Corrections are applied to
   `data/evaluation_examples.json`; disagreements and decisions are recorded in
   `annotation_log.md`.
4. **Sign-off.** Only after review may `review_status` become
   `"human-reviewed"`; `reviewer` and `reviewed_at` are mandatory. The evaluator
   rejects a `human-reviewed` example without those fields.
5. **Final evaluation.** Run `python ner_project/src/evaluate_model.py` after
   sign-off. The test metrics in the README must come from the signed-off file.
   Rebuilding the set from scratch resets the status to
   `pending-human-review`; use `mark_evaluation_reviewed.py` (or the builder's
   `--review-status human-reviewed --reviewer ... --reviewed-at ...` options)
   only after the human check.

## 6. Evaluation protocol

- Matching is **exact-span + exact-label**. No partial credit is given.
- A wrong boundary counts as one false positive and one false negative.
- The test set is never used for training or model selection. It is fixed
  before training; the baseline and final-model metrics are reported from the
  same set.
- Validation-based tuning uses weak-labelled validation articles only.
- The trained model is evaluated on the test set after the best configuration
  and epoch have been selected.
- Metrics must be reported honestly, including a regression relative to the
  baseline or the previous model version.

## 7. Data handling

- Raw CSV files and full article bodies are gitignored.
- `data/annotations.jsonl` contains only offsets, labels, IDs, split/cluster
  metadata, and a text hash; it contains no title or body text.
- `data/evaluation_examples.json` contains short verbatim excerpts because they
  are required to reproduce the gold offsets. Do not add full articles.
- The project does not redistribute the dataset or article text.
