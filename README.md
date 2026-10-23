# Article Named Entity Recognition (NER) Service
## Maybank Malaysia Online Media – Study Case

A Flask web service that extracts named entities (people, organizations, places,
dates, amounts, events, and nationality/group names) from news articles. It
uses a spaCy `en_core_web_sm` model fine-tuned on the provided Maybank Malaysia
online-media corpus.

The project now includes the study-case requirements that were missing from the
first version: exact deduplication and a near-duplicate audit, cluster-aware
70/15/15 splitting, validation-based model selection, a traceable held-out
evaluation set built from real CSV rows, annotation guidelines and logs, and
text-free annotation artifacts.

> **Review status.** The 12-excerpt / 122-span evaluation set is marked
> `human-reviewed` (reviewer: **adam**, 2026-10-23). The test metrics below are
> computed from that signed-off file. The set is still small, so treat the
> numbers as a small-sample estimate rather than a production benchmark. See
> [Reviewing the evaluation set](#reviewing-the-evaluation-set).

---

## Architecture

```
┌──────────────────────────────────────────────────┐
│              Browser Panel (/)                   │
│  [Title Input] + [Body Textarea] + [Submit Btn]  │
└─────────────────────┬────────────────────────────┘
                      │ POST /extract-entities (JSON)
┌─────────────────────▼────────────────────────────┐
│           Flask Backend (src/app.py)              │
│  • Input validation (title, body, length, JSON)   │
│  • Calls ner_pipeline.extract_entities()          │
└─────────────────────┬────────────────────────────┘
                      │
┌─────────────────────▼────────────────────────────┐
│         NER Pipeline (src/ner_pipeline.py)        │
│  • Lazy-loads spaCy model (trained > baseline)    │
│  • Combines title + "\n\n" + body                 │
│  • Returns entities with text, label, start, end  │
└──────────────────────────────────────────────────┘
```

The service accepts up to **50,000 characters** of combined title + body text.
Character offsets always refer to that combined string.

A separate read-only **`/evaluation`** page renders the checked-in model and
evaluation artifacts (`training_report.json`, `evaluation_report.json`, and
`evaluation_examples.json`) without calling the NER pipeline or exposing article
text.

### Dataset

The loader expects the eight CSV files and `clipping-information.txt` in a
folder named `drive-download-20261016T000615Z-1-001`, next to `ner_project/`:

```
data-science-kelompok7-fresh/
├── ner_project/
└── drive-download-20261016T000615Z-1-001/
    ├── Maybank_*.csv
    └── clipping-information.txt
```

The folder is intentionally gitignored. Do not commit the raw articles or CSV
files. On Windows, the files can be placed with:

```powershell
$src = "D:\Downloads\dataset_online_news_malaysia-*\dataset_online_news_malaysia"
$dst = ".\drive-download-20261016T000615Z-1-001"
New-Item -ItemType Directory -Force -Path $dst | Out-Null
Copy-Item "$src\DATASET\*.csv" $dst -Force
Copy-Item "$src\clipping-information.txt" $dst -Force
```

#### Data profile after exact deduplication

| Metric | Value |
|---|---:|
| Raw rows in the eight CSVs | 1,291 |
| Duplicate rows removed by `original_id` | 386 |
| Duplicate rows removed by normalised `title + body` | 169 |
| **Total duplicate rows removed** | **555** |
| Unique articles after deduplication | **736** |
| Missing titles / bodies | 0 / 0 |
| Non-empty `entities` column | **0 / 1,291** |
| English articles used for NER | **577** |
| Other languages | ms 117, zh 28, id 12, vi 1, es 1 |

| Clipping name | Raw rows | Unique | ID dupes | Content dupes | English |
|---|---:|---:|---:|---:|---:|
| Hong Leong Bank | 96 | 82 | 0 | 14 | 70 |
| CIMB Group Malaysia | 347 | 273 | 29 | 45 | 232 |
| Public Bank Berhad | 2 | 2 | 0 | 0 | 0 |
| RHB Bank | 105 | 55 | 38 | 12 | 52 |
| MDV | 18 | 18 | 0 | 0 | 16 |
| Maybank CSR | 160 | 82 | 60 | 18 | 78 |
| Maybank Foundation | 29 | 6 | 16 | 7 | 6 |
| Maybank Group | 534 | 218 | 243 | 73 | 123 |
| **Total** | **1,291** | **736** | **386** | **169** | **577** |

The provided `entities` column is empty in every raw row. It was inspected and
is **not** treated as gold. Training labels are weak labels produced by the
baseline `en_core_web_sm` model; only the evaluation set has human-reviewed
span annotations.

#### Duplicate and near-duplicate handling

1. **Exact duplicates** are removed by `original_id` and then by a SHA-256 hash
   of normalised `title + body` (whitespace collapsed, lowercased). This catches
   syndicated copies that have different article IDs.
2. **Near-duplicates** among the 577 English articles are grouped before
   splitting. The audit links two articles when normalised title similarity is
   at least **0.92** and the first **2,000 characters** of the normalised bodies
   have similarity at least **0.90** (Python `difflib.SequenceMatcher`).
3. Near-duplicate clusters are **kept entirely in one split**. They are not
   deleted silently; the audit is written to `data/training_report.json`.

Current English near-duplicate audit:

| Metric | Value |
|---|---:|
| English documents | 577 |
| Near-duplicate clusters | 522 |
| Multi-document clusters | 45 |
| Documents in multi-document clusters | 100 |
| Largest cluster | 3 documents |
| Linked pairs | 65 |

Rerun the profile with:

```bash
python -c "import sys; sys.path.insert(0, 'ner_project/src'); from data_loader import load_all; a, r = load_all(); print(r)"
```

### Data split

The model uses a **70/15/15 cluster-aware split** with random seed **42**. A
cluster is never split across train, validation, and test.

| Split | Documents | Near-duplicate clusters | Weak spans (all labels) | Weak spans (8 evaluation labels) |
|---|---:|---:|---:|---:|
| Train | 404 | 360 | 37,061 | 28,481 |
| Validation | 86 | 83 | 5,707 | 4,308 |
| Test | 87 | 79 | 7,379 | 5,872 |

Only English articles are used for training. The training labels are **weak
labels** produced by baseline `en_core_web_sm`, not human-reviewed annotations.
The validation split therefore measures agreement with those weak labels, not
true accuracy. The 87-document test split is held out from training; it is not
used for model selection. The 12-excerpt evaluation set is drawn from that test
split and is the only set used for the headline test metrics.

Class imbalance is substantial. Of all weak training spans (including
out-of-scope labels), ORG accounts for **28.2%**, PERSON **15.5%**, DATE
**15.3%**, and GPE **12.5%**, while MONEY is **1.5%**, EVENT **0.3%**, and LOC
**0.8%**. The evaluation set deliberately includes EVENT, LOC, and MONEY
examples, but the rare-class scores are still unstable.

### Evaluation set

`data/evaluation_examples.json` contains **12 short excerpts / 122 gold spans**
from 12 different held-out test articles: 11 entity-rich excerpts plus one
negative-control excerpt with no gold entities. Every excerpt is a verbatim
substring of a real test article and keeps:

- `original_id` and 1-based `source_row`;
- `source_name`, `date_published`, and `clipping`;
- `split = "test"` and the near-duplicate cluster ID;
- the original body start/end offsets for the excerpt;
- `review_status`, `annotator`, `reviewer`, and `reviewed_at`.

Label coverage: ORG 48, GPE 20, DATE 16, PERSON 12, MONEY 10, EVENT 9, NORP 4,
LOC 3. The set is reproducible with `src/build_evaluation_set.py`; it is not a
set of hand-written examples and it does not overlap the training articles.

The excerpt selection deliberately covers repeated entities, titles,
possessives, honorifics, legal suffixes, abbreviations, dates with quarters,
comma-formatted money, nested event/organisation names, a building, a region,
nationality adjectives, and a negative control. The detailed label and boundary
rules are in [`docs/annotation_guidelines.md`](docs/annotation_guidelines.md);
the signed-off decisions and review notes are in
[`docs/annotation_log.md`](docs/annotation_log.md).

---

## Setup & Installation

Run these commands from the repository root (the folder that contains
`ner_project/` and the dataset).

### Prerequisites
- Python 3.11+ (tested on Python 3.14.7)
- Virtual environment (recommended)

### 1. Create virtual environment
```bash
python3 -m venv venv          # Linux/macOS
py -m venv venv               # Windows
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate.bat     # Windows cmd
venv\Scripts\Activate.ps1     # Windows PowerShell
```

### 2. Install dependencies
```bash
pip install -r ner_project/requirements.txt
python -m spacy download en_core_web_sm
```

Place the dataset as described above before running the training pipeline.

### 3. Run the evaluation (no retraining)

This loads the existing fine-tuned model and the evaluation set, validates every
gold span, and writes `data/evaluation_report.json`:

```bash
python ner_project/src/evaluate_model.py
```

Useful options:

```bash
python ner_project/src/evaluate_model.py \
  --examples ner_project/data/evaluation_examples.json \
  --trained-model ner_project/data/trained_model \
  --out ner_project/data/evaluation_report.json
```

If the evaluation set is still pending human review, the command prints a
warning and records `all_human_reviewed: false` in the report.

### 4. (Optional) Rebuild the evaluation set and retrain

Rebuild the held-out set from the deterministic test split:

```bash
python ner_project/src/build_evaluation_set.py
```

Train the model (this takes several minutes on CPU):

```bash
python ner_project/src/annotate_and_train.py
```

The training script:

- loads and profiles all eight CSV files;
- removes exact duplicates and groups near-duplicates;
- auto-annotates English articles with baseline spaCy (weak supervision);
- splits 70/15/15 by near-duplicate cluster / article identity (seed 42);
- fine-tunes two documented configurations for 3 epochs each;
- evaluates validation after **every epoch** and keeps the best epoch per
  configuration;
- selects the final configuration by validation exact-span F1;
- evaluates the untouched test set **after** selection;
- saves the selected model to `data/trained_model/`;
- writes `data/training_report.json` and a text-free
  `data/annotations.jsonl`.

The evaluation set is never overwritten by the training script. Rebuilding it
resets review status to `pending-human-review` unless you explicitly pass the
review options.

### 5. Reviewing the evaluation set

The current set is signed off by **adam** (2026-10-23). Use the commands below
only if labels change or the set needs another review.

Print every excerpt, gold span, and context:

```bash
python ner_project/src/review_evaluation_set.py
```

After a human has checked every span, record sign-off:

```bash
python ner_project/src/mark_evaluation_reviewed.py --reviewer "Nama Reviewer"
```

The signed-off set is already reflected in the metrics below. If you change any
label, rerun the evaluation command and update the metrics. The evaluation
validator refuses a `human-reviewed` example without a reviewer and date.

### 6. Start the service

```bash
python ner_project/src/app.py
```

The service runs at **http://localhost:5000**. The NER panel is at `/`; a
model and evaluation summary is at `/evaluation`. Health check:

```bash
curl http://localhost:5000/health
```

---

## Evaluation Methodology

| Choice | Value |
|---|---|
| Evaluation set | 12 excerpts: 11 entity-rich + 1 negative control |
| Gold spans | 122 |
| Text unit | `title + "\n\n" + body` |
| Labels evaluated | PERSON, ORG, GPE, LOC, DATE, MONEY, EVENT, NORP |
| Matching | **Exact-span + exact-label** |
| Repeated mentions | Every non-overlapping mention is annotated, including title mentions |
| Nested spans | Not annotated; the longest valid span is kept |
| Out-of-scope labels | CARDINAL, PERCENT, ORDINAL, TIME and other spaCy labels are ignored for scoring |
| Test used for tuning? | **No** |
| Validation used for tuning? | **Yes** — best epoch and configuration by validation F1 |
| Review status | **human-reviewed** (adam, 2026-10-23) |

Gold spans are checked with `evaluation.validate_eval_examples()`: for every
entity, `combined[start:end] == text`, spans are non-overlapping, labels are in
the supported set, and provenance is present. The service may still return
other spaCy labels; those predictions are not part of the reported metrics.

Exact-span matching is deliberately strict. A prediction such as `the third
quarter of 2026` does **not** match the gold span `third quarter of 2026`; it
counts as one false positive and one false negative. This exposes boundary
errors that a partial-span or text-only evaluation would hide.

---

## Results

### Overall (held-out evaluation set)

| Model | Precision | Recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| Baseline `en_core_web_sm` | 0.4701 | 0.4508 | 0.4603 | 55 | 62 | 67 |
| Fine-tuned `dropout 0.25 / batch 8`, epoch 1 | 0.4839 | 0.4918 | 0.4878 | 60 | 64 | 62 |

The fine-tuned model improves F1 from **0.4603 to 0.4878** (+0.0275 absolute, about **+6.0% relative**), with precision rising from 0.4701 to 0.4839 and recall from 0.4508 to 0.4918. It gains 5 true positives while adding 2 false positives and removing 5 false negatives; macro F1 rises from 0.4668 to 0.4941. The gain is modest and uneven: GPE, LOC, NORP, and DATE are relatively strong, while EVENT is 0.000, MONEY 0.125, and PERSON 0.345. These test metrics come from the human-reviewed evaluation set (reviewer: adam, 2026-10-23); the set is small, so treat the numbers as a small-sample estimate.

### Per-label metrics (fine-tuned model)

| Label | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| DATE | 0.483 | 0.875 | 0.622 | 16 |
| EVENT | 0.000 | 0.000 | 0.000 | 9 |
| GPE | 0.923 | 0.600 | 0.727 | 20 |
| LOC | 1.000 | 0.667 | 0.800 | 3 |
| MONEY | 0.167 | 0.100 | 0.125 | 10 |
| NORP | 0.800 | 1.000 | 0.889 | 4 |
| ORG | 0.431 | 0.458 | 0.444 | 48 |
| PERSON | 0.294 | 0.417 | 0.345 | 12 |

### Validation vs test

| Split | Labels | Precision | Recall | F1 |
|---|---|---|---:|---:|
| Validation (selected epoch 1, weak labels) | 4,308 weak spans | 0.8217 | 0.8122 | 0.8170 |
| Test (baseline) | 122 human-reviewed spans | 0.4701 | 0.4508 | 0.4603 |
| Test (fine-tuned) | 122 human-reviewed spans | 0.4839 | 0.4918 | 0.4878 |

Validation numbers measure agreement with baseline-generated weak labels and are
only a monitoring signal. The test row is the number to report for the study
case. The test set was not used for training or model selection.

### Validation-based model selection

| Configuration | Best epoch | Validation precision | Validation recall | Validation F1 | Training time |
|---|---:|---:|---:|---:|---:|
| `dropout 0.35 / batch 8` | 1 | 0.7897 | 0.7878 | 0.7888 | 624.1 s |
| `dropout 0.25 / batch 8` | 1 | 0.8217 | 0.8122 | **0.8170** | 568.2 s |

Both configurations peaked at epoch 1; later epochs reduced validation F1 (0.35: 0.7888 → 0.7793 → 0.7792; 0.25: 0.8170 → 0.7863 → 0.8133). The selected model is `dropout 0.25 / batch 8`, epoch 1. Total tuning time was 1,192.3 s (~19.9 min) on CPU; the selected configuration took 568.2 s.

The selection criterion was validation exact-span + label F1. The final test
metrics were computed once after the winning configuration and epoch were
chosen. This avoids selecting a model on the test set.

### Error analysis

`data/evaluation_report.json` keeps every false positive and false negative with
context. Representative errors from the fine-tuned model:

- **EVENT is effectively unlearned.** Gold `Bursa Malaysia IR Awards 2026` and `Maybank Group Sustainability Week (MGSW) 2026` are missed; the model emits ORG fragments (`Bursa Malaysia IR Awards`, `BERNAMA - Maybank Group Sustainability`) and DATE fragments (`2026`, `Week 2026`). EVENT has 9 gold spans, 0 true positives, and 1 false positive.
- **MONEY is unreliable.** Gold `RM2.5 billion`, `RM2.5bil`, `RM750 million`, `RM138,600`, `RM14,000`, and `80 sen` are missed; `RM138,600` is labelled ORG in one place, while percentages such as `30.95 per cent`, `five per cent`, `4.8 per cent`, and `5.0 per cent` are labelled MONEY. This creates both false negatives and false positives.
- **Title/body boundary and GPE.** In `eval_004` and `eval_005`, the model emits `KLSE Screener\n\nKUALA` as PERSON and misses `KUALA LUMPUR` GPE; `KUALA LUMPUR` is also missed in `eval_001`, `eval_003`, `eval_007`, `eval_008`, and `eval_010`. The `\n\n` boundary disturbs tokenisation/context.
- **Person boundaries and honorifics.** `Datuk Seri Anwar Ibrahim` becomes `Datuk Seri` + `Anwar Ibrahim’s` (two false positives and a missed full span); `Prof Dr Mohd Mizan Mohammad Aslam` becomes `Prof Dr` + `Mohd Mizan Mohammad Aslam`; `Manchester United` is labelled PERSON, while byline `ANN ZAINUDDIN` is labelled ORG.
- **ORG over-prediction.** Generic title text is tagged ORG (`Hong Leong Win Best`, `CFO Honours`, `CFO`, `Best Company`, `Best Financial Reporting`, and `Asset Quality Risks Build` in the negative control), while `Maybank` is missed in several title/body mentions. ORG alone contributes 29 FP and 26 FN.
- **Determiner boundaries.** `World Cup` is predicted as `the World Cup`, and `third quarter of 2026` as `the third quarter of 2026`; exact-span matching counts these as FP+FN even though the semantic entity is right.

These examples come from `data/evaluation_report.json`, computed on the human-reviewed evaluation set.

**What would improve the model:** a human-corrected training set (especially for
EVENT, LOC, MONEY, and Malaysian person names), deterministic boundary
normalisation for leading determiners, a money pattern for `RM…`/`…million`/
`…billion`, gazetteers for Malaysian organisations and place names, and more
human-reviewed test examples for rare classes. These are more valuable than a
larger weak-labelled set.

---

## API Usage

### POST /extract-entities

**Request:**
```json
{
  "title": "Maybank expands in Kuala Lumpur",
  "body": "Maybank Group CEO Datuk Khairussaleh Ramli said the bank will expand in Kuala Lumpur on 4 September 2026."
}
```

**Response (200 OK):**
```json
{
  "entities": [
    {
      "text": "Kuala Lumpur",
      "label": "GPE",
      "start": 19,
      "end": 31
    },
    {
      "text": "Maybank Group",
      "label": "ORG",
      "start": 33,
      "end": 46
    },
    {
      "text": "Datuk Khairussaleh Ramli",
      "label": "PERSON",
      "start": 51,
      "end": 75
    },
    {
      "text": "Kuala Lumpur",
      "label": "GPE",
      "start": 105,
      "end": 117
    },
    {
      "text": "4 September 2026",
      "label": "DATE",
      "start": 121,
      "end": 137
    }
  ],
  "meta": {
    "entity_count": 5,
    "model": "trained-model",
    "text_length": 138
  }
}
```

The offsets are zero-based character indices into:

```text
Maybank expands in Kuala Lumpur\n\nMaybank Group CEO Datuk Khairussaleh Ramli said the bank will expand in Kuala Lumpur on 4 September 2026.
```

The combined string is **138 characters** long. The `meta.text_length` field
reports that combined length, not the body length alone. This sample exercises
ORG, PERSON, GPE, and DATE. It does **not** claim that MONEY or EVENT extraction
works well; see the error analysis above. The service accepts at most **50,000
characters**.

**Error (400 Bad Request):**
```json
{"error": "title must not be empty"}
```

### GET /health
```json
{
  "status": "ok",
  "model": "trained-model",
  "supported_labels": ["CARDINAL", "DATE", "EVENT", "FAC", "GPE", "LANGUAGE", "LAW", "LOC", "MONEY", "NORP", "ORDINAL", "ORG", "PERCENT", "PERSON", "PRODUCT", "QUANTITY", "TIME", "WORK_OF_ART"],
  "max_text_length": 50000
}
```

### GET /evaluation

Browser page summarising the selected model, dataset profile, validation tuning,
per-label held-out metrics, and human-review status. It reads
`data/training_report.json`, `data/evaluation_report.json`, and
`data/evaluation_examples.json` at request time. It does not expose article
text.

---

## Supported Entity Labels

| Label | Description | Example |
|---|---|---|
| PERSON | Person names | Datuk Khairussaleh Ramli |
| ORG | Organizations / companies | Maybank Group, CIMB |
| GPE | Geopolitical entities | Malaysia, Kuala Lumpur |
| LOC | Non-GPE locations | Menara Maybank |
| DATE | Dates / time periods | Q3 2026, 15 September 2026 |
| MONEY | Monetary values | RM2.5 billion |
| EVENT | Named events / awards | Euromoney Awards 2026 |
| NORP | Nationalities / political groups | Malaysian, ASEAN |

The service may emit additional spaCy labels such as CARDINAL, PERCENT, ORDINAL
or TIME. Those are outside the eight labels used by the evaluation and are
ignored when computing the reported metrics.

---

## Running Tests

From the repository root:

```bash
python -m pytest ner_project/tests -q
```

The suite contains **36 tests** covering:

- normal input (English, Malay, punctuation);
- empty title/body validation;
- maximum-length enforcement;
- malformed JSON and wrong content type;
- repeated entities and deterministic output;
- API endpoint responses (health, extract, HTML panel, model/evaluation page);
- evaluation span validation, exact-span matching, per-label counts,
  out-of-scope label filtering, and FP/FN context;
- repository evaluation-set provenance and metadata;
- text-free `annotations.jsonl` checks;
- near-duplicate clustering and cluster-aware splitting;
- the rule that a `human-reviewed` claim requires a reviewer and date.

The test `test_repository_evaluation_examples_are_valid` fails immediately if any
gold offset stops pointing at its own entity text, preventing the earlier offset
bug from silently returning to the report.

---

## Model Details

| Setting | Value |
|---|---|
| Base model | `en_core_web_sm` 3.8.0 (spaCy 3.8.16) |
| Configurations tried | `dropout 0.35 / batch 8` and `dropout 0.25 / batch 8` |
| Epochs per configuration | 3 |
| Selection | Best epoch per configuration, then best configuration by validation exact-span F1 |
| Selected configuration | `dropout 0.25 / batch 8` |
| Selected epoch | 1 (best validation F1) |
| Selected validation F1 | 0.8170 (P 0.8217, R 0.8122) |
| Seed | 43 (config 2); config 1 used seed 42 |
| Data split | 70% / 15% / 15% |
| Split unit | Near-duplicate cluster / article identity |
| Train / validation / test documents | 404 / 86 / 87 |
| Training labels | Weak supervision from baseline spaCy |
| Final test labels | Human-reviewed project annotations (reviewer: adam, 2026-10-23) |
| Annotation format | Text-free JSONL (`data/annotations.jsonl`) |
| Matching criterion | Exact-span + label |
| Test used for tuning | No |
| Training duration | 1,192.3 s total (~19.9 min) for both configurations on CPU; selected configuration 568.2 s. Weak annotation took about 3.2 min before training. |
| Model artifact | spaCy pipeline saved at `data/trained_model` |

### Model choice rationale

`en_core_web_sm` was selected because:

- the majority of unique articles are in English (577 of 736);
- it supports all required entity types (PERSON, ORG, GPE, LOC, DATE, MONEY,
  EVENT, NORP);
- it is small and fast for a prototype service;
- it supports fine-tuning without a GPU.

For production, `en_core_web_trf` (transformer-based) would likely yield higher
accuracy, at a much higher CPU/GPU cost.

---

## Known Limitations

1. **Small evaluation set.** The 12-excerpt / 122-span set is human-reviewed
   but small, especially for EVENT (9 spans), LOC (3), and NORP (4); per-label
   metrics are therefore noisy.
2. **English-only.** The NER model works best on English text; Malay and Chinese
   articles will produce poor results.
3. **Malaysian entities.** Malaysian proper nouns (Datuk Seri, Dewan Rakyat,
   Bursa Malaysia) may be misclassified.
4. **Abbreviations.** Acronyms like HLB, MDV, PBB may not be detected as ORG.
5. **Honorifics.** Malaysian honorifics (Dato', Tan Sri, Encik) are sometimes
   merged with person names or split into separate entities.
6. **Weak supervision.** Training uses auto-annotation, not fully human-reviewed
   labels; this introduces label noise. Only the evaluation set has
   human-reviewed span annotations.
7. **Exact-span sensitivity.** Determiners and punctuation at span boundaries
   cause false positives/negatives even when the entity is semantically correct.
8. **Out-of-scope labels.** The service can emit CARDINAL/PERCENT/ORDINAL/TIME
   predictions that are not part of the eight-label evaluation; the UI may show
   them.
9. **Small test set.** The 122-span evaluation set is larger than the first
   version but still small, especially for EVENT (9) and LOC (3).
10. **Near-duplicate threshold.** The near-duplicate audit uses a documented
    sequence-similarity heuristic; a different threshold could change the
    cluster count and split.

---

## Responsible Data Use

- Articles are used only for NER training/demo; no redistribution.
- Raw CSVs and full article text are gitignored and are not committed.
- `data/annotations.jsonl` is text-free: it stores IDs, offsets, labels, split
  metadata, and a SHA-256 hash, never the article text.
- `data/evaluation_examples.json` contains short verbatim excerpts because the
  original text is required to reproduce offsets; it does not contain full
  articles.
- No personal data is stored by the service (stateless).
- Biases: the dataset focuses on Malaysian banking; the model may underperform
  on other domains.
- Copyright: articles belong to their respective publishers; do not commit raw
  article text.

---

## Reproducibility and Artifacts

| Artifact | Purpose |
|---|---|
| `src/data_loader.py` | CSV loading, exact dedup, near-duplicate audit, provenance |
| `src/annotate_and_train.py` | Weak annotation, cluster split, validation tuning, training |
| `src/build_evaluation_set.py` | Rebuilds the 12-excerpt evaluation set from the test split |
| `src/evaluate_model.py` | Baseline vs fine-tuned evaluation on the evaluation set |
| `src/evaluation.py` | Span validation and exact-span metrics |
| `src/review_evaluation_set.py` | Prints excerpts and spans for human review |
| `src/mark_evaluation_reviewed.py` | Records human sign-off after review |
| `src/templates/evaluation.html` | Model-card page rendered at `/evaluation` |
| `data/evaluation_examples.json` | Evaluation excerpts, gold spans, and provenance |
| `data/annotations.jsonl` | Text-free weak annotations for all 577 English articles |
| `data/training_report.json` | Data profile, split, tuning history, and metrics |
| `data/evaluation_report.json` | Reproducible baseline/trained evaluation output |
| `docs/annotation_guidelines.md` | Label definitions and boundary rules |
| `docs/annotation_log.md` | Signed-off decisions, corrections, and review notes |
| `tests/test_evaluation.py` | Evaluation, provenance, annotation, and split tests |

---

## Questions Answered (Study Case)

**Q1. What makes a text span a named entity?**
A named entity is a real-world object referred to by name: a person, place,
organization, date, amount, event, or group. "Maybank" is an ORG; "bank" alone
is not. "Malaysia" is a GPE; "Malaysian" is a NORP.

**Q2. Why combine title and body?**
The title often contains the article's main entity (e.g., "Maybank posts
profit"). Processing independently would lose context, and title mentions would
be missed. The combined string is the same one used for training and evaluation,
so offsets are consistent.

**Q3. Label mapping to user-facing categories:**
PERSON→People, ORG→Organizations, GPE+LOC→Places, DATE→Dates, MONEY→Amounts,
EVENT→Events, NORP→Nationalities/groups.

**Q4. Tokenization / language effects:**
spaCy tokenizes on whitespace and punctuation. Malay text (e.g., "RM500") may
tokenize differently. Capitalization guides NER; lowercased text loses
person/org signals. Chinese text requires a different model.

**Q5. Text vs label vs offset:**
`text` is the surface string ("Maybank"), `label` is the category (ORG),
`start`/`end` are zero-based character indices in the combined title+body
string. The evaluator asserts that `combined[start:end] == text` before scoring.
For the sample request above, the combined string is 138 characters; the sample
response uses offsets from that combined string (`Kuala Lumpur` at 19 and 105,
`Maybank Group` at 33, `Datuk Khairussaleh Ramli` at 51, and
`4 September 2026` at 121), not from the title or body alone.

**Q6. Same entity, different spellings:**
"CIMB", "CIMB Group", "CIMB Bank" are treated as separate mentions.
Normalization (entity linking) would be needed to merge them. The evaluation set
includes repeated and variant mentions so this behavior is visible.

**Q7. Most important errors for news monitoring:**
False negatives on ORG entities are most costly — a news monitor that misses
"Maybank" in an article would fail its core purpose. False negatives on EVENT
and MONEY also matter for monitoring announcements and financial figures. False
positives on PERSON are annoying but less harmful.

**Q8. Privacy, copyright, bias:**
Articles may mention individuals in sensitive contexts. The model may reflect
biases in the training corpus (e.g., over-detecting certain organizations). All
article content is copyrighted by original publishers. Raw articles are not
committed; only short evaluation excerpts and text-free annotation metadata are
included.

**Q9. How are duplicates and near-duplicates handled?**
Exact duplicates are removed by `original_id` and by normalised `title + body`.
Near-duplicates are clustered by title/body similarity and each cluster is kept
in one split, so syndicated copies cannot leak across train, validation, and
test. The audit and thresholds are recorded in `data/training_report.json`.

**Q10. Why use validation rather than the test set for tuning?**
Tuning on the test set would make the reported test metrics optimistic. This
project evaluates validation after every epoch, selects the best epoch and
configuration by validation exact-span F1, and only then evaluates the held-out
test set once. The test metrics are therefore an honest estimate for the chosen
configuration.
