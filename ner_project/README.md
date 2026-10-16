# Article Named Entity Recognition (NER) Service
## Maybank Malaysia Online Media – Study Case

A web service that extracts named entities (people, organizations, locations, dates, etc.)
from news articles using a spaCy-based NER pipeline fine-tuned on the provided dataset.

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

### Dataset
8 CSV files containing Maybank Malaysia online media articles (Sep–Oct 2026):

| Clipping Name      | File                          | Rows |
|--------------------|-------------------------------|------|
| Hong Leong Bank    | ...9xtqAsPitG.csv             |   96 |
| CIMB Group Malaysia| ...bRQ9afwSIP.csv             |  347 |
| Public Bank Berhad | ...4sxH8Dz0hr.csv             |    2 |
| RHB Bank           | ...58yzyiBQ9U.csv             |  105 |
| MDV                | ...WdjZl0bRfZ.csv             |   18 |
| Maybank CSR        | ...i3BJEnT77K.csv             |  160 |
| Maybank Foundation | ...-nwub4cROX.csv             |   29 |
| Maybank Group      | ...hMxhx9QLTq.csv             |  534 |

**Language distribution:** en (1059), ms (177), zh (37), id (15), vi/es (2)

---

## Setup & Installation

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### 1. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate.bat      # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. (Optional) Run training pipeline
```bash
cd src
python annotate_and_train.py
```
This will:
- Load and profile all 8 CSV files
- Auto-annotate using baseline spaCy (weak supervision)
- Split 70/15/15 by article identity (seed=42)
- Fine-tune the NER model on training data
- Evaluate on gold-labeled evaluation set
- Save the trained model to `data/trained_model/`

### 4. Start the service
```bash
cd src
python app.py
```
The service runs at: **http://localhost:5000**

---

## API Usage

### POST /extract-entities

**Request:**
```json
{
  "title": "Maybank posts record profit in Q3 2026",
  "body": "Maybank Group, headquartered in Kuala Lumpur, reported a net profit of RM2.5 billion for the third quarter of 2026."
}
```

**Response (200 OK):**
```json
{
  "entities": [
    {"text": "Maybank Group", "label": "ORG", "start": 0, "end": 13},
    {"text": "Kuala Lumpur", "label": "GPE", "start": 32, "end": 44},
    {"text": "RM2.5 billion", "label": "MONEY", "start": 60, "end": 73},
    {"text": "third quarter of 2026", "label": "DATE", "start": 82, "end": 103}
  ],
  "meta": {
    "entity_count": 4,
    "model": "en_core_web_sm",
    "text_length": 116
  }
}
```

**Error (400 Bad Request):**
```json
{"error": "title must not be empty"}
```

### GET /health
```json
{"status": "ok", "model": "en_core_web_sm", "supported_labels": ["CARDINAL", "DATE", "EVENT", ...]}
```

---

## Supported Entity Labels

| Label   | Description                     | Example                    |
|---------|---------------------------------|----------------------------|
| PERSON  | Person names                    | Datuk Khairussaleh Ramli   |
| ORG     | Organizations / companies       | Maybank Group, CIMB        |
| GPE     | Geopolitical entities           | Malaysia, Kuala Lumpur     |
| LOC     | Non-GPE locations               | Menara Maybank             |
| DATE    | Dates / time periods            | Q3 2026, 15 September 2026 |
| MONEY   | Monetary values                 | RM2.5 billion              |
| EVENT   | Named events / awards           | Euromoney Awards 2026      |
| NORP    | Nationalities / political groups| Malaysian, ASEAN           |

---

## Running Tests

```bash
cd src
python -m pytest ../tests/test_ner_service.py -v
```

20 test cases covering:
- Normal input (English, Malay, punctuation)
- Empty title/body validation
- Max length enforcement
- Malformed JSON
- Repeated entities
- Deterministic output
- API endpoint responses (health, extract, HTML panel)

---

## Model Details

| Setting              | Value                   |
|----------------------|-------------------------|
| Base model           | en_core_web_sm (spaCy)  |
| Fine-tuning epochs   | 3                       |
| Dropout              | 0.35                    |
| Batch size           | 8                       |
| Random seed          | 42                      |
| Data split           | 70% / 15% / 15%         |
| Split unit           | Article (not sentence)  |
| Annotation format    | JSONL (span-level)      |
| Matching criterion   | Exact-span + label      |

### Model Choice Rationale
`en_core_web_sm` was selected because:
- The majority of articles (>83%) are in English
- It supports all required entity types (PERSON, ORG, GPE, LOC, DATE, MONEY, EVENT)
- Small and fast for a prototype service
- Supports fine-tuning without GPU requirements

For production, `en_core_web_trf` (transformer-based) would yield higher accuracy.

---

## Known Limitations

1. **English-only**: The NER model works best on English text; Malay and Chinese articles will produce poor results
2. **Malaysian entities**: Malaysian proper nouns (Datuk Seri, Dewan Rakyat, Bursa Malaysia) may be misclassified
3. **Abbreviations**: Acronyms like HLB, MDV, PBB may not be detected as ORG
4. **Honorifics**: Malaysian honorifics (Dato', Tan Sri, Encik) are sometimes merged with person names
5. **Weak supervision**: Training uses auto-annotation (not fully human-reviewed); this introduces label noise

---

## Responsible Data Use

- Articles are used only for NER training/demo; no redistribution
- No personal data is stored (stateless service)
- Biases: the dataset focuses on Malaysian banking; model may underperform on other domains
- Copyright: articles belong to their respective publishers; do not commit raw article text

---

## Questions Answered (Study Case)

**Q1. What makes a text span a named entity?**
A named entity is a real-world object referred to by name: a person, place, organization, date, or similar proper noun. "Maybank" is an ORG; "bank" alone is not.

**Q2. Why combine title and body?**
The title often contains the article's main entity (e.g., "Maybank posts profit"). Processing independently would lose context — body references may be pronouns or abbreviated forms that require the title for resolution.

**Q3. Label mapping to user-facing categories:**
PERSON→People, ORG→Organizations, GPE+LOC→Places, DATE→Dates, MONEY→Amounts, EVENT→Events

**Q4. Tokenization / language effects:**
spaCy tokenizes on whitespace and punctuation. Malay text (e.g., "RM500") may tokenize differently. Capitalization guides NER — lowercased text loses person/org signals. Chinese text requires a different model.

**Q5. Text vs label vs offset:**
`text` is the surface string ("Maybank"), `label` is the category (ORG), `start`/`end` are zero-based character indices in the combined string.

**Q6. Same entity, different spellings:**
"CIMB", "CIMB Group", "CIMB Bank" are treated as separate entities. Normalization (entity linking) would be needed to merge them.

**Q7. Most important errors for news monitoring:**
False negatives on ORG entities are most costly — a news monitor that misses "Maybank" in an article would fail its core purpose. False positives on PERSON are annoying but less harmful.

**Q8. Privacy, copyright, bias:**
Articles may mention individuals in sensitive contexts. The model may reflect biases in the training corpus (e.g., over-detecting certain organizations). All article content is copyrighted by original publishers.
