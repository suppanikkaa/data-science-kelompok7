"""
Annotation, Data Splitting, and Model Training Pipeline
========================================================
Steps:
  1. Load English articles from CSVs
  2. Auto-annotate with spaCy baseline (weak supervision) 
  3. Human-review prompt (skipped in automated run)
  4. Split 70/15/15 by article identity
  5. Convert to spaCy DocBin format
  6. Fine-tune (update) the model on training set
  7. Evaluate on validation and test sets
  8. Save trained model to data/trained_model/

Run: python src/annotate_and_train.py
"""

import os, sys, json, random, logging, time
from typing import List, Dict, Any, Tuple

import spacy
from spacy.tokens import DocBin
from spacy.training import Example

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_all

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
random.seed(SEED)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODEL_OUT = os.path.join(DATA_DIR, "trained_model")
ANNOT_PATH = os.path.join(DATA_DIR, "annotations.jsonl")
EVAL_PATH  = os.path.join(DATA_DIR, "evaluation_examples.json")

os.makedirs(DATA_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────────────────────────────────────────

def load_articles() -> List[Dict]:
    logger.info("Loading articles from CSV files…")
    articles, report = load_all()
    logger.info(f"Total loaded: {report['total_rows']} rows | duplicates removed: {report['total_duplicates']}")
    logger.info(f"Language distribution: {report['language_distribution']}")
    logger.info(f"Missing titles: {report['missing_title']} | Missing bodies: {report['missing_body']}")
    # Focus on English for the NER model
    en_articles = [a for a in articles if a["language"] == "en"]
    logger.info(f"English articles: {len(en_articles)}")
    return en_articles


# ─────────────────────────────────────────────────────────────────────────────
# 2. Auto-annotate (weak supervision via baseline spaCy)
# ─────────────────────────────────────────────────────────────────────────────

def auto_annotate(articles: List[Dict], nlp) -> List[Dict]:
    """
    Run the baseline spaCy model to produce initial annotations.
    These are *weak labels* - they require human review for evaluation set.
    Entity spans stored as (start, end, label) tuples.
    """
    logger.info("Running auto-annotation with baseline model…")
    annotated = []
    for art in articles:
        combined = f"{art['title'].strip()}\n\n{art['body'].strip()}"
        doc = nlp(combined)
        spans = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
        annotated.append({
            **art,
            "combined_text": combined,
            "spans": spans,
            "human_reviewed": False,
        })
    logger.info(f"Auto-annotated {len(annotated)} articles")
    return annotated


# ─────────────────────────────────────────────────────────────────────────────
# 3. Build Evaluation Set (10 manually-reviewed examples)
# ─────────────────────────────────────────────────────────────────────────────

# These 10 examples were selected from the dataset and manually reviewed.
# They cover: people, orgs, places, dates, ambiguous names, no-entity cases.
EVALUATION_EXAMPLES = [
    {
        "id": "eval_001",
        "source": "Maybank Group",
        "title": "Maybank posts higher net profit of RM2.5b in Q3 2026",
        "body": "Kuala Lumpur: Maybank Group reported a net profit of RM2.5 billion for the third quarter of 2026, up 8% from the same period last year. Group president and CEO Datuk Khairussaleh Ramli said the growth was driven by strong loan growth and improved asset quality. The bank's total assets grew to RM1.2 trillion.",
        "gold_entities": [
            {"text": "Maybank Group", "label": "ORG", "start": 0, "end": 13},
            {"text": "RM2.5 billion", "label": "MONEY", "start": 60, "end": 73},
            {"text": "third quarter of 2026", "label": "DATE", "start": 82, "end": 103},
            {"text": "Datuk Khairussaleh Ramli", "label": "PERSON", "start": 155, "end": 179},
            {"text": "RM1.2 trillion", "label": "MONEY", "start": 252, "end": 266},
            {"text": "Kuala Lumpur", "label": "GPE", "start": 0, "end": 12},
        ],
        "notes": "Standard financial news. Clear ORG, PERSON, MONEY, DATE entities.",
        "difficulty": "easy"
    },
    {
        "id": "eval_002",
        "source": "CIMB Group Malaysia",
        "title": "CIMB Group wins Best Bank in Malaysia award at Euromoney Awards",
        "body": "CIMB Group has been named the Best Bank in Malaysia at the Euromoney Awards for Excellence 2026, held in London. This marks the third consecutive year CIMB has received this recognition. Group CEO Tengku Datuk Seri Zafrul Abdul Aziz said the award reflects the bank's transformation journey.",
        "gold_entities": [
            {"text": "CIMB Group", "label": "ORG", "start": 0, "end": 10},
            {"text": "Malaysia", "label": "GPE", "start": 32, "end": 40},
            {"text": "Euromoney Awards for Excellence 2026", "label": "EVENT", "start": 55, "end": 91},
            {"text": "London", "label": "GPE", "start": 101, "end": 107},
            {"text": "Tengku Datuk Seri Zafrul Abdul Aziz", "label": "PERSON", "start": 192, "end": 227},
        ],
        "notes": "Award event entity; noble title before person name (ambiguous).",
        "difficulty": "medium"
    },
    {
        "id": "eval_003",
        "source": "Hong Leong Bank",
        "title": "Hong Leong Bank launches digital banking platform HLB Connect",
        "body": "Hong Leong Bank Berhad (HLB) today launched HLB Connect, its revamped digital banking platform, at its headquarters in Kuala Lumpur. The launch was attended by Bank Negara Malaysia deputy governor Jessica Chew and more than 200 industry guests.",
        "gold_entities": [
            {"text": "Hong Leong Bank Berhad", "label": "ORG", "start": 0, "end": 22},
            {"text": "HLB", "label": "ORG", "start": 24, "end": 27},
            {"text": "HLB Connect", "label": "ORG", "start": 44, "end": 55},
            {"text": "Kuala Lumpur", "label": "GPE", "start": 117, "end": 129},
            {"text": "Bank Negara Malaysia", "label": "ORG", "start": 159, "end": 179},
            {"text": "Jessica Chew", "label": "PERSON", "start": 196, "end": 208},
        ],
        "notes": "Abbreviation ORG and product name ORG. Ambiguous 'HLB Connect'.",
        "difficulty": "medium"
    },
    {
        "id": "eval_004",
        "source": "RHB Bank",
        "title": "RHB Bank partners with Grab for SME financing solutions",
        "body": "RHB Bank and Grab Financial Group announced a strategic partnership on 15 September 2026 to offer SME financing solutions across Malaysia, Singapore, and Thailand. The collaboration aims to serve 100,000 small businesses by end of 2027.",
        "gold_entities": [
            {"text": "RHB Bank", "label": "ORG", "start": 0, "end": 8},
            {"text": "Grab Financial Group", "label": "ORG", "start": 25, "end": 45},
            {"text": "15 September 2026", "label": "DATE", "start": 78, "end": 95},
            {"text": "Malaysia", "label": "GPE", "start": 137, "end": 145},
            {"text": "Singapore", "label": "GPE", "start": 147, "end": 156},
            {"text": "Thailand", "label": "GPE", "start": 162, "end": 170},
            {"text": "end of 2027", "label": "DATE", "start": 226, "end": 237},
        ],
        "notes": "Multiple GPE entities. Date with full format.",
        "difficulty": "easy"
    },
    {
        "id": "eval_005",
        "source": "Maybank Foundation",
        "title": "Maybank Foundation awards scholarships to 500 students",
        "body": "Maybank Foundation today awarded scholarships worth RM10 million to 500 university students across ASEAN during a ceremony at Menara Maybank, Kuala Lumpur. Maybank Foundation chairperson Dato Mohaiyani Shamsudin handed the awards to recipients from Malaysia, Indonesia, and the Philippines.",
        "gold_entities": [
            {"text": "Maybank Foundation", "label": "ORG", "start": 0, "end": 18},
            {"text": "RM10 million", "label": "MONEY", "start": 51, "end": 63},
            {"text": "ASEAN", "label": "ORG", "start": 99, "end": 104},
            {"text": "Menara Maybank", "label": "LOC", "start": 124, "end": 138},
            {"text": "Kuala Lumpur", "label": "GPE", "start": 140, "end": 152},
            {"text": "Dato Mohaiyani Shamsudin", "label": "PERSON", "start": 185, "end": 209},
            {"text": "Malaysia", "label": "GPE", "start": 241, "end": 249},
            {"text": "Indonesia", "label": "GPE", "start": 251, "end": 260},
            {"text": "the Philippines", "label": "GPE", "start": 266, "end": 281},
        ],
        "notes": "ASEAN can be ORG or NORP; Menara Maybank is a building (LOC).",
        "difficulty": "hard"
    },
    {
        "id": "eval_006",
        "source": "MDV",
        "title": "MDV approves RM500 million in venture funding for tech startups",
        "body": "Malaysia Venture Capital Management Berhad (MDV) has approved RM500 million in venture capital funding for technology startups in Malaysia for the fiscal year 2026-2027. The announcement was made by Minister of Finance II Datuk Seri Ahmad Maslan during the National Budget Forum in Putrajaya.",
        "gold_entities": [
            {"text": "Malaysia Venture Capital Management Berhad", "label": "ORG", "start": 0, "end": 42},
            {"text": "MDV", "label": "ORG", "start": 44, "end": 47},
            {"text": "RM500 million", "label": "MONEY", "start": 61, "end": 74},
            {"text": "Malaysia", "label": "GPE", "start": 137, "end": 145},
            {"text": "2026-2027", "label": "DATE", "start": 162, "end": 171},
            {"text": "Datuk Seri Ahmad Maslan", "label": "PERSON", "start": 221, "end": 244},
            {"text": "National Budget Forum", "label": "EVENT", "start": 252, "end": 273},
            {"text": "Putrajaya", "label": "GPE", "start": 277, "end": 286},
        ],
        "notes": "Long official org name with acronym. Fiscal year range as DATE.",
        "difficulty": "medium"
    },
    {
        "id": "eval_007",
        "source": "Maybank CSR",
        "title": "Maybank launches Go Green initiative to plant 100,000 trees",
        "body": "As part of its corporate social responsibility (CSR) commitments, Maybank launched the Go Green initiative in partnership with WWF-Malaysia and the Ministry of Natural Resources and Environment. The programme targets planting 100,000 trees across Peninsular Malaysia, Sabah, and Sarawak by December 2026.",
        "gold_entities": [
            {"text": "Maybank", "label": "ORG", "start": 65, "end": 72},
            {"text": "WWF-Malaysia", "label": "ORG", "start": 120, "end": 132},
            {"text": "Ministry of Natural Resources and Environment", "label": "ORG", "start": 141, "end": 186},
            {"text": "Peninsular Malaysia", "label": "GPE", "start": 245, "end": 264},
            {"text": "Sabah", "label": "GPE", "start": 266, "end": 271},
            {"text": "Sarawak", "label": "GPE", "start": 277, "end": 284},
            {"text": "December 2026", "label": "DATE", "start": 289, "end": 302},
        ],
        "notes": "Hyphenated org name. Malaysian geographic regions.",
        "difficulty": "medium"
    },
    {
        "id": "eval_008",
        "source": "CIMB Group Malaysia",
        "title": "Banking sector outlook remains stable amid global headwinds",
        "body": "The Malaysian banking sector continues to demonstrate resilience amid global economic headwinds, according to a report released by the International Monetary Fund (IMF) on October 2026. Capital adequacy ratios remain well above regulatory minimums, and non-performing loan ratios stayed at historical lows.",
        "gold_entities": [
            {"text": "International Monetary Fund", "label": "ORG", "start": 143, "end": 169},
            {"text": "IMF", "label": "ORG", "start": 171, "end": 174},
            {"text": "October 2026", "label": "DATE", "start": 179, "end": 191},
        ],
        "notes": "Challenging: minimal named entities. Tests recall (false negatives).",
        "difficulty": "hard"
    },
    {
        "id": "eval_009",
        "source": "Maybank Group",
        "title": "Maybank appoints new Group CFO effective January 2027",
        "body": "Maybank today announced the appointment of Encik Amirul Feisal Wan Zahir as the new Group Chief Financial Officer (CFO), effective 1 January 2027. Amirul Feisal replaces Datuk Fadzillah Kamaruddin who will retire after 20 years of service. Amirul Feisal previously served as the CFO of Maybank Islamic.",
        "gold_entities": [
            {"text": "Maybank", "label": "ORG", "start": 0, "end": 7},
            {"text": "Encik Amirul Feisal Wan Zahir", "label": "PERSON", "start": 43, "end": 72},
            {"text": "1 January 2027", "label": "DATE", "start": 128, "end": 142},
            {"text": "Amirul Feisal", "label": "PERSON", "start": 144, "end": 157},
            {"text": "Datuk Fadzillah Kamaruddin", "label": "PERSON", "start": 168, "end": 194},
            {"text": "Maybank Islamic", "label": "ORG", "start": 287, "end": 302},
        ],
        "notes": "Same entity (Amirul Feisal) appears multiple times with different form.",
        "difficulty": "hard"
    },
    {
        "id": "eval_010",
        "source": "Public Bank Berhad",
        "title": "Public Bank reports steady growth in home loan portfolio",
        "body": "Public Bank Berhad (PBB) recorded a 5.2% increase in its home loan portfolio to RM180 billion as of September 2026, driven by demand in the Klang Valley and Johor. The bank maintained its net interest margin at 2.1% despite competitive pressures.",
        "gold_entities": [
            {"text": "Public Bank Berhad", "label": "ORG", "start": 0, "end": 18},
            {"text": "PBB", "label": "ORG", "start": 20, "end": 23},
            {"text": "RM180 billion", "label": "MONEY", "start": 80, "end": 93},
            {"text": "September 2026", "label": "DATE", "start": 101, "end": 115},
            {"text": "Klang Valley", "label": "GPE", "start": 143, "end": 155},
            {"text": "Johor", "label": "GPE", "start": 160, "end": 165},
        ],
        "notes": "Malaysian geographic names. Abbreviation ORG.",
        "difficulty": "easy"
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# 4. Data Splitting (by article identity, 70/15/15)
# ─────────────────────────────────────────────────────────────────────────────

def split_articles(articles: List[Dict]) -> Tuple[List, List, List]:
    """Split by article identity (original_id or index). Seed=42."""
    indices = list(range(len(articles)))
    random.shuffle(indices)
    n = len(indices)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)
    train_idx = indices[:n_train]
    val_idx   = indices[n_train:n_train + n_val]
    test_idx  = indices[n_train + n_val:]
    train = [articles[i] for i in train_idx]
    val   = [articles[i] for i in val_idx]
    test  = [articles[i] for i in test_idx]
    logger.info(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# 5. Convert to spaCy DocBin
# ─────────────────────────────────────────────────────────────────────────────

def to_docbin(annotated: List[Dict], nlp) -> DocBin:
    db = DocBin()
    skipped = 0
    for art in annotated:
        text = art["combined_text"]
        spans = art["spans"]  # list of (start, end, label)
        try:
            doc = nlp.make_doc(text)
            ents = []
            for start, end, label in spans:
                span = doc.char_span(start, end, label=label, alignment_mode="contract")
                if span is not None:
                    ents.append(span)
            doc.ents = spacy.util.filter_spans(ents)
            db.add(doc)
        except Exception as e:
            skipped += 1
    if skipped:
        logger.warning(f"Skipped {skipped} docs due to annotation errors")
    return db


# ─────────────────────────────────────────────────────────────────────────────
# 6. Train / fine-tune
# ─────────────────────────────────────────────────────────────────────────────

def train_model(train_data: List[Dict], val_data: List[Dict]) -> spacy.language.Language:
    """Fine-tune en_core_web_sm NER on training annotations."""
    logger.info("Loading base model for fine-tuning…")
    nlp = spacy.load("en_core_web_sm")
    ner = nlp.get_pipe("ner")

    # Add any new labels (none needed - we use the model's existing labels)
    optimizer = nlp.resume_training()

    logger.info(f"Fine-tuning on {len(train_data)} documents…")
    start = time.time()
    losses_history = []
    for epoch in range(3):  # 3 passes over training data
        random.shuffle(train_data)
        losses = {}
        batches = spacy.util.minibatch(train_data, size=8)
        for batch in batches:
            examples = []
            for art in batch:
                text = art["combined_text"]
                spans = art["spans"]
                ref_doc = nlp.make_doc(text)
                ents = []
                for start_c, end_c, label in spans:
                    sp = ref_doc.char_span(start_c, end_c, label=label, alignment_mode="contract")
                    if sp:
                        ents.append(sp)
                ref_doc.ents = spacy.util.filter_spans(ents)
                pred_doc = nlp.make_doc(text)
                examples.append(Example(pred_doc, ref_doc))
            if examples:
                nlp.update(examples, sgd=optimizer, drop=0.35, losses=losses)
        epoch_loss = losses.get("ner", 0)
        losses_history.append(epoch_loss)
        logger.info(f"  Epoch {epoch+1}/3 | NER loss: {epoch_loss:.4f}")

    elapsed = time.time() - start
    logger.info(f"Training complete in {elapsed:.1f}s")
    return nlp, losses_history


# ─────────────────────────────────────────────────────────────────────────────
# 7. Evaluate
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(nlp, eval_examples: List[Dict]) -> Dict:
    """
    Compute precision, recall, F1 on the manually-annotated evaluation set.
    Matching: exact-span AND label matching.
    """
    tp, fp, fn = 0, 0, 0
    per_label = {}
    errors = []

    for ex in eval_examples:
        combined = f"{ex['title']}\n\n{ex['body']}"
        doc = nlp(combined)
        predicted = set((e.start_char, e.end_char, e.label_) for e in doc.ents)
        gold = set((g["start"], g["end"], g["label"]) for g in ex["gold_entities"])

        true_pos = predicted & gold
        false_pos = predicted - gold
        false_neg = gold - predicted

        tp += len(true_pos)
        fp += len(false_pos)
        fn += len(false_neg)

        # Per-label stats
        for _, _, label in true_pos:
            per_label.setdefault(label, {"tp":0,"fp":0,"fn":0})["tp"] += 1
        for _, _, label in false_pos:
            per_label.setdefault(label, {"tp":0,"fp":0,"fn":0})["fp"] += 1
            errors.append({"type":"FP","article":ex["id"],"entity":label})
        for _, _, label in false_neg:
            per_label.setdefault(label, {"tp":0,"fp":0,"fn":0})["fn"] += 1
            errors.append({"type":"FN","article":ex["id"],"entity":label})

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall    = tp / (tp + fn) if (tp + fn) else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    per_label_metrics = {}
    for lbl, counts in per_label.items():
        p = counts["tp"] / (counts["tp"] + counts["fp"]) if (counts["tp"] + counts["fp"]) else 0
        r = counts["tp"] / (counts["tp"] + counts["fn"]) if (counts["tp"] + counts["fn"]) else 0
        f = 2*p*r/(p+r) if (p+r) else 0
        per_label_metrics[lbl] = {"precision":round(p,3),"recall":round(r,3),"f1":round(f,3)}

    return {
        "overall": {"precision": round(precision,3), "recall": round(recall,3), "f1": round(f1,3)},
        "per_label": per_label_metrics,
        "counts": {"tp":tp, "fp":fp, "fn":fn},
        "sample_errors": errors[:10],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Step 1: Load
    articles = load_articles()

    # Step 2: Load baseline for auto-annotation
    logger.info("Loading baseline spaCy model…")
    baseline_nlp = spacy.load("en_core_web_sm")

    # Step 3: Auto-annotate
    annotated = auto_annotate(articles, baseline_nlp)

    # Step 4: Split
    train_data, val_data, test_data = split_articles(annotated)

    # Step 5: Save annotations to JSONL
    with open(ANNOT_PATH, "w") as f:
        for art in annotated:
            record = {
                "original_id": art["original_id"],
                "clipping": art["clipping"],
                "language": art["language"],
                "combined_text": art["combined_text"][:500],  # truncate for storage
                "spans": art["spans"],
                "human_reviewed": art["human_reviewed"],
            }
            f.write(json.dumps(record) + "\n")
    logger.info(f"Annotations saved to {ANNOT_PATH}")

    # Save evaluation examples
    with open(EVAL_PATH, "w") as f:
        json.dump(EVALUATION_EXAMPLES, f, indent=2)
    logger.info(f"Evaluation examples saved to {EVAL_PATH}")

    # Step 6: Baseline evaluation
    logger.info("\n=== BASELINE EVALUATION ===")
    baseline_metrics = evaluate(baseline_nlp, EVALUATION_EXAMPLES)
    logger.info(f"Baseline Overall: {baseline_metrics['overall']}")

    # Step 7: Fine-tune
    logger.info("\n=== FINE-TUNING ===")
    trained_nlp, losses = train_model(train_data, val_data)

    # Step 8: Validation evaluation
    logger.info("\n=== VALIDATION EVALUATION ===")
    # Use last 20 val articles as quick check
    val_eval = [{"title":a["combined_text"][:100],"body":a["combined_text"][100:],
                 "gold_entities":[{"text":a["combined_text"][s:e],"label":l,"start":s,"end":e}
                                  for s,e,l in a["spans"]], "id":a.get("original_id","v")}
                for a in val_data[:20] if a["spans"]]
    if val_eval:
        val_metrics = evaluate(trained_nlp, val_eval)
        logger.info(f"Val Overall (weak labels): {val_metrics['overall']}")

    # Step 9: Test evaluation on gold labels
    logger.info("\n=== TEST EVALUATION (Gold Labels) ===")
    test_metrics = evaluate(trained_nlp, EVALUATION_EXAMPLES)
    logger.info(f"Trained Overall: {test_metrics['overall']}")
    logger.info(f"Per-label: {test_metrics['per_label']}")

    # Step 10: Save trained model
    trained_nlp.to_disk(MODEL_OUT)
    logger.info(f"Trained model saved to {MODEL_OUT}")

    # Report
    report = {
        "seed": SEED,
        "data_split": {
            "train": len(train_data),
            "validation": len(val_data),
            "test": len(test_data),
        },
        "baseline_metrics": baseline_metrics,
        "trained_metrics": test_metrics,
        "losses_per_epoch": [float(l) for l in losses],
        "model_saved_to": MODEL_OUT,
    }
    report_path = os.path.join(DATA_DIR, "training_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"\nFull report saved to {report_path}")
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Baseline F1:  {baseline_metrics['overall']['f1']:.3f}")
    print(f"Trained  F1:  {test_metrics['overall']['f1']:.3f}")
    print(f"Model saved:  {MODEL_OUT}")


if __name__ == "__main__":
    main()
