"""
NER Pipeline - Named Entity Recognition using spaCy
Supports: en_core_web_sm (baseline) and fine-tuned model
"""

import os
import spacy
import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

MODEL_NAME = os.environ.get("NER_MODEL", "en_core_web_sm")
MAX_TEXT_LENGTH = int(os.environ.get("MAX_TEXT_LENGTH", 50000))

_nlp: Optional[spacy.language.Language] = None


def get_nlp() -> spacy.language.Language:
    """Lazy-load the spaCy model (singleton)."""
    global _nlp
    if _nlp is None:
        trained_path = os.path.join(os.path.dirname(__file__), "..", "data", "trained_model")
        if os.path.isdir(trained_path):
            logger.info(f"Loading trained model from {trained_path}")
            _nlp = spacy.load(trained_path)
        else:
            logger.info(f"Loading pretrained model: {MODEL_NAME}")
            _nlp = spacy.load(MODEL_NAME)
    return _nlp


def extract_entities(title: str, body: str) -> Dict[str, Any]:
    """
    Combine title + body and run NER.
    Returns: {entities: [{text, label, start, end}], meta: {entity_count, model, text_length}}
    """
    if not title or not title.strip():
        raise ValueError("title must not be empty")
    if not body or not body.strip():
        raise ValueError("body must not be empty")

    combined = f"{title.strip()}\n\n{body.strip()}"
    if len(combined) > MAX_TEXT_LENGTH:
        raise ValueError(
            f"Input exceeds maximum length of {MAX_TEXT_LENGTH} characters (got {len(combined)})."
        )

    nlp = get_nlp()
    doc = nlp(combined)

    entities = []
    seen = set()
    for ent in doc.ents:
        key = (ent.start_char, ent.end_char, ent.label_)
        if key not in seen:
            seen.add(key)
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
            })

    trained_path = os.path.join(os.path.dirname(__file__), "..", "data", "trained_model")
    model_used = "trained-model" if os.path.isdir(trained_path) else MODEL_NAME

    return {
        "entities": entities,
        "meta": {
            "entity_count": len(entities),
            "model": model_used,
            "text_length": len(combined),
        },
    }


def supported_labels() -> List[str]:
    """Return entity labels supported by the loaded model."""
    nlp = get_nlp()
    return nlp.pipe_labels.get("ner", [])
