import os
import re
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# --- Dynamic Path Configuration ---
current_script_dir = Path(__file__).resolve().parent
model_training_dir = current_script_dir.parent
CUSTOM_MODEL_PATH = model_training_dir / "bert_model" / "finetuned_model" / "best-model"

print(f"[DEBUG] Resolved Model Path: {CUSTOM_MODEL_PATH}")
print(f"[DEBUG] Path Exists? {CUSTOM_MODEL_PATH.exists()}")

_cached_nlp = None

def get_nlp_pipeline():
    """Lazy load the BERT model."""
    global _cached_nlp
    if _cached_nlp is not None:
        return _cached_nlp

    if not CUSTOM_MODEL_PATH.exists():
        raise FileNotFoundError(f"Custom model not found at {CUSTOM_MODEL_PATH}")

    try:
        print(f"Loading custom BERT model from: {CUSTOM_MODEL_PATH}")
        tokenizer = AutoTokenizer.from_pretrained(str(CUSTOM_MODEL_PATH))
        model = AutoModelForTokenClassification.from_pretrained(str(CUSTOM_MODEL_PATH))
        _cached_nlp = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
        print("Custom BERT model loaded successfully.")
        return _cached_nlp
    except Exception as e:
        print(f"Error loading custom BERT model: {e}")
        raise e

def filter_overlapping_entities(entities):
    """Filters overlapping entities, keeping the longest."""
    if not entities:
        return []
    sorted_entities = sorted(entities, key=lambda x: (x['start'], -(x['end'] - x['start'])))
    non_overlapping = []
    last_end = -1
    for entity in sorted_entities:
        if entity['start'] >= last_end:
            non_overlapping.append(entity)
            last_end = entity['end']
    return non_overlapping

def anonymize_text_with_bert(text: str):
    """Anonymizes text using the custom fine-tuned BERT model."""
    try:
        nlp = get_nlp_pipeline()
        results = nlp(text)
        entities = []
        for res in results:
            entities.append({
                'start': res['start'],
                'end': res['end'],
                'label': res['entity_group']
            })
        final_entities = filter_overlapping_entities(entities)
        final_entities.sort(key=lambda x: x['start'], reverse=True)
        anonymized_text = text
        for entity in final_entities:
            tag = f"[{entity['label']}]"
            start = max(0, entity['start'])
            end = min(len(anonymized_text), entity['end'])
            anonymized_text = anonymized_text[:start] + tag + anonymized_text[end:]
        return anonymized_text, final_entities
    except Exception as e:
        print(f"Error during BERT anonymization: {e}")
        return text, []

def anonymize_text_with_spacy(text: str):
    """Placeholder: Not implemented yet."""
    raise NotImplementedError("spaCy model not yet generated.")

def anonymize_text_with_ensemble(text: str):
    """Placeholder: Requires spaCy."""
    raise NotImplementedError("Ensemble requires both BERT and spaCy.")