import os
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import logging

# Configure logging
logger = logging.getLogger(__name__)

# --- Model Loading ---
# Construct the path to the BERT model relative to this script's location.
script_dir = os.path.dirname(os.path.abspath(__file__))
BERT_MODEL_PATH = os.path.join(script_dir, "bert_model")

nlp = None

try:
    if os.path.exists(BERT_MODEL_PATH):
        # Ensure the model can be loaded
        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PATH)
        model = AutoModelForTokenClassification.from_pretrained(BERT_MODEL_PATH)
        
        # Use a pipeline for easier NER; 'simple' aggregation groups word pieces
        nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
        logger.info("BERT model loaded successfully from local path.")
    else:
        logger.warning(f"BERT model not found at the expected path: {BERT_MODEL_PATH}")
        nlp = None
except Exception as e:
    logger.error(f"Error loading BERT model: {e}")
    logger.error("This can happen if the 'bert_model' directory is missing, incomplete, or if the 'config.json' is misconfigured.")
    nlp = None

def anonymize_text_with_bert(text: str) -> tuple[str, list]:
    """
    Anonymizes text using the pre-trained local BERT model.
    Replaces identified entities with a placeholder like [PERSON], [LOCATION], etc.

    Returns:
        A tuple containing:
        - The anonymized text string.
        - A list of dictionaries, where each dictionary represents an entity.
    """
    if not nlp:
        logger.warning("BERT model is not available. Cannot anonymize text.")
        return "BERT model is not available. Cannot anonymize text.", []

    try:
        logger.info("BERT model processing text...")
        # Perform Named Entity Recognition
        ner_results = nlp(text)
        logger.info(f"Found {len(ner_results)} total entities.")

        # Create a structured list of the entities that were found
        entities_found = [
            {"start": entity['start'], "end": entity['end'], "label": entity['entity_group'].upper()}
            for entity in ner_results
        ]

        if not ner_results:
            return text, []

        # Sort entities by start index in reverse order for safe replacement
        # in the text string.
        reversed_entities = sorted(ner_results, key=lambda x: x['start'], reverse=True)
        
        anonymized_text = text
        for entity in reversed_entities:
            start = entity['start']
            end = entity['end']
            original_word = text[start:end]
            
            entity_label = entity['entity_group'].upper()
            tag = f"[{entity_label}]"
            
            logger.info(f"  -> Anonymizing: '{original_word}' with label '{tag}'")
            anonymized_text = anonymized_text[:start] + tag + anonymized_text[end:]

        return anonymized_text, entities_found

    except Exception as e:
        logger.error(f"An error occurred during BERT anonymization: {e}")
        return "An error occurred during anonymization. Please check the logs.", [] 