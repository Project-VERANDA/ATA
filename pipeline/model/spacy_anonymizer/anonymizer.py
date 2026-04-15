import spacy
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Model Loading ---
# Construct the path to the spaCy model relative to this script's location.
script_dir = os.path.dirname(os.path.abspath(__file__))
# The spaCy training process saves the model directly in this directory.
SPACY_MODEL_PATH = os.path.join(script_dir, "model")
NLP = None

logger.info("--- spaCy Anonymizer Script Initializing ---")
try:
    logger.info(f"Attempting to load custom spaCy model from: {SPACY_MODEL_PATH}")
    if os.path.exists(SPACY_MODEL_PATH):
        NLP = spacy.load(SPACY_MODEL_PATH)
        logger.info("Custom spaCy model loaded successfully.")
    else:
        logger.warning(f"Model path does not exist: {SPACY_MODEL_PATH}")
except Exception as e:
    logger.error(f"Error loading custom spaCy model: {e}", exc_info=True)
    NLP = None

def is_spacy_model_available():
    """Checks if the custom spaCy model was loaded successfully."""
    return NLP is not None

def anonymize_text_with_spacy(text: str) -> tuple[str, list]:
    """
    Anonymizes a given text using the custom-trained spaCy model for NER.
    It replaces recognized entities with their labels (e.g., [NAME_PATIENT]).

    Args:
        text: The text to be anonymized.

    Returns:
        A tuple containing:
        - The anonymized text string.
        - A list of dictionaries, where each dictionary represents an entity.
    """
    if not NLP:
        raise RuntimeError("Custom spaCy model is not loaded. Cannot perform anonymization.")
    if not text:
        return "", []

    doc = NLP(text)
    logger.info(f"spaCy model processing text. Found {len(doc.ents)} total entities.")

    # Filter out speaker tags and sort entities by start position
    ents = sorted(
        [ent for ent in doc.ents if not ent.text.strip().startswith("SPEAKER_")],
        key=lambda e: e.start_char
    )
    
    # Create a structured list of the entities that were found
    entities_found = [
        {"start": ent.start_char, "end": ent.end_char, "label": ent.label_}
        for ent in ents
    ]

    logger.info(f"Found {len(ents)} non-speaker entities to anonymize.")
    if not ents:
        return text, []

    # Build the anonymized text piece by piece
    result_parts = []
    last_end = 0
    for ent in ents:
        logger.info(f"  -> Anonymizing: '{ent.text}' with label '[{ent.label_}]'")
        # Append the text from the end of the last entity to the start of this one
        result_parts.append(text[last_end:ent.start_char])
        # Append the replacement tag
        result_parts.append(f"[{ent.label_}]")
        # Update the position of the last character processed
        last_end = ent.end_char

    # Append any remaining text after the last entity
    result_parts.append(text[last_end:])

    return "".join(result_parts), entities_found

if __name__ == '__main__':
    # This block allows for direct testing of the anonymizer script.
    if NLP:
        logger.info("spaCy model loaded successfully. Running a test.")
        sample_text = "SPEAKER_00: My name is John Doe and I live in Berlin. SPEAKER_01: It's nice to meet you, John."
        anonymized_output, _ = anonymize_text_with_spacy(sample_text)
        
        print("\n--- Anonymization Test ---")
        print(f"Original Text:    {sample_text}")
        print(f"Anonymized Text:  {anonymized_output}")
        print("--------------------------")
    else:
        logger.error("Skipping test run because spaCy model could not be loaded.")
        logger.error(f"Please ensure a valid model exists at: {SPACY_MODEL_PATH}") 