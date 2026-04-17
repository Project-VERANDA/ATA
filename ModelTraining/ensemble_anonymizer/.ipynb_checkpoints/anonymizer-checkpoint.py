import re
from bert_anonymizer.anonymizer import anonymize_text_with_bert
from spacy_anonymizer.anonymizer import anonymize_text_with_spacy

def filter_overlapping_entities(entities):
    """
    Filters a list of entities to remove overlaps.
    If two entities overlap, the one that is longer is kept.
    """
    # Sort by start position, and then by end position in descending order
    # to prioritize longer spans if start positions are the same.
    sorted_entities = sorted(entities, key=lambda x: (x['start'], -x['end']))
    
    if not sorted_entities:
        return []

    non_overlapping = []
    # Start with the first entity (which is the earliest and longest at its position)
    last_entity = sorted_entities[0]

    for current_entity in sorted_entities[1:]:
        # If the current entity starts after the last one ends, there's no overlap.
        if current_entity['start'] >= last_entity['end']:
            non_overlapping.append(last_entity)
            last_entity = current_entity
        # If they overlap, we've already chosen the longer one due to the sorting,
        # so we just ignore the shorter, contained entity.
        else:
            # This 'else' implicitly handles the case where the current_entity is either
            # fully contained within the last_entity or starts at the same position
            # but is shorter. In these cases, we do nothing and keep last_entity.
            pass

    non_overlapping.append(last_entity)
    return non_overlapping

def anonymize_text_with_ensemble(original_text: str) -> str:
    """
    Anonymizes text using a parallel ensemble of spaCy and BERT models.
    It runs both models on the original text and merges their findings.
    """
    # Pass 1: Run both models in parallel on the original text
    _, spacy_entities = anonymize_text_with_spacy(original_text)
    _, bert_entities = anonymize_text_with_bert(original_text)

    # Combine the lists of entities from both models
    combined_entities = spacy_entities + bert_entities
    
    # Remove duplicates and resolve overlaps
    # This function will keep the longest entity in case of an overlap
    final_entities = filter_overlapping_entities(combined_entities)
    
    # Sort the final list by start position in reverse order for safe replacement
    final_entities.sort(key=lambda x: x['start'], reverse=True)

    # Reconstruct the text from the original using the final merged entity list
    anonymized_text = original_text
    for entity in final_entities:
        start = entity['start']
        end = entity['end']
        tag = f"[{entity['label']}]"
        anonymized_text = anonymized_text[:start] + tag + anonymized_text[end:]
        
    return anonymized_text 