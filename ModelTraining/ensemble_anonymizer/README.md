# Ensemble Anonymizer

This directory contains the ensemble-based anonymization component of the application. It combines the predictions from both the BERT and spaCy models to achieve a more robust and accurate anonymization result.

## Anonymization Logic

The core anonymization logic is in `anonymizer.py`. This script:
1.  Imports the anonymization functions from both `bert_anonymizer` and `spacy_anonymizer`.
2.  Provides the `anonymize_text_with_ensemble()` function, which is called by the main web application.
3.  Runs both models on the input text to get two sets of anonymized entities.
4.  Merges the results, giving priority to the BERT model's predictions in case of overlapping entities. This leverages BERT's higher accuracy for complex entities while still benefiting from spaCy's speed and efficiency.
5.  Returns a single, consolidated anonymized text.

## How It Works

The ensemble method is designed to maximize recall, ensuring that as many PII entities as possible are caught. By combining a transformer-based model (BERT) with a traditional CNN-based model (spaCy), it covers a wider range of linguistic patterns.

This approach does not require its own model or training process. It is a post-processing step that intelligently combines the outputs of the other two models. Therefore, to improve the performance of the ensemble, you should focus on re-training or fine-tuning the individual BERT and spaCy models. 