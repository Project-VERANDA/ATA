# spaCy Anonymizer

This directory contains the spaCy-based anonymization component of the application. It includes the trained model, the Python script for anonymization, and all necessary files to reproduce the model training.

## Anonymization Logic

The core anonymization logic is in `anonymizer.py`. This script:
1.  Loads the custom-trained NER model from the `model/` directory.
2.  Provides the `anonymize_text_with_spacy()` function, which is called by the main web application.
3.  Takes raw text, identifies named entities (like names, locations, etc.), and replaces them with corresponding tags (e.g., `[NAME_PATIENT]`).
4.  Includes a safeguard to prevent speaker tags (e.g., `SPEAKER_00`) from being incorrectly anonymized.

## Model Training

The model used by this anonymizer was trained on a custom dataset for recognizing personally identifiable information in German medical transcripts. The training process is fully reproducible.

To retrain the model, run the `finetune.py` script from the `training` directory.

1.  **Navigate to the training directory:**
    ```bash
    cd spacy_anonymizer/training
    ```

2.  **Run the Training Script:**
    Execute the `finetune.py` script. This will use the data in the `data/` directory, run the spaCy training process, and save the best-performing model directly to the `spacy_anonymizer/model/` directory, where the application will automatically load it.
    ```bash
    python finetune.py
    ```

You can customize the training by passing arguments. For example, use `--use_gpu 0` to train on a GPU. Use `python finetune.py --help` to see all available options. After the script finishes, the new model is immediately ready for use by the application. 