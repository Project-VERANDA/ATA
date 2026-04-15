# BERT Anonymizer

This directory contains the BERT-based anonymization component of the application. It includes the trained model, the Python script for anonymization, and all necessary files to reproduce the model training.

## Anonymization Logic

The core anonymization logic is in `anonymizer.py`. This script:
1.  Loads the custom-trained NER model from the `bert_model/` directory.
2.  Provides the `anonymize_text_with_bert()` function, which is called by the main web application.
3.  Takes raw text, identifies named entities based on the fine-tuned BERT model, and replaces them with corresponding tags (e.g., `[NAME_PATIENT]`).

## Model Training

The model used by this anonymizer is a fine-tuned version of `bert-base-cased`, trained on a custom dataset for recognizing personally identifiable information in German medical transcripts. The training process is fully reproducible.

To retrain the model, navigate to the `training` directory and run the fine-tuning script.

1.  **Navigate to the training directory:**
    ```bash
    cd bert_anonymizer/training
    ```

2.  **Run the Fine-Tuning Script:**
    Execute the `finetune.py` script. This will use the data in the `data/` directory and save the newly trained model to a `bert_model` directory inside `training`.
    ```bash
    python finetune.py
    ```
    You can use `python finetune.py --help` to see all available options for customizing paths, models, and training parameters.

3.  **Replace the Old Model:**
    After training completes, you must copy the new model artifacts into the `bert_anonymizer/bert_model/` directory to be used by the application.
    ```bash
    # (From the 'training' directory)
    # First, remove the old model files
    rm -rf ../bert_model/*

    # Copy the new model files
    cp -r bert_model/* ../bert_model/
    ```

After these steps, the web application will use your newly trained BERT model. 