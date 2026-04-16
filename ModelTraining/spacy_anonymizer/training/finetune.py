import os
import spacy
from spacy.tokens import DocBin
from spacy.cli.train import train as spacy_train
import argparse
import logging
from cassis import load_cas_from_json
from tqdm import tqdm
import time
import sys
import tempfile
from thinc.api import Config

# --- Configuration ---
# Ensure the project root is in the python path for consistent imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Data Processing Functions ---

def process_json_directory(directory_path):
    """Processes a directory of UIMA CAS JSON files."""
    data = []
    logging.info(f"Processing JSON files in: {directory_path}")
    for filename in os.listdir(directory_path):
        if not filename.endswith('.json'):
            continue
        
        file_path = os.path.join(directory_path, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                cas = load_cas_from_json(f)
            
            text = cas.get_sofa().sofaString
            if not text:
                logging.warning(f"No text found in {filename}, skipping.")
                continue

            entities = [(ann.begin, ann.end, ann.label) for ann in cas.select_all() if hasattr(ann, 'label') and ann.label]
            data.append({'text': text, 'entities': entities})
        except Exception as e:
            logging.error(f"Error processing {filename}: {e}")
    logging.info(f"Found {len(data)} files with text in {directory_path}")
    return data

def clean_entity_spans(text, entities):
    """Cleans entity spans by removing whitespace and invalid spans."""
    cleaned = []
    for start, end, label in entities:
        if start < 0 or end > len(text) or start >= end or not label: continue
        while start < end and text[start].isspace(): start += 1
        while end > start and text[end - 1].isspace(): end -= 1
        if start < end and text[start:end].strip():
            cleaned.append((start, end, label))
    return cleaned

def create_spacy_docbin(data, nlp, output_file):
    """Converts data into a .spacy file."""
    doc_bin = DocBin()
    for example in tqdm(data, desc=f"Creating {os.path.basename(output_file)}"):
        text = example["text"]
        labels = clean_entity_spans(text, example["entities"])
        doc = nlp.make_doc(text)
        ents = [doc.char_span(start, end, label=label, alignment_mode="expand") for start, end, label in labels]
        doc.ents = spacy.util.filter_spans([e for e in ents if e is not None])
        doc_bin.add(doc)
    doc_bin.to_disk(output_file)
    logging.info(f"Successfully created {output_file} with {len(doc_bin)} documents.")

# --- Main Finetuning Function ---

def main():
    parser = argparse.ArgumentParser(description="Finetune a spaCy NER model from UIMA CAS JSON files.")
    parser.add_argument("--train_dir", type=str, default="data/train", help="Directory with training JSON files, relative to training_root.")
    parser.add_argument("--dev_dir", type=str, default="data/test", help="Directory with development JSON files, relative to training_root.")
    parser.add_argument("--config_path", type=str, default="config.cfg", help="Path to the spaCy config file, relative to training_root.")
    parser.add_argument("--output_dir", type=str, default="../model", help="Directory to save the trained model, relative to training_root.")
    parser.add_argument("--use_gpu", type=int, default=0, help="GPU ID to use (0 for MPS/GPU, -1 for CPU).")
    args = parser.parse_args()

    tic = time.perf_counter()

    # Define the root for all training-related activities
    training_root = os.path.dirname(__file__)
    original_cwd = os.getcwd()

    # Define paths for the .spacy files relative to the training root
    training_path = os.path.join(training_root, "train.spacy")
    dev_path = os.path.join(training_root, "dev.spacy")

    try:
        # --- Step 1: Prepare data ---
        logging.info("--- Step 1: Preparing data ---")
        nlp = spacy.blank("de")
        
        train_data = process_json_directory(os.path.join(training_root, args.train_dir))
        if train_data: create_spacy_docbin(train_data, nlp, training_path)

        dev_data = process_json_directory(os.path.join(training_root, args.dev_dir))
        if dev_data: create_spacy_docbin(dev_data, nlp, dev_path)

        # --- Step 2: Change CWD and Train model ---
        # This is the key change: we run the trainer from within the training directory.
        # This ensures all relative paths in config.cfg are resolved correctly.
        logging.info(f"--- Step 2: Changing CWD to '{training_root}' for training ---")
        os.chdir(training_root)
        
        # We need to make sure the output directory exists from this new CWD
        os.makedirs(args.output_dir, exist_ok=True)
        
        # These overrides are now relative to the new CWD
        overrides = {
            "paths.train": "./train.spacy",
            "paths.dev": "./dev.spacy",
        }
        
        # If using a GPU, set the allocator to 'pytorch' for MPS compatibility
        if args.use_gpu >= 0:
            if spacy.util.is_package("torch"):
                overrides["training.gpu_allocator"] = "pytorch"
                logging.info("Set 'training.gpu_allocator' to 'pytorch' for MPS/GPU support.")
            else:
                logging.warning("PyTorch not found. GPU training might fail. Please install spacy[apple] or spacy[cuda].")

        spacy_train(args.config_path, args.output_dir, use_gpu=args.use_gpu, overrides=overrides)
        logging.info(f"Training complete. Model saved to '{os.path.abspath(args.output_dir)}'.")

    except Exception as e:
        logging.error(f"An error occurred during finetuning: {e}", exc_info=True)
    finally:
        # --- Step 3: Clean up and restore CWD ---
        os.chdir(original_cwd)
        logging.info(f"--- Step 3: Restored CWD to '{os.getcwd()}' ---")
        
        if os.path.exists(training_path):
            os.remove(training_path)
            logging.info(f"Removed temporary file: {training_path}")
        if os.path.exists(dev_path):
            os.remove(dev_path)
            logging.info(f"Removed temporary file: {dev_path}")

    toc = time.perf_counter()
    logging.info(f"Finetuning process finished in {toc - tic:0.4f} seconds.")

if __name__ == "__main__":
    main() 