import os
import torch
from transformers import BertForTokenClassification, BertTokenizerFast, Trainer, TrainingArguments
from torch.utils.data import Dataset
import json
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnonymizationDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def load_data(data_dir):

    """Loads UIMA CAS JSON data, skipping type system definitions."""
    texts = []
    annotations = []
    
    logger.info(f"Scanning directory: {data_dir}")
    
    for filename in os.listdir(data_dir):
        if not filename.endswith(".json"):
            continue
            
        filepath = os.path.join(data_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Skipping invalid JSON: {filename}")
            continue

        # 1. Skip Type System Definitions
        # These typically start with %TYPES or define types rather than instances
        if isinstance(data, dict) and ('%TYPES' in data or '%NAME' in data):
            logger.debug(f"Skipping type system file: {filename}")
            continue

        # 2. Extract Text (Sofa String)
        sofa_string = ""
        # Try common keys for the text content
        candidates = ['sofaString', 'text', 'sofaString', 'cas']
        
        # Check top level
        for key in candidates:
            if key in data:
                val = data[key]
                if isinstance(val, str):
                    sofa_string = val
                    break
                elif isinstance(val, dict) and 'sofaString' in val:
                    sofa_string = val['sofaString']
                    break
        
        # If still empty, maybe it's nested differently? 
        # Some WebAnno exports put text in 'sofa' -> 'sofaString'
        if not sofa_string and 'sofa' in data:
            if isinstance(data['sofa'], dict):
                sofa_string = data['sofa'].get('sofaString', '')
        
        if not sofa_string:
            logger.warning(f"Could not find text in {filename}. Skipping.")
            continue

        texts.append(sofa_string)

        # 3. Extract Annotations
        ents = []
        fs_list = []
        
        # Try to find the list of features/annotations
        # Common keys: 'fsArray', 'annotations', '_referenced_fss' (your original guess)
        annotation_keys = ['fsArray', 'annotations', '_referenced_fss', 'features']
        
        found_key = None
        for key in annotation_keys:
            if key in data:
                val = data[key]
                if isinstance(val, list):
                    fs_list = val
                    found_key = key
                    break
            # Check nested under 'cas'
            if 'cas' in data and key in data['cas']:
                val = data['cas'][key]
                if isinstance(val, list):
                    fs_list = val
                    found_key = key
                    break
        
        if not fs_list:
            logger.warning(f"No annotation list found in {filename} (tried keys: {annotation_keys}). Skipping.")
            annotations.append([])
            continue

        # Filter for 'custom.Span'
        for item in fs_list:
            if not isinstance(item, dict):
                continue
            if item.get('_type') == 'custom.Span':
                ents.append({
                    'label': item.get('label'),
                    'begin': item.get('begin'),
                    'end': item.get('end')
                })
        
        annotations.append(ents)
        logger.debug(f"Processed {filename}: {len(ents)} entities found.")

    return texts, annotationss

def main():
    # --- Configuration ---
    MODEL_NAME = 'bert-base-german-cased'
    TRAIN_DIR = 'bert_anonymizer/training/data/train'
    TEST_DIR = 'bert_anonymizer/training/data/test'
    OUTPUT_DIR = 'bert_anonymizer/training/bert_model_finetuned'

    # --- Load and Prepare Data ---
    logger.info("Loading and preparing data...")
    train_texts, train_annotations = load_data(TRAIN_DIR)
    val_texts, val_annotations = load_data(TEST_DIR)
    
    label_list = sorted(list(set(tag['label'] for doc_tags in train_annotations + val_annotations for tag in doc_tags)))
    label_map = {label: i for i, label in enumerate(label_list)}

    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    
    def tokenize_and_align_labels(texts, annotations):
        tokenized_inputs = tokenizer(texts, truncation=True, padding=True, is_split_into_words=False)
        labels = []
        for i, doc_annotations in enumerate(annotations):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_id = 'O'
                    for ent in doc_annotations:
                        if ent['begin'] <= tokenizer.convert_ids_to_tokens(tokenized_inputs['input_ids'][i])[word_idx].startswith('##') == False and \
                           ent['end'] >= tokenizer.convert_ids_to_tokens(tokenized_inputs['input_ids'][i]).find(tokenizer.convert_ids_to_tokens(tokenized_inputs['input_ids'][i])[word_idx]) + len(tokenizer.convert_ids_to_tokens(tokenized_inputs['input_ids'][i])[word_idx]):
                            label_id = ent['label']
                            break
                    label_ids.append(label_map.get(label_id, label_map['O']))
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        return tokenized_inputs, labels

    train_encodings, train_labels = tokenize_and_align_labels(train_texts, train_annotations)
    val_encodings, val_labels = tokenize_and_align_labels(val_texts, val_annotations)

    train_dataset = AnonymizationDataset(train_encodings, train_labels)
    val_dataset = AnonymizationDataset(val_encodings, val_labels)

    # --- Model Fine-tuning ---
    logger.info("Starting model fine-tuning...")
    model = BertForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(label_list))

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    trainer.train()
    logger.info("Fine-tuning complete.")

    # --- Save Model ---
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logger.info(f"Fine-tuned model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main() 
