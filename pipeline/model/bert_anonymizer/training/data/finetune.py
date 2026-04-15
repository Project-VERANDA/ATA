
"""
Fine-tune a German BERT NER model on PHI annotations that are stored in
UIMA CAS JSON 0.4.0 files.

Folders expected
────────────────
DATA_DIR/
    train/   *.json   ← annotated CAS files
    dev/     *.json
    test/    *.json   (optional)

Outputs
───────
OUT_DIR/
    best-model/  ← ready to load in Pipeline2.py
"""

# --- CONFIG -----------------------------------------------------------------
DATA_DIR = '/Users/deryaerman/Desktop/School/Uni/Text-Anonymiser Project/Audio-Transcript-Anonymizer-TUB-AP/finetune/data/data(new split)'        # ..../train  ..../dev  ..../test
BASE_MODEL = "domischwimmbeck/bert-base-german-cased-fine-tuned-ner"       # or domischwimmbeck/...
#BASE_MODEL = "Davlan/distilbert-base-multilingual-cased-ner-hrl"
#BASE_MODEL = '/Users/deryaerman/Desktop/School/Uni/Text-Anonymiser Project/Audio-Transcript-Anonymizer-TUB-AP/finetune/finetuned_model/finetuned-bert-german-pii'  # or any other model from HuggingFace
#BASE_MODEL = 'deepset/gelectra-base'
OUT_DIR  = '/Users/deryaerman/Desktop/School/Uni/Text-Anonymiser Project/Audio-Transcript-Anonymizer-TUB-AP/finetune/finetuned_model'
BATCH = 4
EPOCHS = 4
# ---------------------------------------------------------------------------

import json, os, random, itertools, re
from pathlib import Path
from collections import defaultdict

from datasets import Dataset, DatasetDict
import evaluate
from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                          DataCollatorForTokenClassification,
                          TrainingArguments, Trainer)
            

metric = evaluate.load("seqeval")


# ---------- 1  READ + CONVERT UIMA CAS JSON → token list + BIO tags ---------
def cas_to_sentences(path):
    """
    Convert a UIMA-CAS JSON 0.4.0 file into
        [{tokens:[...], ner_tags:[...]} …]    (BIO tagging)
    Works for both the flat '%FEATURE_STRUCTURES' format and the older
    '_referenced_fss' layout.
    """
    import itertools, json, re
    with open(path, encoding="utf-8") as f:
        j = json.load(f)

    # -------- 1. Collect every feature structure --------------------------
    if "_referenced_fss" in j:                         # old layout
        fs_list = list(itertools.chain.from_iterable(j["_referenced_fss"].values()))
    elif "%FEATURE_STRUCTURES" in j:                  # your files
        fs_list = j["%FEATURE_STRUCTURES"]
    else:
        raise ValueError("Unsupported CAS layout in file: " + str(path))

    # -------- 2. Get raw text (sofaString) ---------------------------------
    sofa_obj = next(fs for fs in fs_list if fs["%TYPE"] == "uima.cas.Sofa")
    text     = sofa_obj["sofaString"]

    # -------- 3. Split out tokens and PHI spans ----------------------------
    TOK_TYPE_SUFFIX = ".Token"
    tokens_raw = [fs for fs in fs_list if fs["%TYPE"].endswith(TOK_TYPE_SUFFIX)]
    spans_raw  = [fs for fs in fs_list if fs["%TYPE"] == "custom.Span"]

    # char-offset → label map
    span_map = {}
    for s in spans_raw:
        lbl = s.get("label") or s.get("string")
        if not lbl:               # skip spans that carry no usable label
            continue
        span_map[(s["begin"], s["end"])] = lbl

    # sort tokens in reading order
    tokens_raw.sort(key=lambda t: t["begin"])

    SENT_END = re.compile(r"[.!?]$")   # crude sentence boundary
    sentences, cur_tok, cur_tag = [], [], []

    for tok in tokens_raw:
        word = text[tok["begin"]:tok["end"]]

        # default outside-entity tag
        lab  = "O"
        for (s, e), lbl in span_map.items():
            if s <= tok["begin"] < tok["end"] <= e:
                lab = "B-" + lbl if tok["begin"] == s else "I-" + lbl
                break

        cur_tok.append(word)
        cur_tag.append(lab)

        if SENT_END.search(word):
            sentences.append({"tokens": cur_tok, "ner_tags": cur_tag})
            cur_tok, cur_tag = [], []

    if cur_tok:                       # tail of file
        sentences.append({"tokens": cur_tok, "ner_tags": cur_tag})

    return sentences




def load_split(folder):
    """Return list of sentence dicts for one split folder."""
    data = []
    for fp in Path(folder).glob("*.json"):
        data.extend(cas_to_sentences(fp))
    return data


ds_dict = {}

for split in ("train", "test"):
    split_dir = Path(DATA_DIR) / split
    if split_dir.exists():
        ds_dict[split] = Dataset.from_list(load_split(split_dir))

datasets = DatasetDict(ds_dict)
print(datasets)

# ---------- 2  Build label-id maps -----------------------------------------
all_tags = set()
for split in datasets:                       # train  +  validation (+ test)
    for row in datasets[split]["ner_tags"]:
        all_tags.update(row)

labels = ["O"] + sorted(all_tags - {"O"})    # keep O first
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}

def encode_labels(example):
    example["ner_tags"] = [label2id[t] for t in example["ner_tags"]]
    return example

datasets = datasets.map(encode_labels, batched=False)

# ---------- 3  Tokenise + align word-pieces ---------------------------------
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

def tokenize_and_align(ex):
    tokenised = tokenizer(
        ex["tokens"],
        is_split_into_words=True,
        truncation=True
    )
    word_ids = tokenised.word_ids()
    aligned = []
    for w_id in word_ids:
        if w_id is None:
            aligned.append(-100)
        else:
            aligned.append(ex["ner_tags"][w_id])
    tokenised["labels"] = aligned
    return tokenised

tokenised_ds = datasets.map(tokenize_and_align, batched=False,
                            remove_columns=["tokens", "ner_tags"])

# ---------- 4  Model + Trainer ---------------------------------------------
model = AutoModelForTokenClassification.from_pretrained(
    BASE_MODEL,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True 
)

args = TrainingArguments(
    OUT_DIR,
    per_device_train_batch_size=BATCH,
    per_device_eval_batch_size=BATCH,
    gradient_accumulation_steps=4,
    num_train_epochs=EPOCHS,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    warmup_ratio=0.1,      # 10% of total steps used for linear warmup
    lr_scheduler_type="linear",
    weight_decay=0.01,
    fp16=False,
)

def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    labels_ = p.label_ids

    true_preds, true_labels = [], []
    for p_row, l_row in zip(preds, labels_):
        tp, tl = [], []
        for p_i, l_i in zip(p_row, l_row):
            if l_i != -100:
                tp.append(id2label[p_i])
                tl.append(id2label[l_i])
        true_preds.append(tp)
        true_labels.append(tl)
    return metric.compute(predictions=true_preds, references=true_labels)

collator = DataCollatorForTokenClassification(tokenizer)

trainer = Trainer(
    model,
    args,
    train_dataset=tokenised_ds["train"],
    eval_dataset=tokenised_ds["test"],
    data_collator=collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model(Path(OUT_DIR) / "best-model")
tokenizer.save_pretrained(Path(OUT_DIR) / "best-model")

print("\n=== FINAL EVAL on validation set ===")
final_metrics = trainer.evaluate()
for k, v in final_metrics.items():
    if k.startswith("eval_overall"):
        print(f"{k:20s} : {v:.4f}")

print("Model saved to", Path(OUT_DIR) / "best-model")
