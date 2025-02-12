import os
import json
import torch
import numpy as np
import torch.nn.functional as F
from tqdm.auto import tqdm
from datasets import Dataset
import evaluate  # ✅ Standard import

from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator
)
from peft import LoraConfig, get_peft_model, PeftModel

# ✅ Define Paths and Configuration
DATA_PATH = {
    "train": "covid-qa/covid-qa-train.json",
    "dev": "covid-qa/covid-qa-dev.json",
    "test": "covid-qa/covid-qa-test.json",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = "deepset/roberta-base-squad2"

# ✅ LoRA Configuration
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["query", "value"],
)

# ✅ Training Hyperparameters
TRAINING_ARGS = TrainingArguments(
    output_dir="./lora_roberta_qa",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    learning_rate=5e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=100,
    report_to="none",
)

# ✅ Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL)

# ✅ Load Model
model = AutoModelForQuestionAnswering.from_pretrained(MODEL)

# ✅ Apply LoRA
model = get_peft_model(model, peft_config)
model.to(DEVICE)

# ✅ Load Evaluation Metric
eval_metric = evaluate.load("squad")

def load_json_file_to_dict(file_name: str):
    return json.load(open(file_name))

def preprocess_data(data_dict: dict):
    """Pre-process data to SQuAD format"""
    temp = {"id": [], "title": [], "context": [], "question": [], "answers": []}
    for article in data_dict["data"]:
        for paragraph in article["paragraphs"]:
            for qa_pair in paragraph["qas"]:
                for ans in qa_pair["answers"]:
                    temp["answers"].append({"answer_start": [ans["answer_start"]], "text": [ans["text"]]})
                    temp["question"].append(qa_pair["question"])
                    temp["context"].append(paragraph["context"])
                    temp["title"].append(paragraph["document_id"])
                    temp["id"].append(qa_pair["id"])
    return temp

def load_data(split="dev"):
    return Dataset.from_dict(preprocess_data(load_json_file_to_dict(DATA_PATH[split])))

def tokenize_features(examples):
    """Tokenization function that also computes answer start/end positions."""
    examples["question"] = [q.lstrip() for q in examples["question"]]

    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=384,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    expanded_examples = {key: [] for key in examples.keys()}
    expanded_examples["start_positions"] = []
    expanded_examples["end_positions"] = []

    for i, sample_idx in enumerate(sample_mapping):
        for key in examples.keys():
            expanded_examples[key].append(examples[key][sample_idx])

        answers = examples["answers"][sample_idx]
        if len(answers["answer_start"]) == 0:
            expanded_examples["start_positions"].append(0)
            expanded_examples["end_positions"].append(0)
        else:
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            sequence_ids = tokenized_examples.sequence_ids(i)
            token_start_index = 0
            token_end_index = len(offset_mapping[i]) - 1

            while token_start_index < len(offset_mapping[i]) and (
                sequence_ids[token_start_index] != 1 or offset_mapping[i][token_start_index][0] <= start_char
            ):
                token_start_index += 1

            while token_end_index >= 0 and (
                sequence_ids[token_end_index] != 1 or offset_mapping[i][token_end_index][1] >= end_char
            ):
                token_end_index -= 1

            if token_start_index >= len(offset_mapping[i]) or token_end_index >= len(offset_mapping[i]):
                expanded_examples["start_positions"].append(0)
                expanded_examples["end_positions"].append(0)
            else:
                expanded_examples["start_positions"].append(token_start_index)
                expanded_examples["end_positions"].append(token_end_index)

    return expanded_examples | tokenized_examples

train_dataset = load_data("train").map(tokenize_features, batched=True, remove_columns=["title", "answers"])
dev_dataset = load_data("dev").map(tokenize_features, batched=True, remove_columns=["title", "answers"])
test_dataset = load_data("test").map(tokenize_features, batched=True, remove_columns=["title", "answers"])

trainer = Trainer(
    model=model,
    args=TRAINING_ARGS,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)

trainer.train()

predictions_output = trainer.predict(test_dataset)
start_logits, end_logits = predictions_output.predictions[:2]

with open("qa_evaluation_results.json", "w") as f:
    json.dump(predictions_output.metrics, f, indent=4)

print("Final Test Results:", json.dumps(predictions_output.metrics, indent=4))

model.save_pretrained("./lora_roberta_qa")
