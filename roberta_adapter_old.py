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

def postprocess_predictions(predictions, dataset):
    """Convert model logits into readable text predictions"""
    # Debug 
    print(f"Predictions type: {type(predictions)}")
    print(f"Predictions shape: {np.array(predictions).shape}")
    print(f"First 5 Predictions: {predictions[:5]}")

    if len(predictions) > 2:
        all_start_logits, all_end_logits = predictions[:2]  # ✅ Extract first two elements
    else:
        all_start_logits, all_end_logits = predictions

    formatted_predictions = []
    for i, example in enumerate(dataset):
        start_logits = all_start_logits[i]
        end_logits = all_end_logits[i]

        # Get the most probable start and end positions
        start_index = np.argmax(start_logits)
        end_index = np.argmax(end_logits)

        # Extract answer from original context
        context = example["context"]
        tokens = tokenizer.convert_ids_to_tokens(example["input_ids"])

        if start_index < len(tokens) and end_index < len(tokens) and start_index <= end_index:
            answer_tokens = tokens[start_index : end_index + 1]
            answer_text = tokenizer.convert_tokens_to_string(answer_tokens)
        else:
            answer_text = ""

        formatted_predictions.append({
            "id": example["id"],
            "prediction_text": answer_text
        })

    return formatted_predictions

def compute_metrics(eval_pred):
    """Compute Exact Match (EM) and F1 for QA with actual references."""
    predictions, _ = eval_pred

    # Convert raw logits to readable answers
    formatted_predictions = postprocess_predictions(predictions, test_dataset)

    # Load actual reference answers from test dataset
    test_data = load_data("test")  # Load full dataset to get real answers
    references = [
        {"id": ex["id"], "answers": ex["answers"]}
        for ex in test_data
    ]

    # Compute the real evaluation metrics
    results = eval_metric.compute(predictions=formatted_predictions, references=references)

    # Add additional fields
    results["total"] = len(predictions[0])  # Fix total to count predictions properly
    results["HasAns_total"] = results.get("total", 0)
    results["HasAns_exact"] = results.get("exact", 0)
    results["HasAns_f1"] = results.get("f1", 0)

    return results

train_dataset = load_data("train").map(tokenize_features, batched=True, remove_columns=["title", "answers"])
dev_dataset = load_data("dev").map(tokenize_features, batched=True, remove_columns=["title", "answers"])

test_features = load_data("test").map(tokenize_features, batched=True, remove_columns=["title", "answers"])

if "offset_mapping" in test_features.column_names:
    test_dataset = test_features.remove_columns(["offset_mapping"])
else:
    test_dataset = test_features

trainer = Trainer(
    model=model,
    args=TRAINING_ARGS,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

predictions_output = trainer.predict(test_dataset)

# Convert predictions properly
formatted_predictions = postprocess_predictions(predictions_output.predictions, test_dataset)

# Generate correct evaluation scores
metrics = compute_metrics((predictions_output.predictions, None))

# Save predictions separately for debugging
with open("qa_predictions.json", "w") as f:
    json.dump(formatted_predictions, f, indent=4, ensure_ascii=False)

# Save to JSON file
with open("qa_evaluation_results.json", "w") as f:
    json.dump(metrics, f, indent=4, ensure_ascii=False)

print("Final Test Results:", json.dumps(metrics, indent=4))
