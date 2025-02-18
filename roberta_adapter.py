import os
import json
import torch
import random
import numpy as np
import torch.nn.functional as F  # ✅ For loss computation
from tqdm.auto import tqdm
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator
)
from peft import LoraConfig, get_peft_model, PeftModel  # ✅ LoRA Import

seed = 1234
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

MODEL_SAVE_PATH = "./covid_qa_roberta_lora"

# ✅ Define Paths and Configuration
DATA_PATH = {
    "train": "covid-qa/covid-qa-train.json",
    "dev": "covid-qa/covid-qa-dev.json",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = "deepset/roberta-base-squad2"

# ✅ LoRA Configuration
peft_config = LoraConfig(
    r=8,  # LoRA rank
    lora_alpha=16,  # Scaling factor
    lora_dropout=0.1,  # Dropout for LoRA layers
    target_modules=["query", "value"],  # Apply LoRA to attention layers
)

# ✅ Training Hyperparameters
TRAINING_ARGS = TrainingArguments(
    output_dir=MODEL_SAVE_PATH,
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

def load_json_file_to_dict(file_name: str):
    """Load JSON file into a dictionary"""
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
    """Load dataset from JSON"""
    data_dict = load_json_file_to_dict(DATA_PATH[split])
    return Dataset.from_dict(preprocess_data(data_dict))

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
        return_offsets_mapping=True,  # Needed for mapping answer positions
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        sequence_ids = tokenized_examples.sequence_ids(i)
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]

        # If no valid answer, assign CLS token as answer position
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Find token positions
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if tokenizer.padding_side == "right" else 0):
                token_start_index += 1

            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if tokenizer.padding_side == "right" else 0):
                token_end_index -= 1

            # Check if answer is inside the span
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)

                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples

# ✅ Load and preprocess datasets separately
train_dataset = load_data("train")
dev_dataset = load_data("dev")

# ✅ Tokenize train and validation sets separately
tokenized_train_dataset = train_dataset.map(tokenize_features, batched=True, remove_columns=train_dataset.column_names)
tokenized_dev_dataset = dev_dataset.map(tokenize_features, batched=True, remove_columns=dev_dataset.column_names)

class QuestionAnsweringTrainer(Trainer):
    """Custom Trainer that computes loss for Question Answering"""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Custom loss function"""

        outputs = model(**inputs)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        start_positions = inputs["start_positions"]
        end_positions = inputs["end_positions"]

        # Compute loss (CrossEntropyLoss)
        loss_start = F.cross_entropy(start_logits, start_positions)
        loss_end = F.cross_entropy(end_logits, end_positions)
        loss = (loss_start + loss_end) / 2  # Average both losses
        
        return (loss, outputs) if return_outputs else loss

# ✅ Initialize Trainer
trainer = QuestionAnsweringTrainer(
    model=model,
    args=TRAINING_ARGS,
    train_dataset=tokenized_train_dataset,  
    eval_dataset=tokenized_dev_dataset,  
    processing_class=tokenizer,
)

# ✅ Train Model
trainer.train()

# Save fine-tuned model with LoRA
model.save_pretrained(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)

print(f"Fine-tuning completed with LoRA. Model saved to {MODEL_SAVE_PATH}")