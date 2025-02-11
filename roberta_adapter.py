import os
import json
import torch
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
    r=8,  # LoRA rank
    lora_alpha=16,  # Scaling factor
    lora_dropout=0.1,  # Dropout for LoRA layers
    target_modules=["query", "value"],  # Apply LoRA to attention layers
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