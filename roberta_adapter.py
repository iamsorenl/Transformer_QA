import json
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer, DefaultDataCollator
from datasets import Dataset
import numpy as np
import torch
import random
from peft import LoraConfig, get_peft_model

seed = 1234
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

MODEL_SAVE_PATH = "./covid_qa_roberta_lora"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "deepset/roberta-base-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the Covid-QA training and validation datasets
DATA_FILES = {
    "train": "covid-qa/covid-qa-train.json",
    "dev": "covid-qa/covid-qa-dev.json"
}

datasets = {split: json.load(open(file, "r", encoding="utf-8")) for split, file in DATA_FILES.items()}

train_data = datasets["train"]
dev_data = datasets["dev"]

# Process dataset into Hugging Face format
def process_data(data):
    processed = {"id": [], "context": [], "question": [], "answers": []}
    for entry in data["data"]:
        for paragraph in entry["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                if not qa["answers"]:
                    continue  # Skip questions with no answers
                processed["id"].append(qa["id"])
                processed["context"].append(context)
                processed["question"].append(qa["question"])
                processed["answers"].append({
                    "text": [ans["text"] for ans in qa["answers"]],
                    "answer_start": [ans["answer_start"] for ans in qa["answers"]]
                })
    return Dataset.from_dict(processed)

# Convert datasets
train_dataset = process_data(train_data)
dev_dataset = process_data(dev_data)

def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=512,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )
    
    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []
    
    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)
        
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1
        
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)
            
            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)
    
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

# Tokenize datasets
train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=["id", "question", "context", "answers"])
dev_dataset = dev_dataset.map(preprocess_function, batched=True, remove_columns=["id", "question", "context", "answers"])

data_collator = DefaultDataCollator()

# Load model
model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)

# Apply LoRA configuration
lora_config = LoraConfig(
    r=8,  # Rank of LoRA matrices
    lora_alpha=32,  # Scaling factor
    target_modules=["query", "value"],  # Apply LoRA to attention layers
    lora_dropout=0.1,
    bias="none"
)

# Wrap the model with LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir=MODEL_SAVE_PATH,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=False,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,
)

trainer.train()

# Save fine-tuned model with LoRA
model.save_pretrained(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)

print(f"Fine-tuning completed with LoRA. Model saved to {MODEL_SAVE_PATH}")