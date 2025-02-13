import os
import json
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

# ✅ Define paths
DATA_PATH = {
    "train": "covid-qa/covid-qa-train-squad.json",
    "dev": "covid-qa/covid-qa-dev-squad.json",
    "test": "covid-qa/covid-qa-test-squad.json",
}

SAVE_DIR = "covid_qa_tokenized"
MODEL_NAME = "deepset/roberta-base-squad2"

# ✅ Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
pad_on_right = tokenizer.padding_side == "right"

# ✅ Tokenization Parameters
MAX_LEN = 384
DOC_STRIDE = 128

def load_squad_data(split="dev"):
    """Load a JSON SQuAD dataset into HuggingFace Dataset format."""
    with open(DATA_PATH[split], "r", encoding="utf-8") as f:
        squad_data = json.load(f)
    
    # Flatten SQuAD data
    data = []
    for article in squad_data["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                data.append({
                    "id": qa["id"],
                    "question": qa["question"],
                    "context": context,
                    "answers": qa["answers"],
                })

    return Dataset.from_list(data)

def tokenize_features(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
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
        start_char = answer[0]["answer_start"]
        end_char = answer[0]["answer_start"] + len(answer[0]["text"])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
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

def tokenize_and_save():
    """Tokenizes the dataset and saves it in Arrow format."""
    os.makedirs(SAVE_DIR, exist_ok=True)

    for split in ["train", "dev", "test"]:
        print(f"🔹 Tokenizing {split} set...")
        dataset = load_squad_data(split)
        tokenized_dataset = dataset.map(tokenize_features, batched=True, remove_columns=["context", "question", "answers"])
        tokenized_dataset.save_to_disk(os.path.join(SAVE_DIR, f"{split}.arrow"))

    print(f"🎉 Tokenized datasets saved in `{SAVE_DIR}`.")


# Run tokenization & save
tokenize_and_save()
