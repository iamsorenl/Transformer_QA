import json

DATA_PATHS = {
    "train": "covid-qa/covid-qa-train-squad.json",
    "dev": "covid-qa/covid-qa-dev-squad.json",
    "test": "covid-qa/covid-qa-test-squad.json",
}

def validate_squad_format(file_path):
    """Validates SQuAD-formatted JSON to check for inconsistencies."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    errors = []
    
    for article in data["data"]:
        if "title" not in article:
            errors.append(f"❌ Missing 'title' in {file_path}")

        for paragraph in article["paragraphs"]:
            context = paragraph.get("context", "")

            if not isinstance(context, str) or len(context) == 0:
                errors.append(f"❌ Invalid 'context' in {file_path}")

            for qa in paragraph["qas"]:
                q_id = qa.get("id", "UNKNOWN_ID")
                question = qa.get("question", "")

                if not isinstance(question, str) or len(question) == 0:
                    errors.append(f"❌ Missing or invalid 'question' for ID {q_id} in {file_path}")

                is_impossible = qa.get("is_impossible", False)
                answers = qa.get("answers", [])

                if not is_impossible:
                    if len(answers) == 0:
                        errors.append(f"❌ No answers for non-impossible question ID {q_id} in {file_path}")

                    for ans in answers:
                        text = ans.get("text", "")
                        start = ans.get("answer_start", -1)

                        if not isinstance(text, str) or len(text) == 0:
                            errors.append(f"❌ Empty answer text for ID {q_id} in {file_path}")
                        
                        if not isinstance(start, int) or start < 0 or start >= len(context):
                            errors.append(f"❌ Invalid answer_start {start} for ID {q_id} (context length: {len(context)}) in {file_path}")

    return errors

# ✅ Run validation on all splits
for split, path in DATA_PATHS.items():
    print(f"🔍 Validating {split} set...")
    issues = validate_squad_format(path)
    if issues:
        print("\n".join(issues))
    else:
        print(f"✅ {split} set is correctly formatted!")

