import json

def validate_squad_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        squad_data = json.load(file)

    print(f"✅ Checking: {file_path}")
    
    # Check top-level keys
    if "version" not in squad_data or "data" not in squad_data:
        print("❌ Invalid JSON format: Missing 'version' or 'data' key")
        return
    
    # Check articles
    for article in squad_data["data"]:
        if "title" not in article or "paragraphs" not in article:
            print("❌ Missing 'title' or 'paragraphs' in an article")
            return
        
        for paragraph in article["paragraphs"]:
            if "context" not in paragraph or "qas" not in paragraph:
                print("❌ Missing 'context' or 'qas' in a paragraph")
                return
            
            for qa in paragraph["qas"]:
                if "id" not in qa or "question" not in qa or "answers" not in qa:
                    print("❌ Missing 'id', 'question', or 'answers' in a QA pair")
                    return
                
                if not qa["is_impossible"] and not qa["answers"]:
                    print(f"❌ Question {qa['id']} is answerable but has no answers!")
                    return

    print("✅ JSON structure is valid!\n")

# Validate all generated SQuAD JSON files
for split in ["train", "dev", "test"]:
    validate_squad_json(f"covid-qa/covid-qa-{split}-squad.json")
