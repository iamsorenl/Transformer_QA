import json

DATA_PATH = {
    "train": "covid-qa/covid-qa-train.json",
    "dev": "covid-qa/covid-qa-dev.json",
    "test": "covid-qa/covid-qa-test.json",
}

def convert_to_squad_format(input_path: str, output_path: str):
    """
    Convert a COVID-QA dataset into SQuAD format and save as a JSON file.
    
    :param input_path: Path to the input COVID-QA JSON file.
    :param output_path: Path to save the output SQuAD JSON file.
    """
    with open(input_path, "r", encoding="utf-8") as file:
        data_dict = json.load(file)
    
    squad_data = {"version": "1.0", "data": []}
    
    for article in data_dict["data"]:
        squad_article = {"title": article.get("title", "untitled"), "paragraphs": []}
        
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            paragraph_data = {"context": context, "qas": []}
            
            for qa_pair in paragraph["qas"]:
                qas_entry = {
                    "id": qa_pair["id"],
                    "question": qa_pair["question"],
                    "answers": [],
                    "is_impossible": qa_pair.get("is_impossible", False)
                }
                
                if not qas_entry["is_impossible"]:
                    for ans in qa_pair["answers"]:
                        qas_entry["answers"].append({
                            "text": ans["text"],
                            "answer_start": ans["answer_start"]
                        })
                
                paragraph_data["qas"].append(qas_entry)
            
            squad_article["paragraphs"].append(paragraph_data)
        
        squad_data["data"].append(squad_article)
    
    with open(output_path, "w", encoding="utf-8") as outfile:
        json.dump(squad_data, outfile, indent=4, ensure_ascii=False)
    
    print(f"Converted SQuAD dataset saved to {output_path}")

# Convert all splits
for split, input_path in DATA_PATH.items():
    output_path = input_path.replace(".json", "-squad.json")
    convert_to_squad_format(input_path, output_path)
