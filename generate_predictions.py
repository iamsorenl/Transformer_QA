from transformers import pipeline
import json

# Load the RoBERTa model fine-tuned on SQuAD 2.0
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Function to generate predictions for a given dataset
def generate_predictions(input_file, output_file):
    with open(input_file) as f:
        dataset = json.load(f)

    predictions = {}
    total_qids = 0  # Track processed QIDs

    for article in dataset['data']:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                question = qa['question']
                qid = qa['id']
                total_qids += 1

                try:
                    answer = qa_pipeline(question=question, context=context)['answer']
                    predictions[qid] = answer  # Store answer with question ID
                except Exception as e:
                    print(f"⚠️ Error on QID {qid}: {e}")
                    predictions[qid] = ""  # Store empty answer for debugging

    print(f"Processed {total_qids} questions")
    print(f"Saving {len(predictions)} predictions to {output_file}")

    with open(output_file, "w") as f:
        json.dump(predictions, f)

# Run predictions for Dev and Test sets
generate_predictions("covid-qa/covid-qa-dev.json", "pred_dev.json")
generate_predictions("covid-qa/covid-qa-test.json", "pred_test.json")
