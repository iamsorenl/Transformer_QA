import json

# Load gold test data
with open("covid-qa/covid-qa-test.json") as f:
    gold_data = json.load(f)

# Load predictions
with open("pred_test.json") as f:
    pred_data = json.load(f)

# Extract QIDs from both
gold_qids = set(q["id"] for article in gold_data["data"] for p in article["paragraphs"] for q in p["qas"])
pred_qids = set(pred_data.keys())

# Find missing QIDs
missing_qids = gold_qids - pred_qids

print(f"Total Questions in Test Set: {len(gold_qids)}")
print(f"Total Predictions in pred_test.json: {len(pred_qids)}")
print(f"Missing Predictions: {len(missing_qids)}")

if missing_qids:
    print("⚠️ The following QIDs are missing from pred_test.json:")
    print(missing_qids)
