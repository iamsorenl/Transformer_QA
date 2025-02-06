from transformers import pipeline
import argparse

# Load RoBERTa fine-tuned on SQuAD 2.0
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Define a list of test cases with context and corresponding questions
# Wikis used:
# - https://en.wikipedia.org/wiki/Jedi
# - https://en.wikipedia.org/wiki/Frodo_Baggins
# - https://en.wikipedia.org/wiki/University_of_California,_Santa_Cruz
test_cases_part_1 = [
    {
        "context": "Within the Star Wars galaxy, the Jedi are powerful guardians of order and justice, who, through intuition, rigorous training, and intensive self-discipline, are able to wield a supernatural power known as the Force, thus achieving the ability to move objects with the mind, perform incredible feats of strength, and connect to certain people's thoughts.",
        "question": "What is the supernatural power that the Jedi can wield?"  # Specific question
    },
    {
        "context": "Frodo is repeatedly wounded during the quest and becomes increasingly burdened by the Ring as it nears Mordor.",
        "question": "Who is wounded during the quest?"  # Specific question
    },
    {
        "context": "Founded in 1965, UC Santa Cruz began with the intention to showcase progressive, cross-disciplinary undergraduate education, innovative teaching methods and contemporary architecture. The residential college system consists of ten small colleges that were established as a variation of the Oxbridge collegiate university system.",
        "question": "When was UC Santa Cruz founded?"  # Specific question
    },
]

# Mapping test sets
test_sets = {
    1: test_cases_part_1,
    #2: test_cases_part_2,
    #3: test_cases_part_3,
}

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--test_set", type=int, choices=[1, 2, 3], default=1, help="Select test set: 1 (In-Domain), 2 (Edited), 3 (Out-of-Domain)")
args = parser.parse_args()

# Get the selected test set
test_cases = test_sets[args.test_set]

# Iterate through test cases and evaluate RoBERTa's responses
for i, case in enumerate(test_cases, 1):
    print(f"Test Case {i}:")
    print(f"Question: {case['question']}")
    print(f"Context: {case['context']}")
    result = qa_pipeline(question=case["question"], context=case["context"])
    print(f"Answer: {result['answer']}\nConfidence: {result['score']:.4f}\n")
    print("-" * 50)
