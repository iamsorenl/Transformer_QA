import json
from transformers import AutoTokenizer

# ✅ Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")

def validate_tokenization(debug_output):
    """Validates if tokenized start/end positions match the expected answer."""
    expected_answer = debug_output["answers"][0]["text"]
    start_char = debug_output["answers"][0]["answer_start"]
    end_char = start_char + len(expected_answer)

    # ✅ Tokenize context with offset mapping
    tokenized = tokenizer(
        debug_output["context"],
        truncation=True,
        max_length=384,
        return_offsets_mapping=True,  # ✅ Request offset mapping
        padding="max_length",
    )

    input_ids = tokenized["input_ids"]
    offsets = tokenized["offset_mapping"]

    # ✅ Convert character positions to token positions
    start_pos, end_pos = None, None
    for idx, (start, end) in enumerate(offsets):
        if start <= start_char < end and start_pos is None:
            start_pos = idx
        if start < end_char <= end:
            end_pos = idx

    # ✅ Debugging: Check if we found positions
    if start_pos is None or end_pos is None:
        print("❌ Failed to find correct token positions!")
        return

    # ✅ Extract answer using token indices
    extracted_tokens = tokenizer.convert_ids_to_tokens(input_ids[start_pos:end_pos + 1])
    extracted_answer = tokenizer.convert_tokens_to_string(extracted_tokens)

    print("\n🔍 **DEBUGGING TOKENIZATION**")
    print(f"✅ Expected Answer: {expected_answer}")
    print(f"📌 Start Char Index: {start_char}, End Char Index: {end_char}")
    print(f"🔢 Mapped Start Token: {start_pos}, End Token: {end_pos}")
    print(f"🔍 Extracted Answer: {extracted_answer}")

    # ✅ Check if the extracted answer matches the expected answer
    if extracted_answer.strip() == expected_answer.strip():
        print("✅ Tokenization is correct!")
    else:
        print("❌ Mismatch detected. Possible tokenization issue.")
        print(f"⚠️ Tokens Extracted: {extracted_tokens}")
        print(f"⚠️ Tokenized Context: {tokenizer.convert_ids_to_tokens(input_ids)}")  # Debug tokenization

# ✅ Load debug output from JSON file or use a hardcoded example
try:
    with open("debug_output.json", "r") as f:
        debug_output = json.load(f)
    print("✅ Loaded debug output from debug_output.json")
except FileNotFoundError:
    print("❌ debug_output.json not found. Using a hardcoded example instead.")
    debug_output = {
        "id": "4123",
        "question": "What growing dysjunction has been witnessed?",
        "context": "Globally, recent decades have witnessed a growing disjunction, a 'Valley of Death' no less...",
        "answers": [{"answer_start": 54, "text": "a 'Valley of Death' no less"}]
    }

# ✅ Run validation
validate_tokenization(debug_output)
