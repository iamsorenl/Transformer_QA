from transformers import DefaultDataCollator
from peft import LoraConfig, get_peft_model, PeftModel

data_collator = DefaultDataCollator()

from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer

model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")