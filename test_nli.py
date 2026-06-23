from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

model_name = (
    "MoritzLaurer/deberta-v3-base-mnli-fever-anli"
)

print("loading tokenizer")

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)

print("loading model")

model = AutoModelForSequenceClassification.from_pretrained(
    model_name
)

print("success")