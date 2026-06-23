"""
Splits mukulb/combined_medical_corpus into train / val / test CSVs.
Approximate split: 70% train / 15% val / 15% test.
"""

from datasets import load_dataset
from sklearn.model_selection import train_test_split
import pandas as pd

print("Loading dataset...")

dataset = load_dataset(
    "mukulb/combined_medical_corpus",
    split="train"
)

df = dataset.to_pandas()

print(f"Total rows: {len(df)}")

# First cut: 85% train+val, 15% test
train_val, test = train_test_split(
    df,
    test_size=0.15,
    random_state=42,
    shuffle=True
)

# Second cut: ~70% train, ~15% val  (0.1765 of 85% ≈ 15%)
train, val = train_test_split(
    train_val,
    test_size=0.1765,
    random_state=42,
    shuffle=True
)

train.to_csv("train_split.csv", index=False)
val.to_csv("val_split.csv",   index=False)
test.to_csv("test_split.csv",  index=False)

print(f"Train: {len(train)}")
print(f"Val:   {len(val)}")
print(f"Test:  {len(test)}")