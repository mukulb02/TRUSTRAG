from datasets import load_dataset
from sklearn.model_selection import train_test_split

dataset = load_dataset(
    "mukulb/combined_medical_corpus",
    split="train"
)

dataset = dataset.shuffle(seed=42)

train_val, test = train_test_split(
    dataset,
    test_size=0.15,
    random_state=42
)

train, val = train_test_split(
    train_val,
    test_size=0.1765,
    random_state=42
)