"""
Train the XGBoost trust model.

Features (15 total):
  - 3 retrieval signals
  - 5 NLI signals (answer vs query)
  - 2 question similarity signals  ← new
  - 2 cluster signals
  - 1 history signal
  - 2 query metadata signals

Label: NLI(answer vs gold) compound criterion — independent of features.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import xgboost as xgb
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)

FEATURE_COLS = [
    # Retrieval signals
    "retrieval_margin",
    "top1_score",
    "top5_variance",
    # NLI: answer vs query
    "evidence_support_mean",
    "evidence_support_max",
    "evidence_contradiction_mean",
    "evidence_contradiction_max",
    "evidence_neutral_mean",
    # Question similarity (new)
    "question_sim_mean",
    "question_sim_max",
    # Cluster signals
    "cluster_agreement",
    "cluster_entropy",
    # Session history
    "historical_reliability",
    # Query metadata
    "query_length",
    "escalation_count",
]

# --------------------------------------------------
# Load
# --------------------------------------------------

df = pd.read_csv("trust_training.csv")

print(f"\nDataset shape: {df.shape}")
print("\nLabel distribution:")
print(df["label"].value_counts())
print(df["label"].value_counts(normalize=True).round(3))

missing = [c for c in FEATURE_COLS if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

X = df[FEATURE_COLS]
y = df["label"]

neg   = (y == 0).sum()
pos   = (y == 1).sum()
scale = round(neg / pos, 2)
print(f"\nscale_pos_weight: {scale}")

# --------------------------------------------------
# Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Train: {len(X_train)}  Test: {len(X_test)}")

# --------------------------------------------------
# Train
# --------------------------------------------------

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train, y_train)

# --------------------------------------------------
# Evaluate
# --------------------------------------------------

pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]

print("\n=== Evaluation ===")
print(f"Accuracy:  {accuracy_score(y_test, pred):.4f}")
print(f"Precision: {precision_score(y_test, pred, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test, pred, zero_division=0):.4f}")
print(f"F1:        {f1_score(y_test, pred, zero_division=0):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, prob):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, pred, zero_division=0))

importance = pd.Series(
    model.feature_importances_, index=FEATURE_COLS
).sort_values(ascending=False)

print("\nFeature Importances:")
print(importance.round(4))

# --------------------------------------------------
# Save
# --------------------------------------------------

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/trust_xgboost.pkl")
print("\nTrust model saved to models/trust_xgboost.pkl")