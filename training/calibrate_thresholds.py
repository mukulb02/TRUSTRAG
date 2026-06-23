"""
Calibrate decision thresholds for the trust model.
Finds probability cutoffs that correspond to meaningful
precision/recall tradeoffs on held-out data.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, roc_auc_score

FEATURE_COLS = [
    "retrieval_margin", "top1_score", "top5_variance",
    "evidence_support_mean", "evidence_support_max",
    "evidence_contradiction_mean", "evidence_contradiction_max",
    "evidence_neutral_mean",
    "question_sim_mean", "question_sim_max",
    "cluster_agreement", "cluster_entropy",
    "historical_reliability", "query_length", "escalation_count",
]

# --------------------------------------------------
# Load
# --------------------------------------------------

df = pd.read_csv("trust_training.csv")
print(f"Dataset: {df.shape}")

X = df[FEATURE_COLS]
y = df["label"]

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = joblib.load("models/trust_xgboost.pkl")
probs = model.predict_proba(X_test)[:, 1]

# --------------------------------------------------
# Probability distribution
# --------------------------------------------------

print(f"\nProbability distribution on test set:")
print(f"  min:    {probs.min():.4f}")
print(f"  max:    {probs.max():.4f}")
print(f"  mean:   {probs.mean():.4f}")
print(f"  median: {np.median(probs):.4f}")
print(f"  p25:    {np.percentile(probs, 25):.4f}")
print(f"  p50:    {np.percentile(probs, 50):.4f}")
print(f"  p75:    {np.percentile(probs, 75):.4f}")
print(f"  p90:    {np.percentile(probs, 90):.4f}")

print(f"\nBy label:")
print(f"  label=0 mean: {probs[y_test==0].mean():.4f}")
print(f"  label=1 mean: {probs[y_test==1].mean():.4f}")
print(f"  Separation:   {probs[y_test==1].mean() - probs[y_test==0].mean():.4f}")

# --------------------------------------------------
# Precision-recall curve — fix index bounds
# precision/recall have len N+1, thresholds have len N
# --------------------------------------------------

precision, recall, thresholds = precision_recall_curve(y_test, probs)

# Align: drop the last element of precision/recall (appended boundary value)
precision = precision[:-1]
recall    = recall[:-1]

# F1 at each threshold
f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
best_f1_idx       = int(np.argmax(f1_scores))
best_f1_threshold = float(thresholds[best_f1_idx])

print(f"\nBest F1 threshold: {best_f1_threshold:.4f}")
print(f"  Precision: {precision[best_f1_idx]:.4f}")
print(f"  Recall:    {recall[best_f1_idx]:.4f}")
print(f"  F1:        {f1_scores[best_f1_idx]:.4f}")

# --------------------------------------------------
# Find generate threshold (highest precision feasible)
# Walk from high threshold downward, find best precision
# --------------------------------------------------

print(f"\nPrecision at various thresholds:")
for t in [0.70, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20]:
    mask = probs >= t
    if mask.sum() > 0:
        prec = (y_test[mask] == 1).mean()
        rec  = mask[y_test == 1].mean()
        n    = mask.sum()
        print(f"  threshold={t:.2f}: precision={prec:.3f}  recall={rec:.3f}  n_queries={n} ({100*n/len(probs):.1f}%)")
    else:
        print(f"  threshold={t:.2f}: no queries above threshold")

# --------------------------------------------------
# Select thresholds
# generate  = best precision with at least 10% coverage
# escalate  = best F1 threshold (broader net)
# --------------------------------------------------

generate_threshold   = None
escalation_threshold = None

# Find generate threshold: best precision with >=10% coverage
for t in np.arange(0.90, 0.10, -0.01):
    mask = probs >= t
    if mask.sum() >= len(probs) * 0.10:
        prec = (y_test[mask] == 1).mean()
        if prec >= 0.35:
            generate_threshold = float(round(t, 2))
            break

if generate_threshold is None:
    # Fallback: p75 of probability distribution
    generate_threshold = float(round(np.percentile(probs, 75), 2))
    print(f"\nNo threshold achieves precision>=0.35 with 10% coverage")
    print(f"Using p75 as generate threshold: {generate_threshold:.4f}")

# Escalation threshold: p25 of distribution
# Queries below this are genuinely low confidence
escalation_threshold = float(round(np.percentile(probs, 25), 2))

print(f"\n{'='*50}")
print(f"RECOMMENDED THRESHOLDS:")
print(f"  generate_threshold   = {generate_threshold:.4f}")
print(f"  escalation_threshold = {escalation_threshold:.4f}")
print(f"{'='*50}")

# --------------------------------------------------
# Expected routing distribution
# --------------------------------------------------

n_total    = len(probs)
n_generate = (probs >= generate_threshold).sum()
n_escalate = ((probs >= escalation_threshold) & (probs < generate_threshold)).sum()
n_abstain  = (probs < escalation_threshold).sum()

print(f"\nExpected routing distribution:")
print(f"  generate:  {n_generate}/{n_total} ({100*n_generate/n_total:.1f}%)")
print(f"  escalate:  {n_escalate}/{n_total} ({100*n_escalate/n_total:.1f}%)")
print(f"  abstain:   {n_abstain}/{n_total} ({100*n_abstain/n_total:.1f}%)")

# Precision within each bucket
for name, mask in [
    ("generate", probs >= generate_threshold),
    ("escalate", (probs >= escalation_threshold) & (probs < generate_threshold)),
    ("abstain",  probs < escalation_threshold)
]:
    if mask.sum() > 0:
        prec = (y_test[mask] == 1).mean()
        print(f"  {name} precision (label=1 rate): {prec:.3f}")

# --------------------------------------------------
# Save
# --------------------------------------------------

os.makedirs("models", exist_ok=True)
out = {
    "generate_threshold":   generate_threshold,
    "escalation_threshold": escalation_threshold,
    "roc_auc":              round(float(roc_auc_score(y_test, probs)), 4),
    "label_separation":     round(float(probs[y_test==1].mean() - probs[y_test==0].mean()), 4),
    "note": "Calibrated from precision-recall analysis on held-out test split"
}

with open("models/trust_thresholds.json", "w") as f:
    json.dump(out, f, indent=2)

print(f"\nSaved to models/trust_thresholds.json")
print(json.dumps(out, indent=2))