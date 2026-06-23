"""
Build the trust training dataset.

Features:
  - Retrieval signals (margin, top1_score, top5_variance)
  - NLI(answer_of_doc, query)     — evidence alignment at inference time
  - Question similarity(doc_q, query) — question-level match (new)
  - Cluster signals, history, query metadata

Label (training only):
  - NLI(answer_of_doc, gold_answer) with compound criterion:
    best_ent > best_con AND best_ent > 0.15

Corpus is strictly separated from eval rows to prevent data leakage.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import re
import numpy as np
import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from sentence_transformers import SentenceTransformer, util

from agents.retrieval_agent import RetrievalAgent
from utils.cluster_utils import cluster_agreement, cluster_entropy
from utils.reliability_memory import ReliabilityMemory

# --------------------------------------------------
# Config
# --------------------------------------------------

N_SAMPLES    = 10000
NLI_MODEL    = "MoritzLaurer/deberta-v3-base-mnli-fever-anli"
SIM_MODEL    = "aleynahukmet/bge-medical-small-en-v1.5"  # same model as retrieval — better calibrated similarity
DEVICE       = "cuda"
OUTPUT_PATH  = "trust_training.csv"

# --------------------------------------------------
# Helper: extract question / answer from Q&A doc
# --------------------------------------------------

def extract_question(doc_text: str) -> str:
    match = re.search(r'<HUMAN>:\s*(.*?)(?:<ASSISTANT>|$)', doc_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return doc_text[:200]


def extract_answer(doc_text: str) -> str:
    match = re.search(r'<ASSISTANT>:\s*(.*)', doc_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return re.sub(r'<HUMAN>:|<ASSISTANT>:', '', doc_text).strip()

# --------------------------------------------------
# Load split — strict corpus/eval separation
# --------------------------------------------------

df_full = pd.read_csv("train_split.csv")
print(f"Full train split: {len(df_full)} rows")

if N_SAMPLES and N_SAMPLES < len(df_full):
    df_eval   = df_full.head(N_SAMPLES).copy()
    df_corpus = df_full.iloc[N_SAMPLES:].copy()
else:
    split     = int(len(df_full) * 0.8)
    df_eval   = df_full.iloc[:split].copy()
    df_corpus = df_full.iloc[split:].copy()

corpus = df_corpus["text"].tolist()
print(f"Eval rows:   {len(df_eval)}")
print(f"Corpus rows: {len(corpus)}  (no overlap with eval)")

# --------------------------------------------------
# Retrieval agent
# --------------------------------------------------

retrieval_agent = RetrievalAgent(corpus)
memory          = ReliabilityMemory()

# --------------------------------------------------
# NLI model — for FEATURES (doc vs query)
#           + LABELS   (doc vs gold answer)
# --------------------------------------------------

print(f"Loading NLI model ({NLI_MODEL})...")
nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
nli_model     = (
    AutoModelForSequenceClassification
    .from_pretrained(NLI_MODEL)
    .to(DEVICE)
)
nli_model.eval()
print("NLI model ready.")

# --------------------------------------------------
# Similarity model — for question similarity feature
# --------------------------------------------------

print(f"Loading similarity model ({SIM_MODEL})...")
try:
    sim_model     = SentenceTransformer(SIM_MODEL, device=DEVICE)
    sim_available = True
    print("Similarity model ready.")
except Exception as e:
    print(f"WARNING: Could not load similarity model: {e}")
    sim_available = False


def nli_scores(premise_text: str, hypothesis: str) -> dict:
    """NLI on clean answer text — strips tags before inference."""
    premise = extract_answer(premise_text)[:1500]
    inputs  = nli_tokenizer(
        premise, hypothesis[:500],
        truncation=True, max_length=512,
        padding=True, return_tensors="pt"
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        probs = torch.softmax(nli_model(**inputs).logits, dim=1)[0]

    return {
        "support":       probs[2].item(),
        "contradiction": probs[0].item(),
        "neutral":       probs[1].item(),
    }


def evidence_vs_query(docs: list, query: str) -> dict:
    """
    NLI(answer_of_doc, query) — inference-time feature.
    Measures whether retrieved answers support the query.
    """
    if not docs:
        return {
            "support_mean": 0.0, "support_max": 0.0,
            "contradiction_mean": 0.0, "contradiction_max": 0.0,
            "neutral_mean": 1.0,
        }

    supports, contradictions, neutrals = [], [], []
    for doc in docs:
        s = nli_scores(doc["text"], query)
        supports.append(s["support"])
        contradictions.append(s["contradiction"])
        neutrals.append(s["neutral"])

    return {
        "support_mean":          float(np.mean(supports)),
        "support_max":           float(np.max(supports)),
        "contradiction_mean":    float(np.mean(contradictions)),
        "contradiction_max":     float(np.max(contradictions)),
        "neutral_mean":          float(np.mean(neutrals)),
    }


def question_similarity(docs: list, query: str) -> dict:
    """
    Cosine similarity between input query and extracted question
    from each retrieved Q&A doc.
    High score = retrieved doc answers the same question.
    """
    if not sim_available or not docs:
        return {"question_sim_mean": 0.0, "question_sim_max": 0.0}

    query_emb = sim_model.encode(query, normalize_embeddings=True)
    sims = []

    for doc in docs:
        q_text  = extract_question(doc["text"])
        doc_emb = sim_model.encode(q_text, normalize_embeddings=True)
        sims.append(float(util.cos_sim(query_emb, doc_emb).item()))

    return {
        "question_sim_mean": float(np.mean(sims)),
        "question_sim_max":  float(np.max(sims)),
    }


def create_label(docs: list, answer: str):
    """
    NLI(answer_of_doc, gold_answer) — training label only.
    Label=1 if best doc is more supportive than contradictory
    AND has at least minimal entailment confidence (>0.15).
    """
    if not docs:
        return 0, 0.0

    best_ent = 0.0
    best_con = 0.0

    for doc in docs:
        s = nli_scores(doc["text"], answer)
        if s["support"] > best_ent:
            best_ent = s["support"]
            best_con = s["contradiction"]

    label = 1 if (best_ent > best_con and best_ent > 0.25) else 0
    return label, best_ent


# --------------------------------------------------
# Feature extraction loop
# --------------------------------------------------

rows = []

for i, (idx, row) in enumerate(df_eval.iterrows()):
    try:
        query  = row["query"]
        answer = row["answer"]

        retrieval = retrieval_agent.retrieve(query)
        docs      = retrieval["docs"]

        # Feature set 1: NLI(doc vs query) — inference-time signal
        ev  = evidence_vs_query(docs, query)

        # Feature set 2: Question similarity — new signal
        qs  = question_similarity(docs, query)

        # Label: NLI(doc vs gold answer) — training only
        label, best_ent = create_label(docs, answer)

        if i < 50:
            print(
                f"[{i}] BestEnt={best_ent:.3f} "
                f"Support(q)={ev['support_mean']:.3f} "
                f"QSim={qs['question_sim_mean']:.3f} "
                f"Label={label}"
            )

        CA = cluster_agreement(docs)
        CE = cluster_entropy(docs)
        HR = memory.score()

        rows.append({
            # Retrieval signals
            "retrieval_margin":          retrieval["margin"],
            "top1_score":                retrieval["top1_score"],
            "top5_variance":             retrieval["top5_variance"],
            # NLI: answer vs query
            "evidence_support_mean":     ev["support_mean"],
            "evidence_support_max":      ev["support_max"],
            "evidence_contradiction_mean": ev["contradiction_mean"],
            "evidence_contradiction_max":  ev["contradiction_max"],
            "evidence_neutral_mean":     ev["neutral_mean"],
            # Question similarity (new)
            "question_sim_mean":         qs["question_sim_mean"],
            "question_sim_max":          qs["question_sim_max"],
            # Cluster signals
            "cluster_agreement":         CA,
            "cluster_entropy":           CE,
            # Session history
            "historical_reliability":    HR,
            # Query metadata
            "query_length":              len(query.split()),
            "escalation_count":          1 if retrieval["margin"] < 0.20 else 0,
            # Label
            "label":                     label,
        })

        memory.update(float(label))

        if i % 500 == 0:
            print(f"Processed {i} / {len(df_eval)}")

    except Exception as e:
        print(f"Skipped {i}: {e}")

# --------------------------------------------------
# Save
# --------------------------------------------------

trust_df = pd.DataFrame(rows)
trust_df.to_csv(OUTPUT_PATH, index=False)

print(f"\nSaved {len(trust_df)} rows to {OUTPUT_PATH}")
print("\nLabel distribution:")
print(trust_df["label"].value_counts())
print(trust_df["label"].value_counts(normalize=True).round(3))