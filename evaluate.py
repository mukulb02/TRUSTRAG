"""
End-to-end evaluation of AURORA vs baselines on test_split.csv.

Baselines:
  1. dense_only       — dense retrieval only + BART (no BM25, no rerank, no trust)
  2. bm25_only        — BM25 retrieval only + BART (no dense, no rerank, no trust)
  3. hybrid_no_trust  — hybrid retrieval + rerank, unconditional generation
  4. threshold_rag    — hybrid + rerank, gated by fixed top1_score threshold
  5. aurora           — full AURORA pipeline with learned trust gating

Metrics:
  - ROUGE-1/2/L      (generation quality — word overlap)
  - BERTScore F1     (semantic fidelity)
  - Abstention rate  (% queries not answered)
  - False-gen rate   (% low-trust queries that still got answered)
  - Answer rate      (% queries that produced an answer)
"""
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import json
from tqdm import tqdm

import torch
from rouge_score import rouge_scorer
from bert_score import score as bert_score

from models.embedding_model import embed
from agents.retrieval_agent import RetrievalAgent
from agents.evidence_agent import EvidenceAgent
from agents.trust_agent import TrustAgent
from agents.decision_agent import DecisionAgent
from agents.generator_agent import GeneratorAgent
from utils.reliability_memory import ReliabilityMemory
from utils.cluster_utils import cluster_agreement

# --------------------------------------------------
# Config
# --------------------------------------------------

N_EVAL        = 500
THRESHOLD_T   = 9.8       # fixed top1_score threshold for threshold_rag
OUTPUT_JSON   = "evaluation_results.json"
RESULTS_CSV   = "evaluation_results.csv"
ABSTAIN_MSG   = "Insufficient evidence available to answer reliably."

# --------------------------------------------------
# Load data
# --------------------------------------------------

print("Loading data...")
df_test  = pd.read_csv("test_split.csv").head(N_EVAL)
df_train = pd.read_csv("train_split.csv")
corpus   = df_train["text"].tolist()
print(f"Test queries: {len(df_test)}")
print(f"Corpus size:  {len(corpus)}")

# --------------------------------------------------
# Initialise agents
# --------------------------------------------------

print("Initialising agents...")
retrieval_agent = RetrievalAgent(corpus)
evidence_agent  = EvidenceAgent()
trust_agent     = TrustAgent()
decision_agent  = DecisionAgent()
generator_agent = GeneratorAgent()
memory          = ReliabilityMemory()

# --------------------------------------------------
# ROUGE scorer
# --------------------------------------------------

rouge = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"],
    use_stemmer=True
)

# --------------------------------------------------
# Helpers
# --------------------------------------------------

def generate(query, docs, trust_score=0.8):
    if not docs:
        return ABSTAIN_MSG
    return generator_agent.generate(
        query=query, docs=docs, trust_score=trust_score
    )


def compute_rouge(hypothesis, reference):
    if not hypothesis or hypothesis.strip() == ABSTAIN_MSG:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    s = rouge.score(reference, hypothesis)
    return {
        "rouge1": s["rouge1"].fmeasure,
        "rouge2": s["rouge2"].fmeasure,
        "rougeL": s["rougeL"].fmeasure,
    }


# --------------------------------------------------
# Evaluation loop
# --------------------------------------------------

rows = []
print(f"\nRunning evaluation on {N_EVAL} queries...\n")

for i, (_, row) in enumerate(tqdm(df_test.iterrows(), total=len(df_test))):

    query = row["query"]
    gold  = row["answer"]

    try:
        # ── Embed query once ────────────────────────────────────
        query_vec = embed(query)

        # ── 1. Dense only ───────────────────────────────────────
        dense_pts  = retrieval_agent.dense_search(query_vec, limit=5)
        # Add rerank=0.5 placeholder so generator can process
        dense_docs = [
            {"text": d["text"], "rerank": 0.5, "cluster_id": d.get("cluster_id")}
            for d in dense_pts
        ]
        ans_dense = generate(query, dense_docs)

        # ── 2. BM25 only ────────────────────────────────────────
        bm25_raw  = retrieval_agent.sparse_search(query, k=5)
        bm25_docs = [
            {"text": d["text"], "rerank": d.get("bm25_score", 0.5), "cluster_id": None}
            for d in bm25_raw
        ]
        ans_bm25 = generate(query, bm25_docs)

        # ── Shared: hybrid retrieval + rerank ───────────────────
        retrieval   = retrieval_agent.retrieve(query)
        docs        = retrieval["docs"]           # top-5, reranked
        top1_score  = retrieval["top1_score"]
        margin      = retrieval["margin"]
        variance    = retrieval["top5_variance"]
        c_entropy   = retrieval["cluster_entropy"]

        # ── 3. Hybrid no trust ──────────────────────────────────
        ans_hybrid = generate(query, docs)

        # ── 4. Threshold RAG ────────────────────────────────────
        ans_threshold = (
            generate(query, docs)
            if top1_score >= THRESHOLD_T
            else ABSTAIN_MSG
        )

        # ── 5. AURORA ───────────────────────────────────────────
        evidence = evidence_agent.verify(query, docs)
        CA       = cluster_agreement(docs)
        HR       = memory.score()

        trust_score = trust_agent.predict(
            retrieval_margin           = margin,
            top1_score                 = top1_score,
            top5_variance              = variance,
            evidence_support           = evidence["support"],
            evidence_contradiction     = evidence["contradiction"],
            evidence_neutral           = evidence["neutral"],
            evidence_support_max       = evidence.get("support_max", 0.0),
            evidence_contradiction_max = evidence.get("contradiction_max", 0.0),
            question_sim_mean          = evidence.get("question_sim_mean", 0.0),
            question_sim_max           = evidence.get("question_sim_max", 0.0),
            cluster_agreement          = CA,
            cluster_entropy            = c_entropy,
            historical_reliability     = HR,
            query_length               = len(query.split()),
            escalation_count           = 1 if margin < 0.20 else 0
        )

        decision = decision_agent.decide(trust_score)
        memory.update(trust_score)

        if decision == "insufficient":
            ans_aurora = ABSTAIN_MSG
        elif decision == "escalate":
            exp       = retrieval_agent.retrieve(query + " detailed explanation")
            ans_aurora = generate(query, exp["docs"], trust_score)
        else:
            ans_aurora = generate(query, docs, trust_score)

        # ── ROUGE ───────────────────────────────────────────────
        r = {
            "dense":     compute_rouge(ans_dense,     gold),
            "bm25":      compute_rouge(ans_bm25,      gold),
            "hybrid":    compute_rouge(ans_hybrid,    gold),
            "threshold": compute_rouge(ans_threshold, gold),
            "aurora":    compute_rouge(ans_aurora,    gold),
        }

        rows.append({
            "query":    query,
            "gold":     gold,
            "group":    row.get("group_name", ""),

            # Answers
            "ans_dense":     ans_dense,
            "ans_bm25":      ans_bm25,
            "ans_hybrid":    ans_hybrid,
            "ans_threshold": ans_threshold,
            "ans_aurora":    ans_aurora,

            # Trust / routing
            "trust_score": round(trust_score, 4),
            "decision":    decision,

            # ROUGE
            "r1_dense":     r["dense"]["rouge1"],
            "r1_bm25":      r["bm25"]["rouge1"],
            "r1_hybrid":    r["hybrid"]["rouge1"],
            "r1_threshold": r["threshold"]["rouge1"],
            "r1_aurora":    r["aurora"]["rouge1"],

            "r2_dense":     r["dense"]["rouge2"],
            "r2_bm25":      r["bm25"]["rouge2"],
            "r2_hybrid":    r["hybrid"]["rouge2"],
            "r2_threshold": r["threshold"]["rouge2"],
            "r2_aurora":    r["aurora"]["rouge2"],

            "rL_dense":     r["dense"]["rougeL"],
            "rL_bm25":      r["bm25"]["rougeL"],
            "rL_hybrid":    r["hybrid"]["rougeL"],
            "rL_threshold": r["threshold"]["rougeL"],
            "rL_aurora":    r["aurora"]["rougeL"],

            # Abstention flags
            "abstained_threshold": int(ans_threshold == ABSTAIN_MSG),
            "abstained_aurora":    int(ans_aurora    == ABSTAIN_MSG),
        })

        if i % 50 == 0 and i > 0:
            print(
                f"\n[{i}] ROUGE-1 — "
                f"dense={r['dense']['rouge1']:.3f} "
                f"bm25={r['bm25']['rouge1']:.3f} "
                f"hybrid={r['hybrid']['rouge1']:.3f} "
                f"threshold={r['threshold']['rouge1']:.3f} "
                f"aurora={r['aurora']['rouge1']:.3f} | "
                f"decision={decision}"
            )

    except Exception as e:
        print(f"\nSkipped [{i}]: {e}")
        import traceback; traceback.print_exc()

# --------------------------------------------------
# Save per-query results
# --------------------------------------------------

df_results = pd.DataFrame(rows)
df_results.to_csv(RESULTS_CSV, index=False)
print(f"\nSaved per-query results to {RESULTS_CSV}")

# --------------------------------------------------
# Aggregate ROUGE
# --------------------------------------------------

SYSTEMS = ["dense", "bm25", "hybrid", "threshold", "aurora"]

print("\n" + "="*70)
print("EVALUATION RESULTS")
print("="*70)

print(f"\n{'System':<20} {'ROUGE-1':>10} {'ROUGE-2':>10} {'ROUGE-L':>10} {'Abstain%':>10} {'Answer%':>10}")
print("-" * 70)

results = {}
for sys in SYSTEMS:
    r1 = df_results[f"r1_{sys}"].mean()
    r2 = df_results[f"r2_{sys}"].mean()
    rL = df_results[f"rL_{sys}"].mean()

    if sys == "threshold":
        abstain_pct = 100 * df_results["abstained_threshold"].mean()
    elif sys == "aurora":
        abstain_pct = 100 * df_results["abstained_aurora"].mean()
    else:
        abstain_pct = 0.0

    results[sys] = {
        "rouge1": round(r1, 4),
        "rouge2": round(r2, 4),
        "rougeL": round(rL, 4),
        "abstain_rate": round(abstain_pct / 100, 4),
        "answer_rate":  round(1 - abstain_pct / 100, 4),
    }

    print(
        f"{sys:<20} {r1:>10.4f} {r2:>10.4f} {rL:>10.4f} "
        f"{abstain_pct:>9.1f}% {100-abstain_pct:>9.1f}%"
    )

# --------------------------------------------------
# AURORA routing breakdown
# --------------------------------------------------

print(f"\nAURORA routing distribution:")
routing = df_results["decision"].value_counts()
routing_pct = df_results["decision"].value_counts(normalize=True).round(3)
for dec in ["generate", "escalate", "insufficient"]:
    n   = int(routing.get(dec, 0))
    pct = float(routing_pct.get(dec, 0)) * 100
    print(f"  {dec:<15}: {n:>4} ({pct:.1f}%)")

# --------------------------------------------------
# False generation rate
# --------------------------------------------------

low_trust = df_results[df_results["trust_score"] < 0.24]
if len(low_trust) > 0:
    print(f"\nFalse generation rate (low-trust queries, trust < 0.24, n={len(low_trust)}):")
    print(f"  dense_only:      100.0% (always generates)")
    print(f"  bm25_only:       100.0% (always generates)")
    print(f"  hybrid_no_trust: 100.0% (always generates)")
    thr_fgr = (low_trust["ans_threshold"] != ABSTAIN_MSG).mean()
    aur_fgr = (low_trust["ans_aurora"]    != ABSTAIN_MSG).mean()
    print(f"  threshold_rag:   {100*thr_fgr:.1f}%")
    print(f"  aurora:          {100*aur_fgr:.1f}%")

    results["false_gen_rate"] = {
        "dense":     1.0,
        "bm25":      1.0,
        "hybrid":    1.0,
        "threshold": round(thr_fgr, 4),
        "aurora":    round(aur_fgr, 4),
    }

# --------------------------------------------------
# BERTScore
# --------------------------------------------------

print("\nComputing BERTScore (this may take a few minutes)...")
for sys in SYSTEMS:
    col      = f"ans_{sys}"
    answered = df_results[df_results[col] != ABSTAIN_MSG]
    if len(answered) > 0:
        _, _, F1 = bert_score(
            answered[col].tolist(),
            answered["gold"].tolist(),
            lang="en", verbose=False
        )
        results[sys]["bertscore_f1"] = round(float(F1.mean()), 4)
        print(f"  {sys:<20}: {results[sys]['bertscore_f1']:.4f}  (n={len(answered)})")
    else:
        results[sys]["bertscore_f1"] = 0.0

# --------------------------------------------------
# Save summary
# --------------------------------------------------

summary = {
    "n_evaluated": len(df_results),
    "systems":     results,
    "routing": {
        dec: {
            "count": int(routing.get(dec, 0)),
            "pct":   round(float(routing_pct.get(dec, 0)), 3)
        }
        for dec in ["generate", "escalate", "insufficient"]
    }
}

with open(OUTPUT_JSON, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nFull summary saved to {OUTPUT_JSON}")
print(json.dumps(summary, indent=2))
