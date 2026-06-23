from fastapi import FastAPI
from pydantic import BaseModel
import time
import numpy as np

from datasets import load_dataset

from agents.retrieval_agent import RetrievalAgent
from agents.evidence_agent import EvidenceAgent
from agents.trust_agent import TrustAgent
from agents.decision_agent import DecisionAgent
from agents.generator_agent import GeneratorAgent

from utils.reliability_memory import ReliabilityMemory
from utils.cluster_utils import cluster_agreement

# -----------------------------------
# FastAPI app
# -----------------------------------

app = FastAPI(title="AURORA-v2", version="2.0")

# -----------------------------------
# Load corpus
# -----------------------------------

print("[AURORA] Loading dataset...")

dataset = load_dataset(
    "mukulb/combined_medical_corpus",
    split="train"
)

corpus = dataset["text"]

print(f"[AURORA] Dataset loaded — {len(corpus)} documents.")

# -----------------------------------
# Initialize agents
# -----------------------------------

retrieval_agent   = RetrievalAgent(corpus)
evidence_agent    = EvidenceAgent()
trust_agent       = TrustAgent()
decision_agent    = DecisionAgent()
generator_agent   = GeneratorAgent()
reliability_memory = ReliabilityMemory()

# -----------------------------------
# Request schema
# -----------------------------------

class QueryRequest(BaseModel):
    query: str

# -----------------------------------
# Endpoints
# -----------------------------------

@app.get("/health")
def health():
    return {"status": "healthy", "model": "AURORA-v2"}


@app.get("/metrics")
def metrics():
    return {
        "reliability_window_size": reliability_memory.size(),
        "historical_reliability":  round(reliability_memory.score(), 4)
    }


@app.post("/ask")
def ask(request: QueryRequest):

    start_time = time.time()
    query = request.query

    # ─── Agent 1: Retrieval ──────────────────────────────────────
    retrieval = retrieval_agent.retrieve(query)

    docs            = retrieval["docs"]
    margin          = retrieval["margin"]
    bucket          = retrieval["bucket"]
    noise           = retrieval["noise"]
    top1_score      = retrieval["top1_score"]
    top5_variance   = retrieval["top5_variance"]
    cluster_entropy = retrieval["cluster_entropy"]

    # ─── Agent 2: Evidence Verification ─────────────────────────
    # verify(query, docs) → support / contradiction / neutral (means)
    evidence = evidence_agent.verify(query, docs)

    ev_support           = evidence["support"]
    ev_contradiction     = evidence["contradiction"]
    ev_neutral           = evidence["neutral"]
    ev_support_max       = evidence.get("support_max", ev_support)
    ev_contradiction_max = evidence.get("contradiction_max", ev_contradiction)
    ev_question_sim_mean = evidence.get("question_sim_mean", 0.0)
    ev_question_sim_max  = evidence.get("question_sim_max", 0.0)

    # ─── Agent 3: Trust Scoring ──────────────────────────────────
    CA = cluster_agreement(docs)
    HR = reliability_memory.score()

    trust_score = trust_agent.predict(
        retrieval_margin        = margin,
        top1_score              = top1_score,
        top5_variance           = top5_variance,
        evidence_support        = ev_support,
        evidence_contradiction  = ev_contradiction,
        evidence_neutral        = ev_neutral,
        evidence_support_max       = ev_support_max,
        evidence_contradiction_max = ev_contradiction_max,
        question_sim_mean          = ev_question_sim_mean,
        question_sim_max           = ev_question_sim_max,
        cluster_agreement          = CA,
        cluster_entropy         = cluster_entropy,
        historical_reliability  = HR,
        query_length            = len(query.split()),
        escalation_count        = 1 if margin < 0.20 else 0
    )

    # ─── Agent 4: Decision ───────────────────────────────────────
    decision = decision_agent.decide(trust_score)

    # ─── Generate Answer ─────────────────────────────────────────
    if decision == "insufficient":
        answer = (
            "Insufficient evidence available to answer reliably. "
            "Please consult a qualified healthcare professional."
        )

    elif decision == "escalate":
        # Expand the query and re-retrieve
        expanded_retrieval = retrieval_agent.retrieve(query + " detailed explanation")
        docs = expanded_retrieval["docs"]
        answer = generator_agent.generate(
            query=query,
            docs=docs,
            trust_score=trust_score
        )

    else:  # "generate"
        answer = generator_agent.generate(
            query=query,
            docs=docs,
            trust_score=trust_score
        )

    # ─── Update Reliability Memory ───────────────────────────────
    reliability_memory.update(trust_score)

    processing_time = round(time.time() - start_time, 2)

    # ─── Response ────────────────────────────────────────────────
    return {
        "answer":          answer,
        "trust_score":     round(trust_score, 4),
        "decision":        decision,
        "processing_time": processing_time,
        "retrieval": {
            "margin": round(margin, 4),
            "bucket": bucket,
            "noise":  round(noise, 4),
        },
        "evidence": {
            "support":       round(ev_support, 4),
            "contradiction": round(ev_contradiction, 4),
            "neutral":       round(ev_neutral, 4),
        },
        "trust_features": {
            "cluster_agreement":      round(CA, 4),
            "cluster_entropy":        round(cluster_entropy, 4),
            "historical_reliability": round(HR, 4),
            "question_sim_mean":      round(ev_question_sim_mean, 4),
            "question_sim_max":       round(ev_question_sim_max, 4),
            "top1_score":             round(top1_score, 4),
            "top5_variance":          round(top5_variance, 4),
            "retrieval_margin":       round(margin, 4),
        }
    }


# -----------------------------------
# Entry point
# -----------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)