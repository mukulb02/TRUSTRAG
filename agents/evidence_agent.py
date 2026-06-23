import re
import numpy as np
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from sentence_transformers import SentenceTransformer, util

# --------------------------------------------------
# Medical NLI model — trained on MedNLI clinical notes
# Better domain fit than general MNLI for healthcare text
# --------------------------------------------------
NLI_MODEL = "medical-nli/deberta-v3-base-mednli"

# --------------------------------------------------
# Medical semantic similarity model
# Trained on PubMed + MS-MARCO — suited for medical QA retrieval
# Used for question-to-question similarity scoring
# --------------------------------------------------
SIM_MODEL = "aleynahukmet/bge-medical-small-en-v1.5"  # same model as retrieval — better calibrated


def extract_question(doc_text: str) -> str:
    """
    Extract the <HUMAN> question from a Q&A formatted doc.
    Used for question similarity — compare retrieved question to input query.
    """
    match = re.search(r'<HUMAN>:\s*(.*?)(?:<ASSISTANT>|$)', doc_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return doc_text[:200]


def extract_answer(doc_text: str) -> str:
    """
    Extract the <ASSISTANT> answer from a Q&A formatted doc.
    Used for NLI — clean medical prose works better than raw tags.
    """
    match = re.search(r'<ASSISTANT>:\s*(.*)', doc_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return re.sub(r'<HUMAN>:|<ASSISTANT>:', '', doc_text).strip()


class EvidenceAgent:
    """
    Evidence verification agent with two complementary signals:

    1. Question Similarity (sentence-transformers)
       Measures whether the retrieved doc answers the same question
       as the input query. Natural fit for Q&A corpus structure.

    2. NLI Evidence Scores (medical NLI model)
       Measures whether the retrieved answer entails/contradicts the query.
       Uses clean answer text (strips <HUMAN>/<ASSISTANT> tags) for
       better alignment with model training distribution.

    These two signals are independent and complementary:
    - Similarity: topical/question-level match
    - NLI: semantic entailment at answer level
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # -- NLI model --
        print(f"[EvidenceAgent] Loading NLI model ({NLI_MODEL}) on {self.device}...")
        try:
            self.nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
            self.nli_model = (
                AutoModelForSequenceClassification
                .from_pretrained(NLI_MODEL)
                .to(self.device)
            )
            self.nli_model.eval()
            self.nli_available = True
            print("[EvidenceAgent] NLI model loaded.")
        except Exception as e:
            print(f"[EvidenceAgent] WARNING: Could not load NLI model: {e}")
            print("[EvidenceAgent] Falling back to deberta-v3-base-mnli-fever-anli")
            FALLBACK = "MoritzLaurer/deberta-v3-base-mnli-fever-anli"
            self.nli_tokenizer = AutoTokenizer.from_pretrained(FALLBACK)
            self.nli_model = (
                AutoModelForSequenceClassification
                .from_pretrained(FALLBACK)
                .to(self.device)
            )
            self.nli_model.eval()
            self.nli_available = True

        # -- Similarity model --
        print(f"[EvidenceAgent] Loading similarity model ({SIM_MODEL})...")
        try:
            self.sim_model = SentenceTransformer(SIM_MODEL, device=self.device)
            self.sim_available = True
            print("[EvidenceAgent] Similarity model loaded.")
        except Exception as e:
            print(f"[EvidenceAgent] WARNING: Could not load similarity model: {e}")
            self.sim_available = False

        print("[EvidenceAgent] Ready.")

    # --------------------------------------------------
    # NLI scoring — uses clean answer text as premise
    # --------------------------------------------------

    def _nli_scores(self, doc_text: str, hypothesis: str) -> dict:
        """
        Run NLI with clean answer text as premise.
        Strips <HUMAN>/<ASSISTANT> tags so NLI sees clean medical prose.
        """
        premise = extract_answer(doc_text)[:1500]

        inputs = self.nli_tokenizer(
            premise,
            hypothesis[:500],
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            probs = torch.softmax(
                self.nli_model(**inputs).logits, dim=1
            )[0]

        # DeBERTa MNLI label order: 0=contradiction, 1=neutral, 2=entailment
        return {
            "contradiction": probs[0].item(),
            "neutral":       probs[1].item(),
            "support":       probs[2].item(),
        }

    # --------------------------------------------------
    # Question similarity scoring
    # --------------------------------------------------

    def _question_similarity(self, docs: list, query: str) -> dict:
        """
        Compute cosine similarity between the input query and the
        question extracted from each retrieved Q&A doc.

        High similarity = retrieved doc answers the same question.
        This is the most natural signal for a Q&A corpus.
        """
        if not self.sim_available or not docs:
            return {"question_sim_mean": 0.0, "question_sim_max": 0.0}

        query_emb = self.sim_model.encode(query, normalize_embeddings=True)

        sims = []
        for doc in docs:
            q_text = extract_question(doc["text"])
            doc_emb = self.sim_model.encode(q_text, normalize_embeddings=True)
            sim = float(util.cos_sim(query_emb, doc_emb).item())
            sims.append(sim)

        return {
            "question_sim_mean": float(np.mean(sims)),
            "question_sim_max":  float(np.max(sims)),
        }

    # --------------------------------------------------
    # Shared NLI aggregation
    # --------------------------------------------------

    def _aggregate_nli(self, docs: list, hypothesis: str) -> dict:
        if not docs:
            return {
                "support_mean":          0.0,
                "support_max":           0.0,
                "contradiction_mean":    0.0,
                "contradiction_max":     0.0,
                "neutral_mean":          1.0,
            }

        supports, contradictions, neutrals = [], [], []

        for doc in docs:
            s = self._nli_scores(doc["text"], hypothesis)
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

    # --------------------------------------------------
    # Inference mode — called by main.py
    # Hypothesis = input query
    # --------------------------------------------------

    def verify(self, query: str, docs: list) -> dict:
        """
        Full evidence verification at inference time.
        Returns NLI scores (answer vs query) + question similarity scores.
        All signals available without gold answer.
        """
        nli = self._aggregate_nli(docs, query)
        sim = self._question_similarity(docs, query)

        return {
            # NLI signals
            "support":               nli["support_mean"],
            "contradiction":         nli["contradiction_mean"],
            "neutral":               nli["neutral_mean"],
            "support_max":           nli["support_max"],
            "contradiction_max":     nli["contradiction_max"],
            # Question similarity signals (new)
            "question_sim_mean":     sim["question_sim_mean"],
            "question_sim_max":      sim["question_sim_max"],
        }

    # --------------------------------------------------
    # Training mode — called by build_trust_training_dataset.py
    # Hypothesis = gold answer (label generation only)
    # --------------------------------------------------

    def verify_against_answer(self, docs: list, answer: str) -> dict:
        """
        Used during training to generate NLI features (doc vs query)
        and label signals (doc vs gold answer) separately.
        Returns full mean/max dict.
        """
        return self._aggregate_nli(docs, answer)

    def verify_generated(self, docs: list, generated_answer: str) -> dict:
        return self._aggregate_nli(docs, generated_answer)