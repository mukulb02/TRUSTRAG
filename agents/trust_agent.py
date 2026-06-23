import joblib
import numpy as np
import pandas as pd
import os

FEATURE_NAMES = [
    "retrieval_margin",
    "top1_score",
    "top5_variance",
    "evidence_support_mean",
    "evidence_support_max",
    "evidence_contradiction_mean",
    "evidence_contradiction_max",
    "evidence_neutral_mean",
    "question_sim_mean",
    "question_sim_max",
    "cluster_agreement",
    "cluster_entropy",
    "historical_reliability",
    "query_length",
    "escalation_count",
]

# Features that need normalisation due to corpus-size sensitivity
SCALE_FEATURES = [
    "retrieval_margin",
    "top5_variance",
]


class TrustAgent:
    """
    XGBoost trust scorer — 15 features.

    Normalises corpus-sensitive features (retrieval_margin, top5_variance)
    using statistics computed from the training data, ensuring inference
    feature distributions match training distributions regardless of
    corpus size differences.
    """

    def __init__(
        self,
        model_path: str = "models/trust_xgboost.pkl",
        training_data_path: str = "trust_training.csv"
    ):
        self.model = joblib.load(model_path)
        print("[TrustAgent] Model loaded.")

        # Load normalisation statistics from training data
        self.feature_stats = {}
        if os.path.exists(training_data_path):
            df = pd.read_csv(training_data_path)
            for feat in SCALE_FEATURES:
                if feat in df.columns:
                    self.feature_stats[feat] = {
                        "mean": float(df[feat].mean()),
                        "std":  float(df[feat].std()),
                        "min":  float(df[feat].min()),
                        "max":  float(df[feat].max()),
                    }
            print(f"[TrustAgent] Loaded normalisation stats for: {list(self.feature_stats.keys())}")
            for k, v in self.feature_stats.items():
                print(f"  {k}: mean={v['mean']:.3f} std={v['std']:.3f}")
        else:
            print("[TrustAgent] WARNING: training data not found, skipping normalisation")

    def _normalise(self, value: float, feat_name: str) -> float:
        """
        Clip feature to training distribution range.
        Prevents out-of-distribution values from pushing predictions to extremes.
        """
        if feat_name not in self.feature_stats:
            return value
        stats = self.feature_stats[feat_name]
        # Clip to [min, max] of training range
        return float(np.clip(value, stats["min"], stats["max"]))

    def predict(
        self,
        retrieval_margin: float,
        top1_score: float,
        top5_variance: float,
        evidence_support: float           = 0.0,
        evidence_contradiction: float     = 0.0,
        evidence_neutral: float           = 1.0,
        evidence_support_max: float       = 0.0,
        evidence_contradiction_max: float = 0.0,
        question_sim_mean: float          = 0.0,
        question_sim_max: float           = 0.0,
        cluster_agreement: float          = 0.5,
        cluster_entropy: float            = 1.0,
        historical_reliability: float     = 0.5,
        query_length: int                 = 10,
        escalation_count: int             = 0,
    ) -> float:

        # Clip corpus-sensitive features to training range
        retrieval_margin = self._normalise(retrieval_margin, "retrieval_margin")
        top5_variance    = self._normalise(top5_variance,    "top5_variance")

        features = np.array([[
            retrieval_margin,
            top1_score,
            top5_variance,
            evidence_support,
            evidence_support_max,
            evidence_contradiction,
            evidence_contradiction_max,
            evidence_neutral,
            question_sim_mean,
            question_sim_max,
            cluster_agreement,
            cluster_entropy,
            historical_reliability,
            query_length,
            escalation_count,
        ]])

        return float(self.model.predict_proba(features)[0][1])