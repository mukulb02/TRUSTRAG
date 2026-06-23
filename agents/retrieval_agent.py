from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

import numpy as np
from collections import deque

from models.embedding_model import embed
from models.qdrant_store import get_client, COLLECTION_NAME
from utils.cluster_utils import cluster_entropy


class RetrievalAgent:

    def __init__(self, corpus: list):

        self.client = get_client()
        self.corpus = corpus

        print("[RetrievalAgent] Building BM25 index...")

        self.tokenized_corpus = [
            doc.lower().split()
            for doc in corpus
        ]

        self.bm25 = BM25Okapi(self.tokenized_corpus)

        self.cross_encoder = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

        self.margin_history = deque(maxlen=200)

        print("[RetrievalAgent] Ready.")

    # ----------------------------------
    # Query profiling
    # ----------------------------------

    def length_bucket(self, query: str) -> str:
        n = len(query.split())
        if n <= 8:
            return "S"
        elif n <= 20:
            return "M"
        elif n <= 64:
            return "L"
        return "XL"

    def noise_score(self, text: str) -> float:
        upper = sum(c.isupper() for c in text)
        digits = sum(c.isdigit() for c in text)
        return (upper + digits) / max(1, len(text))

    # ----------------------------------
    # Cluster routing
    # Finds the nearest clusters to the query vector
    # by querying Qdrant without a filter and reading
    # which cluster_ids appear most in the top results.
    # ----------------------------------

    def cluster_route(
        self,
        query_vector: list,
        top_clusters: int = 2
    ) -> list:
        """
        Returns the top-k most relevant cluster IDs
        by inspecting the nearest neighbours' cluster payloads.
        """
        response = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=20
        )

        cluster_ids = [
            pt.payload.get("cluster_id")
            for pt in response.points
            if pt.payload.get("cluster_id") is not None
        ]

        if not cluster_ids:
            return []

        # Return most frequent cluster IDs
        unique, counts = np.unique(cluster_ids, return_counts=True)
        sorted_idx = np.argsort(counts)[::-1]
        return list(unique[sorted_idx[:top_clusters]])

    # ----------------------------------
    # Dense Retrieval
    # Returns flat dicts with text + cluster_id preserved
    # ----------------------------------

    def dense_search(
        self,
        query_vector: list,
        limit: int = 30,
        cluster_ids: list = None
    ) -> list:
        """
        Queries Qdrant. If cluster_ids given, runs one query per
        cluster and merges. Returns list of flat dicts.
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        results = []

        if cluster_ids:
            per_cluster = max(1, limit // len(cluster_ids))
            for cid in cluster_ids:
                response = self.client.query_points(
                    collection_name=COLLECTION_NAME,
                    query=query_vector,
                    query_filter=Filter(
                        must=[
                            FieldCondition(
                                key="cluster_id",
                                match=MatchValue(value=int(cid))
                            )
                        ]
                    ),
                    limit=per_cluster
                )
                results.extend(response.points)
        else:
            response = self.client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=limit
            )
            results = response.points

        # Convert Qdrant ScoredPoint → flat dict (carry cluster_id!)
        docs = []
        for pt in results:
            docs.append({
                "text": pt.payload.get("text", ""),
                "cluster_id": pt.payload.get("cluster_id"),
                "group_name": pt.payload.get("group_name", ""),
                "dense_score": pt.score
            })

        return docs

    # ----------------------------------
    # Sparse Retrieval
    # ----------------------------------

    def sparse_search(self, query: str, k: int = 30) -> list:
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        idx = np.argsort(scores)[::-1][:k]

        docs = []
        for i in idx:
            docs.append({
                "text": self.corpus[i],
                "cluster_id": None,   # BM25 has no cluster info
                "bm25_score": float(scores[i])
            })

        return docs

    # ----------------------------------
    # RRF Fusion
    # Merges dense + sparse by reciprocal rank.
    # Preserves cluster_id from dense docs where available.
    # ----------------------------------

    def reciprocal_rank_fusion(
        self,
        dense_docs: list,
        sparse_docs: list,
        k: int = 60
    ) -> list:

        fused = {}   # text → {"fusion": float, "cluster_id": ...}

        for rank, d in enumerate(dense_docs):
            key = d["text"]
            if key not in fused:
                fused[key] = {
                    "text": key,
                    "fusion": 0.0,
                    "cluster_id": d.get("cluster_id")
                }
            fused[key]["fusion"] += 1 / (k + rank)

        for rank, d in enumerate(sparse_docs):
            key = d["text"]
            if key not in fused:
                fused[key] = {
                    "text": key,
                    "fusion": 0.0,
                    "cluster_id": d.get("cluster_id")   # usually None
                }
            fused[key]["fusion"] += 1 / (k + rank)

        merged = sorted(
            fused.values(),
            key=lambda x: x["fusion"],
            reverse=True
        )

        return merged

    # ----------------------------------
    # Cross-Encoder Reranking
    # ----------------------------------

    def rerank(self, query: str, docs: list) -> list:
        pairs = [(query, d["text"]) for d in docs]
        scores = self.cross_encoder.predict(pairs)

        for doc, score in zip(docs, scores):
            doc["rerank"] = float(score)

        docs.sort(key=lambda x: x["rerank"], reverse=True)
        return docs

    # ----------------------------------
    # Margin confidence
    # ----------------------------------

    def margin(self, docs: list) -> float:
        if len(docs) < 2:
            return 1.0
        return max(0.0, docs[0]["rerank"] - docs[1]["rerank"])

    # ----------------------------------
    # Escalation limits
    # ----------------------------------

    def escalation_limits(self, bucket: str) -> tuple:
        return {
            "S":  (40, 40),
            "M":  (50, 50),
            "L":  (60, 60),
            "XL": (80, 80),
        }.get(bucket, (60, 60))

    # ----------------------------------
    # Main Retrieval Pipeline
    # ----------------------------------

    def retrieve(self, query: str) -> dict:

        bucket = self.length_bucket(query)
        noise = self.noise_score(query)
        query_vector = embed(query)

        # Cluster routing for long queries
        cluster_ids = []
        if bucket in ("L", "XL"):
            cluster_ids = self.cluster_route(query_vector)

        # First-pass retrieval
        dense_docs = self.dense_search(
            query_vector,
            limit=30,
            cluster_ids=cluster_ids if cluster_ids else None
        )
        sparse_docs = self.sparse_search(query, k=30)

        docs = self.reciprocal_rank_fusion(dense_docs, sparse_docs)
        docs = self.rerank(query, docs[:50])

        margin_val = self.margin(docs)
        self.margin_history.append(margin_val)

        # Escalation: low-margin → expand retrieval
        if margin_val < 0.20:
            k_dense, k_sparse = self.escalation_limits(bucket)

            dense_docs = self.dense_search(query_vector, limit=k_dense)
            sparse_docs = self.sparse_search(query, k=k_sparse)

            docs = self.reciprocal_rank_fusion(dense_docs, sparse_docs)
            docs = self.rerank(query, docs[:80])

            margin_val = self.margin(docs)

        top5 = docs[:5]

        return {
            "docs":            top5,
            "margin":          margin_val,
            "bucket":          bucket,
            "noise":           noise,
            "cluster_entropy": cluster_entropy(top5),
            "top1_score":      top5[0]["rerank"] if top5 else 0.0,
            "top5_variance":   float(np.var([d["rerank"] for d in top5])),
        }