import numpy as np


def _get_cluster_id(doc: dict):
    """
    Safely extract cluster_id from a document dict.

    After RRF fusion + reranking, docs are flat dicts:
        {"text": ..., "fusion": ..., "rerank": ..., "cluster_id": ...}

    cluster_id is carried forward by retrieval_agent from Qdrant payloads.
    Returns None if missing.
    """
    return doc.get("cluster_id", None)


def cluster_agreement(docs: list) -> float:
    """
    Fraction of top docs that share the majority cluster.
    Returns 0.5 if no cluster info is present.
    """
    clusters = [
        _get_cluster_id(d)
        for d in docs
        if _get_cluster_id(d) is not None
    ]

    if len(clusters) == 0:
        return 0.5

    majority = max(set(clusters), key=clusters.count)

    return clusters.count(majority) / len(clusters)


def cluster_entropy(docs: list) -> float:
    """
    Shannon entropy of cluster distribution across top docs.
    Returns 1.0 if no cluster info is present (maximum uncertainty).
    """
    clusters = [
        _get_cluster_id(d)
        for d in docs
        if _get_cluster_id(d) is not None
    ]

    if len(clusters) == 0:
        return 1.0

    unique, counts = np.unique(clusters, return_counts=True)

    probs = counts / counts.sum()

    entropy = -np.sum(probs * np.log2(probs + 1e-12))

    return float(entropy)