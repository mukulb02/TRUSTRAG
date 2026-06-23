"""
Indexes mukulb/combined_medical_corpus into Qdrant.
Each point stores: text, group_name, cluster_id.

Run once before starting the API:
    cd AURORA && python index_corpus.py
"""

from datasets import load_dataset
from sklearn.cluster import MiniBatchKMeans
import numpy as np

from models.embedding_model import embed_batch
from models.qdrant_store import create_collection, get_client, COLLECTION_NAME

# ----------------------------------
# Setup
# ----------------------------------

create_collection()
client = get_client()

# ----------------------------------
# Load corpus
# ----------------------------------

print("Loading dataset...")

dataset = load_dataset(
    "mukulb/combined_medical_corpus",
    split="train"
)

texts  = dataset["text"]
groups = dataset["group_name"]

print(f"Corpus size: {len(texts)}")

# ----------------------------------
# Embeddings
# ----------------------------------

print("Generating embeddings...")

vectors = embed_batch(list(texts))   # returns numpy array

# ----------------------------------
# Clustering (32 clusters)
# ----------------------------------

print("Clustering...")

kmeans = MiniBatchKMeans(
    n_clusters=32,
    random_state=42,
    batch_size=2048
)

cluster_ids = kmeans.fit_predict(vectors)

print("Cluster distribution:")
unique, counts = np.unique(cluster_ids, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  Cluster {u:2d}: {c} docs")

# ----------------------------------
# Upload to Qdrant
# ----------------------------------

print("Uploading to Qdrant...")

BATCH_SIZE = 1000

points = []
for idx in range(len(texts)):
    points.append({
        "id":     idx,
        "vector": vectors[idx].tolist(),
        "payload": {
            "text":       texts[idx],
            "group_name": groups[idx],
            "cluster_id": int(cluster_ids[idx])
        }
    })

for i in range(0, len(points), BATCH_SIZE):
    batch = points[i: i + BATCH_SIZE]
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=batch
    )
    uploaded = min(i + BATCH_SIZE, len(points))
    print(f"  Uploaded {uploaded}/{len(points)}")

print("\nIndexing complete.")