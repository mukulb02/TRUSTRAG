from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams
)

COLLECTION_NAME = "aurora_medical"

client = QdrantClient(
    host="localhost",
    port=6333
)


def create_collection():

    collections = client.get_collections()

    names = [
        c.name
        for c in collections.collections
    ]

    if COLLECTION_NAME not in names:

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=384,
                distance=Distance.COSINE
            )
        )

        print(
            f"[Qdrant] Created collection: {COLLECTION_NAME}"
        )

    else:

        print(
            f"[Qdrant] Collection already exists: {COLLECTION_NAME}"
        )


def get_client():
    return client