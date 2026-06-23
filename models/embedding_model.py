from sentence_transformers import SentenceTransformer
import torch

MODEL_NAME = "aleynahukmet/bge-medical-small-en-v1.5"

device = "cuda" if torch.cuda.is_available() else "cpu"

model = SentenceTransformer(
    MODEL_NAME,
    device=device
)


def embed(text: str) -> list:
    """Embed a single string. Returns a Python list (for Qdrant)."""
    return model.encode(
        text,
        normalize_embeddings=True
    ).tolist()


def embed_batch(texts: list) -> list:
    """Embed a list of strings. Returns a numpy array."""
    return model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=True
    )