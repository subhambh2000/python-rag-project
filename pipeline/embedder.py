import logging
import os

import torch
from sentence_transformers import SentenceTransformer

MODEL_NAME = os.getenv("EMBEDDING_MODEL", default="all-MiniLM-L6-v2")

device = "cuda" if torch.cuda.is_available() else "cpu"
_model = SentenceTransformer(MODEL_NAME, device=device, trust_remote_code=True)


def embed(texts: list[str]) -> list[list[float]]:
    logging.debug(f"Embedding chunks using model: {MODEL_NAME}")
    # validate first
    for i, text in enumerate(texts):
        if not isinstance(text, str):
            raise ValueError(f"Expected string at index {i}, got {type(text)}")

    encoded_text = _model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return encoded_text.tolist()


def embed_one(text: str) -> list[float]:
    result = embed([text])
    return result[0]


if __name__ == "__main__":
    logging.debug(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    logging.debug(f"GPU: {torch.cuda.get_device_name(0)}")

    test_texts = ["large cap", "equity hybrid", "small cap"]
    embeddings = embed(test_texts)
    print(len(embeddings))
    print(embeddings[0][:5])  # Print first 5 dimensions of the first embedding
