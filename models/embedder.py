from sentence_transformers import SentenceTransformer
import numpy as np
import os
from dotenv import load_dotenv
import logging

class Embedder:
    """
    Wrapper around a SentenceTransformer model for generating text embeddings.

    Loads the embedding model from an environment variable or a default,
    and provides helper methods for embedding documents and queries.
    """

    def __init__(self, model_name: str | None = None):
        if model_name is None:
            model_name = os.getenv(
                "EMBEDDING_MODEL_NAME",
                "all-MiniLM-L6-v2",
            )
        
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            logging.error("Failed to load embedding model %s", model_name)
            raise

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Embeds a list of texts into normalized vectors.
        """
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embeds a single query string into a normalized vector.
        """
        return self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )