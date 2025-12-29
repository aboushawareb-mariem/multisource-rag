import faiss
import numpy as np

class VectorIndex:
    """
    Lightweight wrapper around a FAISS vector index for similarity search.

    Uses inner product similarity on normalized vectors, which corresponds
    to cosine similarity. The index supports adding vectors and querying
    for the top-k most similar entries.
    """
     
    def __init__(self, dim: int):
        """
        Initializes the FAISS index.

        Args:
            dim: Dimensionality of the embedding vectors.
        """
        self.index = faiss.IndexFlatIP(dim)


    def add(self, vectors: np.ndarray):
        """
        Adds embedding vectors to the index.

        Args:
            vectors: 2D array of shape (n_vectors, dim) containing
                normalized embedding vectors.
        """
        self.index.add(vectors)

    def search(self, query_vector: np.ndarray, k: int):
        """
        Searches the index for the top-k most similar vectors.

        Args:
            query_vector: 1D embedding vector for the query.
            k: Number of nearest neighbors to retrieve.

        Returns:
            Tuple of (scores, indices) as returned by FAISS.
        """
        return self.index.search(query_vector.reshape(1, -1), k)
