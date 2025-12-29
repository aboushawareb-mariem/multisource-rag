from rag.video_answer import VideoAnswer

class RAGSystem:
    """
    Core retrieval component of the RAG pipeline.

    This class performs semantic retrieval over a collection of embedded
    chunks using a vector index. It is responsible only for similarity
    search and does not perform answer formatting or refinement.
    """

    def __init__(self, chunks, embedder, index):
        self.chunks = chunks
        self.embedder = embedder
        self.index = index
    
    def answer(self, question: str)  -> tuple[float, int]:
        """
        Retrieves the most relevant chunk for a given question.

        The question is embedded and compared against indexed chunk
        embeddings using the vector index.

        Args:
            question: User query string.

        Returns:
            A tuple containing:
                - best_score: Similarity score of the top matching chunk.
                - best_idx: Index of the matching chunk in `self.chunks`.
        """
        q_vec = self.embedder.embed_query(question)
        scores, indices = self.index.search(q_vec, k=3)

        best_score = scores[0][0]
        best_idx = indices[0][0]
        return best_score, best_idx
