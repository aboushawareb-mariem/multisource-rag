import numpy as np
from indexing.faiss.vector_index import VectorIndex

def test_vector_index_search():
    index = VectorIndex(dim=2)

    vectors = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    index.add(vectors)

    query = np.array([1.0, 0.0])
    scores, indices = index.search(query, k=1)

    assert indices[0][0] == 0
