from rag.retrievel import retrieve_from_videos

class DummyEmbedder:
    def embed_texts(self, texts):
        import numpy as np
        return np.random.rand(len(texts), 4)

    def embed_query(self, query):
        import numpy as np
        return np.random.rand(4)

def test_retrieve_from_videos_runs():
    videos = [
        {
            "video_id": "vid1",
            "video_transcripts": [
                {"id": 1, "timestamp": 0.0, "word": "click"},
                {"id": 2, "timestamp": 0.5, "word": "save"},
            ],
        }
    ]

    answer = retrieve_from_videos(
        question="How do I save?",
        videos=videos,
        embedder=DummyEmbedder(),
        threshold=0.0,
    )

    # Should not crash
    assert answer is not None
