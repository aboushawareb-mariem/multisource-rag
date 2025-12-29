class TranscriptChunk:
    """
    Represents a contiguous chunk of a video transcript.

    A transcript chunk corresponds to a sequence of tokens within a video,
    annotated with token indices and timestamps. Chunks are used as the
    retrieval units for video-based semantic search.
    """
    def __init__(self, video_id, start_token_id, end_token_id, start_timestamp, end_timestamp, text):
        self.video_id = video_id
        self.start_token_id = start_token_id
        self.end_token_id = end_token_id
        self.start_timestamp = start_timestamp
        self.end_timestamp = end_timestamp
        self.text = text
    