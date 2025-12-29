class VideoAnswer:
    """
    Represents an answer retrieved from a video transcript.

    Wraps a TranscriptChunk with additional fields used during answer
    refinement and presentation, such as an optional LLM-refined response.
    """
    
    def __init__(self, chunk):
        self.video_id = chunk.video_id
        self.start_timestamp = chunk.start_timestamp
        self.start_token_id = chunk.start_token_id
        self.end_timestamp = chunk.end_timestamp
        self.end_token_id = chunk.end_token_id
        self.transcript_snippet = chunk.text
        self.refined_answer = None