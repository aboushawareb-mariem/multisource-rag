from preprocessing.video.transcript_chunk import TranscriptChunk

def get_chunk_text(chunk_tokens_list: list[dict]) -> str:
   """
    Reconstructs text for a transcript chunk from token data.

    Concatenates the `word` field of each token in order to form
    a readable text snippet.

    Args:
        chunk_tokens_list: List of token dictionaries containing a `word` field.

    Returns:
        Reconstructed text for the chunk.
    """
   
   text = ' '.join(token['word'] for token in chunk_tokens_list)
   return text

def chunk_video_transcript(video_id: str, tokens: list[dict], chunk_size: int = 20, overlap: int = 2) -> list[TranscriptChunk]:
    """
    Splits a video transcript into overlapping token-based chunks.

    Uses a sliding window approach to create transcript chunks that
    preserve temporal context via token IDs and timestamps.

    Args:
        video_id: Identifier of the source video.
        tokens: Tokenized transcript data containing token IDs, timestamps, and words.
        chunk_size: Number of tokens per chunk.
        overlap: Number of overlapping tokens between consecutive chunks.

    Returns:
        A list of TranscriptChunk objects in chronological order.

    Notes:
        - Chunks are created using a sliding window with overlap.
        - The final chunk may contain fewer than `chunk_size` tokens.
        - Token IDs and timestamps correspond to the original transcript.
    """
        
    all_chunks = []
    step = chunk_size - overlap

    
    for i in range(0, len(tokens), step):
        chunk_tokens = tokens[i : i + chunk_size] # [{}, {}, {}]
        start_chunk = chunk_tokens[0]
        end_chunk = chunk_tokens[-1]
        start_token_id = start_chunk['id']
        end_token_id = end_chunk['id']
        start_timestamp = start_chunk['timestamp']
        end_timestamp = end_chunk['timestamp']
        text = get_chunk_text(chunk_tokens)
        tc = TranscriptChunk(video_id,
                             start_token_id,
                             end_token_id,
                             start_timestamp,
                             end_timestamp,
                             text)
        all_chunks.append(tc)
        
    return all_chunks



