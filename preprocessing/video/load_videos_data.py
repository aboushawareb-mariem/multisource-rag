import json
from pathlib import Path
from typing import List

def load_video_transcripts(folder: str) -> List[dict]:
    """
    Loads video transcript data from JSON files in a directory.

    Each JSON file is expected to contain a video identifier and a list
    of transcript tokens. All valid files in the directory are loaded.

    Args:
        folder: Path to a directory containing video transcript JSON files.

    Returns:
        A list of dictionaries, each with the following keys:
            - video_id: Identifier of the video.
            - video_transcripts: Tokenized transcript data for the video.

    Notes:
        - All `.json` files in the directory are processed.
        - The function assumes a consistent JSON schema across files.
    """
    
    videos= []

    for path in Path(folder).glob("*.json"):
        with open(path, "r") as f:
            data = json.load(f)

        video_id = data["video_id"]
        video_transcripts = data["video_transcripts"]

        videos.append({
            "video_id": video_id,
            "video_transcripts": video_transcripts
        })

    return videos
