from preprocessing.video.video_chunking import chunk_video_transcript

def test_video_chunking_basic():
    tokens = [
        {"id": 1, "timestamp": 0.0, "word": "hello"},
        {"id": 2, "timestamp": 0.5, "word": "world"},
        {"id": 3, "timestamp": 1.0, "word": "click"},
        {"id": 4, "timestamp": 1.5, "word": "save"},
    ]

    chunks = chunk_video_transcript(
        video_id="vid1",
        tokens=tokens,
        chunk_size=2,
        overlap=0,
    )

    # Expect overlapping chunks:
    # [hello world], [click save]
    assert len(chunks) == 2
    assert chunks[0].video_id == "vid1"
    assert "hello world" in chunks[0].text


def test_video_chunking_with_overlap():
    tokens = [
        {"id": 1, "timestamp": 0.0, "word": "hello"},
        {"id": 2, "timestamp": 0.5, "word": "world"},
        {"id": 3, "timestamp": 1.0, "word": "click"},
        {"id": 4, "timestamp": 1.5, "word": "save"},
    ]

    chunks = chunk_video_transcript(
        video_id="vid1",
        tokens=tokens,
        chunk_size=2,
        overlap=1,
    )
    print(f'chunks are')
    for chunk in chunks:
        print(chunk.text)

    # Expect overlapping chunks:
    # [hello world], [world click], [click save], [save]
    assert len(chunks) == 4
    assert chunks[0].text == "hello world"
    assert chunks[1].text == "world click"
    assert chunks[2].text == "click save"

