from rag.format_answers import format_answer
from rag.pdf_answer import PDFAnswer
from rag.video_answer import VideoAnswer
from preprocessing.pdf.pdf_chunk import PDFChunk
from preprocessing.video.transcript_chunk import TranscriptChunk

def test_format_pdf_answer():
    pdf_chunk = PDFChunk(
        pdf_id="doc.pdf",
        page_number=2,
        paragraph_index=1,
        text="Registration instructions",
    )

    pdf = PDFAnswer(pdf_chunk)
    pdf.summary = 'save.'

    output = format_answer(
        question="How do I save?",
        video_answer=None,
        pdf_answer=pdf,
    )

    assert "doc.pdf" in output
    assert "2" in output
    assert "1" in output
    assert pdf.text in output
    assert pdf .summary in output


def test_format_video_answer():
    video_chunk = TranscriptChunk(
        video_id="vid1",
        start_token_id=1,
        end_token_id=4,
        start_timestamp=0.0,
        end_timestamp=2.0,
        text="click save",
    )
    video = VideoAnswer(video_chunk)
    video.refined_answer = "refined click save."

    output = format_answer(
        question="How do I save?",
        video_answer=video,
    )
    
    assert "How do I save?" in output
    assert "refined click save." in output
    assert "Video ID" in output
    assert "refined click save." in output

