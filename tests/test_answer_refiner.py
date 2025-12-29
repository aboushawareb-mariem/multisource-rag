from rag.answer_refiner import AnswerRefiner
from rag.video_answer import VideoAnswer
from preprocessing.video.transcript_chunk import TranscriptChunk
from rag.pdf_answer import PDFAnswer
from rag.video_answer import VideoAnswer
from rag.pdf_answer import PDFAnswer
from preprocessing.pdf.pdf_chunk import PDFChunk

class DummyLLMClient:
    def generate(self, prompt: str) -> str:
        return "Refined output"

def test_answer_refiner_with_video():
    llm = DummyLLMClient()
    refiner = AnswerRefiner(llm)
    video_chunk = TranscriptChunk(
        video_id="vid1",
        start_token_id=1,
        end_token_id=4,
        start_timestamp=0.0,
        end_timestamp=2.0,
        text = 'raw output'
    )

    video = VideoAnswer(video_chunk)

    refined_video, refined_pdf = refiner.refine_answer(
        question="How do I save?",
        video_answer=video,
        pdf_answer=None,
    )

    assert refined_video is not None
    assert refined_video.refined_answer == "Refined output"
    assert refined_pdf is None


    

def test_answer_refiner_with_pdf():
    llm = DummyLLMClient()
    refiner = AnswerRefiner(llm)

    pdf_chunk = PDFChunk(
        pdf_id="doc.pdf",
        page_number=2,
        paragraph_index=1,
        text="Registration instructions",

    )

    pdf = PDFAnswer(pdf_chunk)

    refined_video, refined_pdf = refiner.refine_answer(
        question="How do I register?",
        video_answer=None,
        pdf_answer=pdf,
    )

    assert refined_pdf.summary == "Refined output"
    assert refined_video is None

