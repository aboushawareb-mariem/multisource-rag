from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import logging

from preprocessing.video.load_videos_data import load_video_transcripts
from preprocessing.pdf.load_pdfs_data import load_pdf_collection
from models.embedder import Embedder
from models.gemini_llm_client import GeminiLLMClient
from rag.answer_refiner import AnswerRefiner
from rag.retrievel import retrieve_from_videos, retrieve_from_pdfs
from rag.format_answers import format_answer
import os
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
app = FastAPI(title="RAG QA API")

class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str

@app.on_event("startup")
def startup():
    """
    Initializes the RAG system at application startup.

    Loads video transcripts and PDF documents, prepares embeddings,
    and builds vector indexes used for retrieval during API requests.

    Notes:
        - Data sources are loaded once at startup.
        - Adding new files requires restarting the service or re-indexing.
    """

    global embedder, llm_client, answer_refiner, videos, pdfs

    logging.info("Loading resources...")

    embedder = Embedder()
    llm_client = GeminiLLMClient()
    answer_refiner = AnswerRefiner(llm_client)

    videos = load_video_transcripts(os.getenv("VIDEO_SOURCE_PATH"))
    pdfs = load_pdf_collection(os.getenv("PDF_SOURCE_PATH"))

    logging.info("Startup completed")

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """
    Answers a user question using the RAG pipeline.

    The endpoint retrieves relevant information from video transcripts first.
    If no sufficiently similar video result is found, it falls back to PDF
    documents. The retrieved answer may optionally be refined using an LLM.

    Args:
        req: Request body containing the user question.

    Returns:
        A formatted answer including the source (video or PDF) and the
        refined response, or a message indicating that no relevant answer
        was found.
    """
    
    question = req.question

    video_threshold = float(os.getenv("VIDEO_SIMILARITY_THRESHOLD", 0.7))
    pdf_threshold = float(os.getenv("PDF_SIMILARITY_THRESHOLD", 0.7))

    video_answer = retrieve_from_videos(
        question, videos, embedder, video_threshold
    )

    pdf_answer = None
    if not video_answer:
        pdf_answer = retrieve_from_pdfs(
            question, pdfs, embedder, pdf_threshold
        )

    if not video_answer and not pdf_answer:
        raise HTTPException(
            status_code=404,
            detail="No relevant answer found",
        )

    video_answer, pdf_answer = answer_refiner.refine_answer(
        question,
        video_answer,
        pdf_answer,
    )

    formatted = format_answer(
        question=question,
        video_answer=video_answer,
        pdf_answer= pdf_answer
    )
    logging.info("Video answer: %s", bool(video_answer))
    logging.info("PDF answer: %s", bool(pdf_answer))

    return AskResponse(answer=formatted)



