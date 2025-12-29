import argparse

from preprocessing.video.load_videos_data import load_video_transcripts
from preprocessing.video.video_chunking import chunk_video_transcript
from models.embedder import Embedder
from models.llm_client import LLMClient
from models.gemini_llm_client import GeminiLLMClient
from indexing.faiss.vector_index import VectorIndex
from rag.answer_refiner import AnswerRefiner
from preprocessing.pdf.load_pdfs_data import load_pdf_collection
from preprocessing.pdf.pdf_chunking import chunk_pdf_pages
from rag.pdf_answer import PDFAnswer
from rag.video_answer import VideoAnswer
from rag.retrievel import retrieve_from_videos, retrieve_from_pdfs
from rag.format_answers import format_answer
import os
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

# Silence noisy third-party libraries
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("google").setLevel(logging.WARNING)
logging.getLogger("google_genai").setLevel(logging.WARNING)

def main():
    video_answer = None
    pdf_answer = None
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True, type=str)
    args = parser.parse_args()

    ######## VIDEO SOURCE ########
    videos_path = os.getenv('VIDEO_SOURCE_PATH')
    video_similarity_threshold = float(os.getenv('VIDEO_SIMILARITY_THRESHOLD'))

    # load videos -> [{video_id, video_transcripts:[ {token_id: , timestamp: ...} ]}]
    videos = load_video_transcripts(videos_path)
    logging.info(f"Loaded {len(videos)} videos")

    # Initialize embedder
    embedder = Embedder()
    video_answer = retrieve_from_videos(args.question, videos, embedder, video_similarity_threshold)
    
    ######## PDF SOURCE ########
    if not video_answer:
        pdfs_path = os.getenv('PDF_SOURCE_PATH')
        pdfs = load_pdf_collection(pdfs_path)
        pdf_similarity_threshold = float(os.getenv('PDF_SIMILARITY_THRESHOLD'))
        pdf_answer = retrieve_from_pdfs(args.question, pdfs, embedder, pdf_similarity_threshold)


    
    llm_client = GeminiLLMClient()
    answer_refiner = AnswerRefiner(llm_client)
    video_answer, pdf_answer = answer_refiner.refine_answer(args.question, video_answer, pdf_answer)

    formatted_answer = format_answer(args.question, video_answer, pdf_answer)
    print(formatted_answer)

#TODO add logs + docstrings + tests
if __name__ == "__main__":
    main()
