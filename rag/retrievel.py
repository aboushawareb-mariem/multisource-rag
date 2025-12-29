from preprocessing.video.video_chunking import chunk_video_transcript
from preprocessing.pdf.pdf_chunking import chunk_pdf_pages
from indexing.faiss.vector_index import VectorIndex
from rag.rag_system import RAGSystem
from rag.video_answer import VideoAnswer
from rag.pdf_answer import PDFAnswer
import logging


def retrieve_from_videos(question, videos, embedder, threshold):
    """
    Retrieves the most relevant video-based answer for a given question.

    This function:
    - Chunks all video transcripts
    - Embeds the resulting chunks
    - Builds a vector index
    - Performs semantic retrieval via RagSystem
    - Applies a similarity threshold to accept or reject the result

    Args:
        question: User query string.
        videos: List of video transcript dictionaries.
        embedder: Embedder used to generate text embeddings.
        threshold: Minimum similarity score required to accept a result.

    Returns:
        A VideoAnswer object if a chunk exceeds the similarity threshold,
        otherwise None.
    """

    all_chunks = []

    for video in videos:
        chunks = chunk_video_transcript(
            video_id=video['video_id'],
            tokens=video['video_transcripts'],
        )
        all_chunks.extend(chunks)
    
    logging.info(f"chunked videos into {len(all_chunks)} chunks")
    
    vectors = embedder.embed_texts([chunk.text for chunk in all_chunks])
    index = VectorIndex(dim=vectors.shape[1])
    index.add(vectors)

    rag = RAGSystem(
        chunks=all_chunks,
        embedder=embedder,
        index=index,
    )

    best_score, best_idx = rag.answer(question)
    
    if( best_score is not None and best_score > threshold):
        best_chunk = all_chunks[best_idx]
        logging.info(f"best video chunk retrieved with score: {best_score}. Acceptance threshold: {threshold}")
        return VideoAnswer(best_chunk)
    
    logging.info(f"No video chunk exceeded threshold. Best score: {best_score}, threshold: {threshold}")
    return None

def retrieve_from_pdfs(question, pdfs, embedder, threshold):
    """
    Retrieves the most relevant PDF-based answer for a given question.

    This function:
    - Chunks all PDF documents into paragraph-level chunks
    - Embeds the chunks
    - Builds a vector index
    - Performs semantic retrieval
    - Applies a similarity threshold to accept or reject the result

    Args:
        question: User query string.
        pdfs: List of PDF document dictionaries.
        embedder: Embedder used to generate text embeddings.
        threshold: Minimum similarity score required to accept a result.

    Returns:
        A PDFAnswer object if a chunk exceeds the similarity threshold,
        otherwise None.
    """
    
    all_chunks = []

    for pdf in pdfs:
        chunks = chunk_pdf_pages(pdf)
        all_chunks.extend(chunks)
    
    logging.info(f"chunked pdfs into {len(all_chunks)} chunks")

    texts = [chunk.text for chunk in all_chunks]
    vectors = embedder.embed_texts(texts)
    pdf_vector_index = VectorIndex(dim=vectors.shape[1])
    pdf_vector_index.add(vectors)
    pdf_rag = RAGSystem(
        chunks=all_chunks,
        embedder=embedder,
        index=pdf_vector_index,
    )
    best_score, best_idx = pdf_rag.answer(question)
    
    if(best_score is not None and best_score > threshold):
        best_chunk = all_chunks[best_idx]
        logging.info(f"best pdf chunk retrieved with score: {best_score}. Acceptance threshold: {threshold}")
        return PDFAnswer(best_chunk)
    
    logging.info(f"No pdf chunk exceeded threshold. Best score: {best_score}, threshold: {threshold}")
    return None
