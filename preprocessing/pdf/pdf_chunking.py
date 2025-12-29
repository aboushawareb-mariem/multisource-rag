from preprocessing.pdf.pdf_chunk import PDFChunk

def chunk_pdf_pages(
    pdf: dict,
    max_chars: int = 800,
) -> list[PDFChunk]:
    """
    Splits a PDF document into paragraph-level chunks for retrieval.

    Iterates through each page of the PDF, splits the page text into
    paragraphs, and creates a PDFChunk for each paragraph.

    Args:
        pdf: Dictionary containing the PDF identifier and extracted page texts.
        max_chars: Maximum character length per chunk (currently unused).

    Returns:
        A list of PDFChunk objects representing paragraph-level chunks
        in page order.

    Notes:
        - Page numbers are 1-based and reflect document order.
        - Paragraphs are determined by double newline separation.
    """
    
    chunks = []

    for page_idx, page_text in enumerate(pdf["pages"]):
        paragraphs = [
            p.strip()
            for p in page_text.split("\n\n")
            if p.strip()
        ]

        for para_idx, para in enumerate(paragraphs):
            chunks.append(
                PDFChunk(
                    pdf_id=pdf["pdf_id"],
                    page_number=page_idx + 1,      # page numbering is 1-based
                    paragraph_index=para_idx,
                    text=para,
                )
            )

    return chunks

