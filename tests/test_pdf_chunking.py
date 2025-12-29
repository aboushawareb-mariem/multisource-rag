from preprocessing.pdf.pdf_chunking import chunk_pdf_pages

def test_pdf_chunking_basic():
    pdf = {
        "pdf_id": "test.pdf",
        "pages": [
            "First paragraph.\n\nSecond paragraph.",
        ],
    }

    chunks = chunk_pdf_pages(pdf)

    assert len(chunks) == 2
    assert chunks[0].pdf_id == "test.pdf"
    assert chunks[0].page_number == 1
