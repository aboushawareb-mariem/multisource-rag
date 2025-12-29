class PDFChunk:
    """
    Represents a chunk of text extracted from a PDF document.

    Each chunk corresponds to a paragraph within a page and is
    identified by the source PDF, page number (1-based), and
    paragraph index.
    """
    def __init__(self, pdf_id, page_number, paragraph_index, text):
        self.pdf_id = pdf_id
        self.page_number = page_number
        self.paragraph_index = paragraph_index
        self.text = text