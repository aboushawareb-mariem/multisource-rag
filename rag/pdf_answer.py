class PDFAnswer:
    """
    Represents an answer retrieved from a PDF source.

    Wraps a PDFChunk with additional fields used during answer refinement
    and presentation, such as an optional LLM-generated summary.
    """
    
    def __init__(self, chunk):
        self.pdf_id = chunk.pdf_id
        self.page_number = chunk.page_number
        self.paragraph_index = chunk.paragraph_index
        self.text = chunk.text
        self.summary = None