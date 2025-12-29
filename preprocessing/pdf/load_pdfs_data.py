from pathlib import Path
from preprocessing.pdf.pdf_chunking import chunk_pdf_pages


import fitz  # PyMuPDF

def load_pdf_pages(pdf_path: str) -> list[str]:
    """
    Load and extract text from all pages of a PDF file.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        list[str]: A list where each element contains the extracted text
        of a single page, in page order.
    """
    doc = fitz.open(pdf_path)
    pages = [page.get_text() for page in doc]
    return pages

def load_pdf_collection(pdf_dir: str) -> list:
    """
    Load all PDF files in a directory and extract their page contents.

    Each PDF is represented as a dictionary containing its filename
    and a list of extracted page texts.

    Args:
        pdf_dir (str): Path to a directory containing PDF files.

    Returns:
        list: A list of dictionaries with the following keys:
            - pdf_id (str): Filename of the PDF.
            - pages (list[str]): Extracted text for each page.

    Notes:
        - All `.pdf` files in the directory are loaded.
        - Page numbering is implicit and based on its order.
    """
    pdfs = []

    for pdf_path in Path(pdf_dir).glob("*.pdf"):
        pages = load_pdf_pages(str(pdf_path))

        pdfs.append({
            "pdf_id": pdf_path.name,
            "pages": pages,
        })

    return pdfs
