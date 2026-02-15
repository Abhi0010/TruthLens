"""Extract text from uploaded documents (PDF, DOCX) for fact-checking."""

from typing import Tuple

# Max chars to extract (pipeline uses similar limits)
MAX_EXTRACT_CHARS = 50_000


def extract_text_from_file(uploaded_file) -> Tuple[str, str]:
    """
    Extract plain text from an uploaded file (PDF or DOCX).

    Args:
        uploaded_file: Streamlit UploadedFile or file-like with .name and .read().

    Returns:
        (extracted_text, display_name) where display_name is filename or "Uploaded document".

    Raises:
        ValueError: Unsupported type or extraction failed.
    """
    name = getattr(uploaded_file, "name", "") or "Uploaded document"
    raw = uploaded_file.read()
    if not raw:
        raise ValueError("File is empty.")

    lower = name.lower()
    if lower.endswith(".pdf"):
        return _extract_pdf(raw, name)
    if lower.endswith(".docx"):
        return _extract_docx(raw, name)
    raise ValueError("Unsupported format. Use PDF or DOCX.")


def _extract_pdf(raw: bytes, name: str) -> Tuple[str, str]:
    """Extract text from PDF bytes."""
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ValueError("PDF support requires the pypdf package. Install with: pip install pypdf")

    try:
        from io import BytesIO
        reader = PdfReader(BytesIO(raw))
        parts = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                parts.append(t)
        text = "\n".join(parts).strip()
        if not text or len(text) < 20:
            raise ValueError("Could not extract enough text from this PDF.")
        return (text[:MAX_EXTRACT_CHARS], name)
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise ValueError(f"Failed to read PDF: {e}") from e


def _extract_docx(raw: bytes, name: str) -> Tuple[str, str]:
    """Extract text from DOCX bytes."""
    try:
        from docx import Document
    except ImportError:
        raise ValueError("DOCX support requires python-docx. Install with: pip install python-docx")

    try:
        from io import BytesIO
        doc = Document(BytesIO(raw))
        parts = [p.text for p in doc.paragraphs if p.text.strip()]
        text = "\n".join(parts).strip()
        if not text or len(text) < 20:
            raise ValueError("Could not extract enough text from this document.")
        return (text[:MAX_EXTRACT_CHARS], name)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to read document: {e}") from e
