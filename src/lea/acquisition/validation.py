from pathlib import Path
from typing import Optional
from lea.exceptions import AcquisitionError
from lea.logging import logger

PDF_MAGIC_BYTES = b"%PDF-"

def validate_pdf_bytes(content: bytes, content_type: Optional[str] = None, max_bytes: int = 104857600) -> bool:
    if len(content) > max_bytes:
        raise AcquisitionError(f"PDF exceeds maximum byte limit of {max_bytes} bytes (got {len(content)})")

    if not content.startswith(PDF_MAGIC_BYTES):
        raise AcquisitionError("Downloaded payload does not start with valid %PDF- magic bytes (likely HTML paywall page)")

    if content_type and "application/pdf" not in content_type.lower() and "octet-stream" not in content_type.lower():
        logger.warning(f"Downloaded content-type '{content_type}' is not application/pdf, but magic bytes match.")

    return True

def validate_pdf_file(file_path: str, max_bytes: int = 104857600) -> bool:
    path = Path(file_path)
    if not path.exists():
        raise AcquisitionError(f"File not found: {file_path}")

    size = path.stat().st_size
    if size > max_bytes:
        raise AcquisitionError(f"PDF file size {size} exceeds max limit {max_bytes}")

    with open(path, "rb") as f:
        header = f.read(5)
        if header != PDF_MAGIC_BYTES:
            raise AcquisitionError("PDF file does not start with %PDF- header")

    return True
