import re
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional
import fitz  # PyMuPDF
from lea.exceptions import IngestError
from lea.logging import logger

DOI_REGEX = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")
ARXIV_REGEX = re.compile(r"(?:arXiv:\s*|arxiv/|abs/)?(\d{4}\.\d{4,5}(?:v\d+)?)", re.IGNORECASE)

class PDFParser:
    def __init__(self, pdf_path: str):
        self.path = Path(pdf_path)
        if not self.path.exists():
            raise IngestError(f"PDF file does not exist: {pdf_path}")

    def compute_sha256(self) -> str:
        sha256 = hashlib.sha256()
        with open(self.path, "rb") as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()

    def extract_text(self) -> str:
        try:
            doc = fitz.open(self.path)
            full_text = []
            for page in doc:
                full_text.append(page.get_text())
            doc.close()
            full_str = "\n".join(full_text)
            return full_str.replace("\x00", "").replace("\u0000", "")
        except Exception as exc:
            raise IngestError(f"Failed to extract text using PyMuPDF: {exc}")

    def extract_body_text(self) -> str:
        text = self.extract_text()
        lines = text.splitlines()
        halfway = len(lines) // 2
        ref_pattern = re.compile(
            r"^\s*(?:[0-9.]+\s*)?(?:references|ref\s*er\s*en\s*ces|bibliography|works cited|data availability|code availability|supplementary information|author contributions|competing interests)\s*$",
            re.IGNORECASE
        )

        for i in range(halfway, len(lines)):
            if ref_pattern.match(lines[i].strip()):
                logger.info(f"Stripped back-matter section ('{lines[i].strip()}') starting at line {i}/{len(lines)}")
                return "\n".join(lines[:i])
        return text

    def parse_fallback_metadata(self) -> Dict[str, Any]:
        text = self.extract_text()
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        title = lines[0] if lines else self.path.stem
        # RegEx matching for DOI and ArXiv
        doi_matches = DOI_REGEX.findall(text)
        doi = doi_matches[0].rstrip(".,;") if doi_matches else None

        arxiv_matches = ARXIV_REGEX.findall(text)
        arxiv_id = arxiv_matches[0] if arxiv_matches else None

        return {
            "title": title,
            "abstract": text[:1000],  # Fallback preview
            "authors": [],
            "doi": doi,
            "arxiv_id": arxiv_id,
            "publication_year": None,
            "full_text": text
        }

    def extract_regex_references(self) -> List[Dict[str, Any]]:
        text = self.extract_text()
        references = []

        # Find DOIs in the lower half / reference portion of document
        doi_matches = DOI_REGEX.findall(text)
        seen_dois = set()
        for doi in doi_matches:
            clean_doi = doi.rstrip(".,;")
            if clean_doi not in seen_dois:
                seen_dois.add(clean_doi)
                references.append({
                    "raw_citation": f"DOI: {clean_doi}",
                    "title": None,
                    "authors": [],
                    "doi": clean_doi,
                    "arxiv_id": None,
                    "year": None,
                    "extraction_method": "pymupdf"
                })

        arxiv_matches = ARXIV_REGEX.findall(text)
        seen_arxiv = set()
        for aid in arxiv_matches:
            if aid not in seen_arxiv:
                seen_arxiv.add(aid)
                references.append({
                    "raw_citation": f"arXiv: {aid}",
                    "title": None,
                    "authors": [],
                    "doi": None,
                    "arxiv_id": aid,
                    "year": None,
                    "extraction_method": "pymupdf"
                })

        return references
