import re
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional
import fitz  # PyMuPDF
from lea.exceptions import IngestError
from lea.logging import logger

DOI_REGEX = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")
ARXIV_REGEX = re.compile(r"(?:arXiv:\s*|arxiv/|abs/)?(\d{4}\.\d{4,5}(?:v\d+)?)", re.IGNORECASE)
ABSTRACT_HEADING_REGEX = re.compile(r"^\s*abstract\s*[:.]?\s*$", re.IGNORECASE)
SECTION_HEADING_REGEX = re.compile(
    r"^\s*(?:[0-9]+\.?\s+)?(introduction|keywords|index terms)\b",
    re.IGNORECASE
)
TITLE_TERMINAL_PUNCTUATION = (".", "!", "?", ":")
NON_TITLE_LINE_KEYWORDS_REGEX = re.compile(
    r"\b(anonymous|university|institute|department|"
    r"corresponding author|et al\.?)\b|@",
    re.IGNORECASE
)
TITLE_CONNECTOR_WORDS = {
    "of", "the", "and", "for", "with", "in", "on", "using", "via", "to",
    "a", "an", "from", "into", "under", "over", "as", "is", "are", "vs", "vs."
}
BOILERPLATE_LINE_REGEX = re.compile(
    r"preprint|under review|manuscript submitted|copyright|arxiv:|"
    r"\d+\s*[:–-]\s*\d+\s*[–-]\s*\d+\s*,\s*\d{4}|"  # e.g. "298:1-21, 2026"
    r"^\s*\d+\s*$",  # bare page number
    re.IGNORECASE
)

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

        title = self._extract_title_by_font_size() or (self._extract_fallback_title(lines) if lines else self.path.stem)

        # RegEx matching for the PAPER'S OWN DOI/arXiv ID must be scoped to
        # the front matter only, never the whole document. A paper's own
        # identifier (when it has one) legitimately appears near the title/
        # author block; a *cited* work's identifier does not. Running
        # DOI_REGEX/ARXIV_REGEX over the entire text and taking the first
        # match is wrong for any paper that has no DOI of its own (e.g. an
        # unpublished arXiv preprint) but does cite >=1 work that has one:
        # the first DOI-shaped string anywhere in the document is then
        # guaranteed to be a bibliography entry's DOI, silently misattributed
        # as this paper's own identity -- confirmed live on a preprint whose
        # extracted "self" DOI turned out to be its own first reference's
        # DOI, which downstream discovery then had to detect and discard.
        front_matter = self._extract_front_matter_text(text)
        doi_matches = DOI_REGEX.findall(front_matter)
        doi = doi_matches[0].rstrip(".,;") if doi_matches else None

        arxiv_matches = ARXIV_REGEX.findall(front_matter)
        arxiv_id = arxiv_matches[0] if arxiv_matches else None

        abstract = self._extract_fallback_abstract(lines) or text[:1000]

        return {
            "title": title,
            "abstract": abstract,
            "authors": [],
            "doi": doi,
            "arxiv_id": arxiv_id,
            "publication_year": None,
            "full_text": text
        }

    @staticmethod
    def _extract_front_matter_text(text: str, max_chars: int = 4000) -> str:
        """Returns the text preceding the first major section heading
        (Introduction/Keywords/Index Terms -- the same boundary already used
        by `_extract_fallback_abstract`), which is where a paper's own DOI/
        arXiv identifier legitimately appears (title/author/abstract block).
        Self-identifier extraction must never run over the whole document,
        or a cited work's DOI in the bibliography gets mistaken for the
        paper's own. Falls back to a bounded character-count prefix if no
        such heading is found, so the front-matter window stays bounded
        either way rather than silently degrading back to the whole text.
        """
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if SECTION_HEADING_REGEX.match(line.strip()):
                return "\n".join(lines[:i])
        return text[:max_chars]

    def _extract_title_by_font_size(self) -> Optional[str]:
        """Uses PyMuPDF's per-span font-size metadata to identify the title
        as the largest-font text on the first page. This is a substantially
        more reliable signal than line-order heuristics: running headers,
        DOIs, and author/affiliation lines are almost always rendered in a
        smaller font than the title, whereas plain-text line order provides
        no such distinction. Falls back to None (letting the caller use the
        line-based heuristic) on any failure, empty result, implausible
        length, or a result that looks like a watermark/banner (e.g. an
        arXiv stamp is sometimes rendered in a larger font than the title).
        """
        try:
            doc = fitz.open(self.path)
            if len(doc) == 0:
                doc.close()
                return None
            page_dict = doc[0].get_text("dict")
            doc.close()

            spans = []
            for block in page_dict.get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text_piece = span.get("text", "").strip()
                        if text_piece:
                            spans.append((span.get("size", 0.0), span.get("bbox", (0.0, 0.0, 0.0, 0.0))[1], text_piece))

            if not spans:
                return None

            max_size = max(s[0] for s in spans)
            title_spans = sorted((s for s in spans if s[0] >= max_size - 0.5), key=lambda s: s[1])
            title = " ".join(s[2] for s in title_spans).strip()

            if title and 8 <= len(title) <= 300 and not BOILERPLATE_LINE_REGEX.search(title):
                return title
            return None
        except Exception as exc:
            logger.warning(f"Font-size-based title extraction failed: {exc}")
            return None

    @staticmethod
    def _extract_fallback_title(lines: List[str], max_lines: int = 3) -> str:
        """Academic paper titles frequently wrap across two or more lines in
        PDF-extracted text with no blank line to signal the wrap (PyMuPDF's
        text-mode extraction does not reliably preserve paragraph breaks).
        Naively taking only the first line truncates titles mid-phrase (e.g.
        a title ending "...Estimation for" whose continuation "Low-Frequency
        Variants..." sits on the next line). Merge additional leading lines
        into the title while the accumulated title has not yet reached
        sentence-terminal punctuation, since a genuine title continuation
        line follows a line with no terminal punctuation far more often than
        an author/affiliation line does. Capped at `max_lines` to avoid
        swallowing author/affiliation lines on documents with an
        unpunctuated single-line title.
        """
        if not lines:
            return ""

        # Skip leading running-header/banner lines (e.g. "Preprint: Under
        # Review 298:1-21, 2026") that a title-detection heuristic could
        # otherwise mistake for the start of the title itself.
        start = 0
        while start < len(lines) - 1 and BOILERPLATE_LINE_REGEX.search(lines[start]):
            start += 1

        title = lines[start]
        for line in lines[start + 1:start + max_lines]:
            if title.rstrip().endswith(TITLE_TERMINAL_PUNCTUATION):
                break
            if PDFParser._looks_like_author_or_affiliation_line(line):
                break
            title = f"{title} {line}"
        return title

    @staticmethod
    def _looks_like_author_or_affiliation_line(line: str) -> bool:
        """Heuristic guard so title-line-wrap merging doesn't swallow an
        author/affiliation line that (like the title itself) may lack
        terminal punctuation -- e.g. "Anonymous Author" or "Anonymous
        Institution" on an anonymized submission.
        """
        if NON_TITLE_LINE_KEYWORDS_REGEX.search(line):
            return True
        words = line.split()
        if 1 <= len(words) <= 4:
            lowered = [w.lower().strip(",.") for w in words]
            all_capitalized = all(w[:1].isupper() for w in words if w)
            has_connector = any(w in TITLE_CONNECTOR_WORDS for w in lowered)
            if all_capitalized and not has_connector:
                return True
        return False

    @staticmethod
    def _extract_fallback_abstract(lines: List[str], max_chars: int = 2500) -> Optional[str]:
        """Finds a standalone "Abstract" heading line and returns the text
        following it up to the next section heading (or a length cap),
        rather than the first N raw characters of the whole document (which
        previously mixed the title/author/affiliation lines in with actual
        abstract content, or returned no abstract at all when GROBID was
        also unavailable).
        """
        for i, line in enumerate(lines):
            if ABSTRACT_HEADING_REGEX.match(line):
                body_lines: List[str] = []
                total_chars = 0
                for candidate_line in lines[i + 1:]:
                    if SECTION_HEADING_REGEX.match(candidate_line):
                        break
                    body_lines.append(candidate_line)
                    total_chars += len(candidate_line)
                    if total_chars >= max_chars:
                        break
                abstract = " ".join(body_lines).strip()
                if abstract:
                    return abstract
        return None

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
