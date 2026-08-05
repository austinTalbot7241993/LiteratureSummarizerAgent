from typing import Dict, Any, Tuple

class BibliographyVerifier:
    @staticmethod
    def verify_metadata(paper_meta: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Verifies if bibliographic metadata has essential fields (title, authors, year).
        Returns tuple of (is_verified, normalized_meta).
        """
        title = paper_meta.get("title")
        authors = paper_meta.get("authors") or []
        year = paper_meta.get("publication_year") or paper_meta.get("year")

        is_verified = bool(title and len(title.strip()) > 3 and (authors or year))

        status = {
            "title_valid": bool(title),
            "has_authors": len(authors) > 0,
            "has_year": year is not None,
            "has_doi": bool(paper_meta.get("doi")),
            "has_arxiv": bool(paper_meta.get("arxiv_id")),
            "is_verified": is_verified
        }

        return is_verified, status
