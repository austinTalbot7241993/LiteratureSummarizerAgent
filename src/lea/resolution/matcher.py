from typing import Optional, Dict, Any
from rapidfuzz import fuzz
from lea.resolution.identifiers import (
    normalize_doi, normalize_arxiv, normalize_openalex_id, normalize_s2_id, normalize_title
)

def exact_id_match(meta1: Dict[str, Any], meta2: Dict[str, Any]) -> bool:
    doi1, doi2 = normalize_doi(meta1.get("doi")), normalize_doi(meta2.get("doi"))
    if doi1 and doi2 and doi1 == doi2:
        return True

    arxiv1, arxiv2 = normalize_arxiv(meta1.get("arxiv_id")), normalize_arxiv(meta2.get("arxiv_id"))
    if arxiv1 and arxiv2 and arxiv1 == arxiv2:
        return True

    oa1, oa2 = normalize_openalex_id(meta1.get("openalex_id")), normalize_openalex_id(meta2.get("openalex_id"))
    if oa1 and oa2 and oa1 == oa2:
        return True

    s2_1, s2_2 = normalize_s2_id(meta1.get("s2_id")), normalize_s2_id(meta2.get("s2_id"))
    if s2_1 and s2_2 and s2_1 == s2_2:
        return True

    return False

def fuzzy_title_match(
    meta1: Dict[str, Any],
    meta2: Dict[str, Any],
    similarity_threshold: float = 0.96,
    year_tolerance: int = 1
) -> bool:
    t1 = normalize_title(meta1.get("title"))
    t2 = normalize_title(meta2.get("title"))

    if not t1 or not t2:
        return False

    # Check year tolerance if year is available on both
    y1 = meta1.get("publication_year") or meta1.get("year")
    y2 = meta2.get("publication_year") or meta2.get("year")

    if y1 is not None and y2 is not None:
        try:
            if abs(int(y1) - int(y2)) > year_tolerance:
                return False
        except (ValueError, TypeError):
            pass

    score = fuzz.token_set_ratio(t1, t2) / 100.0
    return score >= similarity_threshold

def is_same_paper(
    meta1: Dict[str, Any],
    meta2: Dict[str, Any],
    similarity_threshold: float = 0.96,
    year_tolerance: int = 1
) -> bool:
    if exact_id_match(meta1, meta2):
        return True
    return fuzzy_title_match(meta1, meta2, similarity_threshold, year_tolerance)
