from typing import List, Dict, Any
from lea.resolution.matcher import is_same_paper
from lea.exceptions import ExclusionViolationError
from lea.logging import logger

class ExclusionEngine:
    def __init__(
        self,
        input_paper_meta: Dict[str, Any],
        cited_references: List[Dict[str, Any]],
        exclusion_status: str = "complete",
        allow_incomplete_citation_exclusion: bool = False,
        similarity_threshold: float = 0.96,
        year_tolerance: int = 1
    ):
        self.input_paper_meta = input_paper_meta
        self.cited_references = cited_references
        self.exclusion_status = exclusion_status
        self.allow_incomplete_citation_exclusion = allow_incomplete_citation_exclusion
        self.similarity_threshold = similarity_threshold
        self.year_tolerance = year_tolerance

    def validate_exclusion_status(self) -> None:
        """Fails closed if citation exclusion cannot be established with complete bibliography."""
        if self.exclusion_status != "complete" and not self.allow_incomplete_citation_exclusion:
            raise ExclusionViolationError(
                "Citation exclusion cannot be established because bibliography extraction is incomplete "
                "and allow_incomplete_citation_exclusion is set to False."
            )

    def filter_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        r"""
        Filters candidates to satisfy invariant:
        S_discovered \cap (C_cited \cup {p_input}) = \emptyset
        """
        self.validate_exclusion_status()

        valid_candidates = []
        excluded_count = 0

        for candidate in candidates:
            # Check 1: Must not match input paper
            if is_same_paper(
                candidate,
                self.input_paper_meta,
                similarity_threshold=self.similarity_threshold,
                year_tolerance=self.year_tolerance
            ):
                logger.debug(f"Excluded candidate matching input paper: {candidate.get('title')}")
                excluded_count += 1
                continue

            # Check 2: Must not match any paper in cited references
            is_cited = False
            for ref in self.cited_references:
                if is_same_paper(
                    candidate,
                    ref,
                    similarity_threshold=self.similarity_threshold,
                    year_tolerance=self.year_tolerance
                ):
                    logger.debug(f"Excluded candidate matching cited reference: {candidate.get('title')}")
                    is_cited = True
                    break

            if is_cited:
                excluded_count += 1
                continue

            valid_candidates.append(candidate)

        logger.info(f"Exclusion engine processed {len(candidates)} candidates: {len(valid_candidates)} kept, {excluded_count} excluded.")
        return valid_candidates
