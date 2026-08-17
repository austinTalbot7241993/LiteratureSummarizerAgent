import asyncio
from typing import List, Dict, Any, Optional, Set
from lea.discovery.openalex import OpenAlexClient
from lea.discovery.semantic_scholar import SemanticScholarClient
from lea.discovery.exclusion import ExclusionEngine
from lea.resolution.matcher import is_same_paper
from lea.resolution.metadata_merge import merge_paper_metadata
from lea.logging import logger
from lea.config import LEAConfig


class SecondaryGraphExpander:
    def __init__(
        self,
        openalex_client: Optional[OpenAlexClient] = None,
        semantic_scholar_client: Optional[SemanticScholarClient] = None,
        config: Optional[LEAConfig] = None
    ):
        self.openalex_client = openalex_client or OpenAlexClient()
        self.semantic_scholar_client = semantic_scholar_client or SemanticScholarClient()
        self.config = config

    async def expand_graph(
        self,
        accepted_candidates: List[Dict[str, Any]],
        input_paper_meta: Dict[str, Any],
        cited_references: List[Dict[str, Any]],
        already_seen_candidates: List[Dict[str, Any]],
        limit_per_paper: int = 50
    ) -> List[Dict[str, Any]]:
        """Harvests 2nd-degree candidate papers related to accepted candidate papers."""
        if not accepted_candidates:
            logger.info("No accepted candidates available to expand secondary graph.")
            return []

        logger.info(f"Expanding 2nd-degree graph across {len(accepted_candidates)} accepted papers...")

        new_raw_candidates: List[Dict[str, Any]] = []

        for paper in accepted_candidates:
            oa_id = paper.get("openalex_id")
            doi = paper.get("doi")
            arxiv_id = paper.get("arxiv_id")
            s2_id = paper.get("s2_id") or doi or arxiv_id

            # 1. Fetch OpenAlex related works
            if oa_id:
                try:
                    oa_cands = await self.openalex_client.find_related_candidates(oa_id, limit=limit_per_paper)
                    new_raw_candidates.extend(oa_cands)
                except Exception as exc:
                    logger.warning(f"Secondary graph expansion OpenAlex error for {oa_id}: {exc}")

            # 2. Fetch Semantic Scholar recommendations
            if s2_id:
                try:
                    s2_cands = await self.semantic_scholar_client.get_paper_recommendations(s2_id, limit=limit_per_paper)
                    new_raw_candidates.extend(s2_cands)
                except Exception as exc:
                    logger.warning(f"Secondary graph expansion S2 error for {s2_id}: {exc}")

        # Deduplicate new raw candidates
        unified_new: List[Dict[str, Any]] = []
        for item in new_raw_candidates:
            if not self._is_paper_in_pool(item, unified_new):
                unified_new.append(item)

        # Apply Citation Exclusion Invariant (input paper + cited references)
        exclusion_engine = ExclusionEngine(
            input_paper_meta=input_paper_meta,
            cited_references=cited_references,
            exclusion_status="complete",
            allow_incomplete_citation_exclusion=getattr(
                getattr(self.config, "extraction", None),
                "allow_incomplete_citation_exclusion",
                False
            )
        )
        filtered_new = exclusion_engine.filter_candidates(unified_new)

        # Filter out papers already evaluated / seen
        all_seen = list(accepted_candidates) + list(already_seen_candidates)
        novel_candidates = [
            cand for cand in filtered_new
            if not self._is_paper_in_pool(cand, all_seen)
        ]

        logger.info(f"Secondary graph expansion discovered {len(novel_candidates)} new valid candidate papers.")
        return novel_candidates

    def _is_paper_in_pool(self, item: Dict[str, Any], pool: List[Dict[str, Any]]) -> bool:
        for existing in pool:
            if is_same_paper(item, existing):
                return True
        return False
