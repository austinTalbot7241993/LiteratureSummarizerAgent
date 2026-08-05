import asyncio
from typing import List, Dict, Any, Optional
from lea.discovery.openalex import OpenAlexClient
from lea.discovery.semantic_scholar import SemanticScholarClient
from lea.discovery.exclusion import ExclusionEngine
from lea.resolution.matcher import is_same_paper
from lea.resolution.metadata_merge import merge_paper_metadata
from lea.logging import logger

class CandidateBuilder:
    def __init__(
        self,
        openalex_client: Optional[OpenAlexClient] = None,
        semantic_scholar_client: Optional[SemanticScholarClient] = None,
        config: Any = None
    ):
        self.openalex_client = openalex_client or OpenAlexClient()
        self.semantic_scholar_client = semantic_scholar_client or SemanticScholarClient()
        self.config = config

    async def build_candidates(
        self,
        input_paper_meta: Dict[str, Any],
        cited_references: List[Dict[str, Any]],
        exclusion_status: str = "complete",
        final_candidate_limit: int = 20,
        source_rrf_k: int = 60
    ) -> List[Dict[str, Any]]:
        # Fetch from OpenAlex
        oa_candidates = []
        oa_id = input_paper_meta.get("openalex_id")
        doi = input_paper_meta.get("doi")
        arxiv_id = input_paper_meta.get("arxiv_id")

        if not oa_id and (doi or arxiv_id):
            work = await self.openalex_client.get_work(doi=doi, arxiv_id=arxiv_id)
            if work:
                oa_id = work.get("openalex_id")

        if oa_id:
            logger.info(f"Fetching OpenAlex related candidates for work {oa_id}...")
            oa_candidates = await self.openalex_client.find_related_candidates(oa_id, limit=100)

        # Fetch from Semantic Scholar
        s2_candidates = []
        s2_id = input_paper_meta.get("s2_id") or doi or arxiv_id
        if s2_id:
            logger.info(f"Fetching Semantic Scholar recommendations for paper {s2_id}...")
            s2_candidates = await self.semantic_scholar_client.get_paper_recommendations(s2_id, limit=100)

        # Merge candidate pools into unified deduplicated list with source ranks
        unified_candidates: List[Dict[str, Any]] = []
        candidate_ranks: Dict[int, Dict[str, int]] = {} # index -> {oa_rank, s2_rank}

        # Process OpenAlex list
        for rank, item in enumerate(oa_candidates, start=1):
            idx = self._find_candidate_index(item, unified_candidates)
            if idx is None:
                unified_candidates.append(item)
                idx = len(unified_candidates) - 1
                item["source_apis"] = ["openalex"]
            else:
                unified_candidates[idx] = merge_paper_metadata(unified_candidates[idx], item)
                if "openalex" not in unified_candidates[idx].get("source_apis", []):
                    unified_candidates[idx].setdefault("source_apis", []).append("openalex")

            candidate_ranks.setdefault(idx, {})["openalex"] = rank

        # Process Semantic Scholar list
        for rank, item in enumerate(s2_candidates, start=1):
            idx = self._find_candidate_index(item, unified_candidates)
            if idx is None:
                unified_candidates.append(item)
                idx = len(unified_candidates) - 1
                item["source_apis"] = ["semantic_scholar"]
            else:
                unified_candidates[idx] = merge_paper_metadata(unified_candidates[idx], item)
                if "semantic_scholar" not in unified_candidates[idx].get("source_apis", []):
                    unified_candidates[idx].setdefault("source_apis", []).append("semantic_scholar")

            candidate_ranks.setdefault(idx, {})["semantic_scholar"] = rank

        # Apply strict exclusion invariant
        exclusion_engine = ExclusionEngine(
            input_paper_meta=input_paper_meta,
            cited_references=cited_references,
            exclusion_status=exclusion_status,
            allow_incomplete_citation_exclusion=getattr(
                getattr(self.config, "extraction", None),
                "allow_incomplete_citation_exclusion",
                False
            )
        )
        filtered_candidates = exclusion_engine.filter_candidates(unified_candidates)

        # Compute RRF score for each filtered candidate
        for c in filtered_candidates:
            idx = self._find_candidate_index(c, unified_candidates)
            ranks = candidate_ranks.get(idx, {})
            score = 0.0
            if "openalex" in ranks:
                score += 1.0 / (source_rrf_k + ranks["openalex"])
            if "semantic_scholar" in ranks:
                score += 1.0 / (source_rrf_k + ranks["semantic_scholar"])
            c["rrf_score"] = score

        filtered_candidates.sort(key=lambda x: x.get("rrf_score", 0.0), reverse=True)

        # Assign final rrf_rank
        for i, cand in enumerate(filtered_candidates, start=1):
            cand["rrf_rank"] = i

        return filtered_candidates[:final_candidate_limit]

    def _find_candidate_index(self, item: Dict[str, Any], pool: List[Dict[str, Any]]) -> Optional[int]:
        for i, existing in enumerate(pool):
            if is_same_paper(item, existing):
                return i
        return None
