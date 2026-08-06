import asyncio
import re
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
        title = input_paper_meta.get("title", "")
        oa_id = input_paper_meta.get("openalex_id")
        doi = input_paper_meta.get("doi")
        arxiv_id = input_paper_meta.get("arxiv_id")

        if not oa_id and (doi or arxiv_id):
            work = await self.openalex_client.get_work(doi=doi, arxiv_id=arxiv_id)
            if work:
                oa_id = work.get("openalex_id")

        # Fetch OpenAlex candidates via related_to AND title search
        oa_candidates = []
        if oa_id:
            logger.info(f"Fetching OpenAlex related candidates for work {oa_id}...")
            oa_candidates = await self.openalex_client.find_related_candidates(oa_id, limit=100)

        if title:
            logger.info(f"Fetching OpenAlex search candidates for query '{title[:60]}...'")
            search_cands = await self.openalex_client.search_candidates(title, limit=50)
            oa_candidates.extend(search_cands)

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

        # Compute combined RRF and dense semantic domain relevance score for each candidate
        target_text = f"{title} {input_paper_meta.get('abstract', '') or ''}".strip()

        target_emb = None
        cand_embs = []
        if target_text and filtered_candidates:
            try:
                from lea.rag.embedder import BGEEmbedder
                emb_model_name = getattr(getattr(self.config, "embedding", None), "model", "BAAI/bge-m3") if self.config else "BAAI/bge-m3"
                embedder = BGEEmbedder(model_name=emb_model_name)
                cand_texts = [f"{c.get('title', '')} {c.get('abstract', '') or ''}".strip() for c in filtered_candidates]
                all_embs = embedder.embed_texts([target_text] + cand_texts)
                if all_embs and len(all_embs) == len(filtered_candidates) + 1:
                    target_emb = all_embs[0]
                    cand_embs = all_embs[1:]
            except Exception as exc:
                logger.warning(f"Dense relevance embedding failed: {exc}. Falling back to unigram matching.")

        import numpy as np
        target_vec = np.array(target_emb) if target_emb else None

        for idx, c in enumerate(filtered_candidates):
            c_idx = self._find_candidate_index(c, unified_candidates)
            ranks = candidate_ranks.get(c_idx, {})
            score = 0.0
            if "openalex" in ranks:
                score += 1.0 / (source_rrf_k + ranks["openalex"])
            if "semantic_scholar" in ranks:
                score += 1.0 / (source_rrf_k + ranks["semantic_scholar"])

            if target_vec is not None and idx < len(cand_embs):
                c_vec = np.array(cand_embs[idx])
                dot = float(np.dot(target_vec, c_vec))
                norm = float(np.linalg.norm(target_vec) * np.linalg.norm(c_vec))
                rel_score = max(0.0, dot / norm) if norm > 0 else 0.0
            else:
                rel_score = self._compute_relevance(target_text, f"{c.get('title', '')} {c.get('abstract', '') or ''}")

            c["domain_relevance"] = rel_score
            c["rrf_score"] = score * (1.0 + 5.0 * rel_score)

        # Filter out low relevance candidates (cosine similarity < 0.30)
        relevant_pool = [c for c in filtered_candidates if c.get("domain_relevance", 0.0) >= 0.30]
        if len(relevant_pool) >= final_candidate_limit:
            filtered_candidates = relevant_pool

        filtered_candidates.sort(key=lambda x: x.get("rrf_score", 0.0), reverse=True)

        # Assign final rrf_rank
        for i, cand in enumerate(filtered_candidates, start=1):
            cand["rrf_rank"] = i

        return filtered_candidates[:final_candidate_limit]

    def _compute_relevance(self, target_text: str, cand_text: str) -> float:
        stop_words = {
            'with', 'from', 'this', 'that', 'using', 'study', 'analysis', 'data', 'model',
            'results', 'effect', 'effects', 'group', 'patient', 'patients', 'journal', 'paper', 'article'
        }
        target_words = set(re.findall(r'\w{4,}', target_text.lower())) - stop_words
        if not target_words:
            return 1.0
        cand_words = set(re.findall(r'\w{3,}', cand_text.lower()))
        overlap = target_words.intersection(cand_words)
        return len(overlap) / (len(target_words) + 1e-5)

    def _find_candidate_index(self, item: Dict[str, Any], pool: List[Dict[str, Any]]) -> Optional[int]:
        for i, existing in enumerate(pool):
            if is_same_paper(item, existing):
                return i
        return None
