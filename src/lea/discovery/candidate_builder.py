import asyncio
import re
from typing import List, Dict, Any, Optional
from lea.discovery.openalex import OpenAlexClient
from lea.discovery.semantic_scholar import SemanticScholarClient
from lea.discovery.exclusion import ExclusionEngine
from lea.discovery.abstract_screener import AbstractScreener
from lea.resolution.matcher import is_same_paper, fuzzy_title_match
from lea.resolution.metadata_merge import merge_paper_metadata
from lea.logging import logger

KEYWORD_STOPWORDS = {
    'with', 'from', 'this', 'that', 'using', 'study', 'analysis', 'data', 'model',
    'results', 'effect', 'effects', 'group', 'patient', 'patients', 'journal', 'paper', 'article',
    'when', 'where', 'which', 'into', 'under', 'over', 'such', 'been', 'have', 'were', 'their', 'than',
    'small', 'broad', 'class', 'methods', 'including', 'standard', 'number', 'positive', 'these', 'also',
    'both', 'each', 'more', 'most', 'other', 'some', 'while', 'well', 'many', 'across', 'based', 'novel'
}


def extract_search_keywords(text: str, max_terms: int = 10) -> str:
    """Extracts a short, distinct list of domain keywords from text for use
    as an OpenAlex/Semantic Scholar search query.

    OpenAlex's `search=` parameter is an AND-of-all-terms full-text filter,
    not a fuzzy relevance ranking (confirmed via its own `x_query.oql` debug
    field: "works where full text has (all these terms)") -- so a long,
    verbose query (e.g. the full title + a raw slice of the abstract) either
    matches nothing or falls back to seemingly-arbitrary noise, and a title
    containing the paper's own coined name (e.g. a novel tool's name) is
    *guaranteed* to match zero other papers, since by definition no prior
    work uses that name. A short list of generic domain terms is far more
    likely to co-occur with genuinely related papers.
    """
    words = re.findall(r"[A-Za-z][A-Za-z\-]{3,}", text)
    seen: List[str] = []
    seen_lower = set()
    for w in words:
        wl = w.lower()
        if wl in KEYWORD_STOPWORDS or wl in seen_lower:
            continue
        seen.append(w)
        seen_lower.add(wl)
        if len(seen) >= max_terms:
            break
    return " ".join(seen)


class CandidateBuilder:
    def __init__(
        self,
        openalex_client: Optional[OpenAlexClient] = None,
        semantic_scholar_client: Optional[SemanticScholarClient] = None,
        config: Any = None,
        llm_backend: Optional[Any] = None,
        embedder: Optional[Any] = None,
        screener: Optional[AbstractScreener] = None
    ):
        self.openalex_client = openalex_client or OpenAlexClient()
        self.semantic_scholar_client = semantic_scholar_client or SemanticScholarClient()
        self.config = config
        self.llm_backend = llm_backend
        self.embedder = embedder
        self.screener = screener or AbstractScreener(config=config, llm_backend=llm_backend, embedder=embedder)

    async def build_candidates(
        self,
        input_paper_meta: Dict[str, Any],
        cited_references: List[Dict[str, Any]],
        exclusion_status: str = "complete",
        final_candidate_limit: int = 20,
        source_rrf_k: int = 60,
        screen_abstracts: Optional[bool] = None,
        screening_method: Optional[str] = None,
        min_relevance_score: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        title = input_paper_meta.get("title", "")
        oa_id = input_paper_meta.get("openalex_id")
        doi = input_paper_meta.get("doi")
        arxiv_id = input_paper_meta.get("arxiv_id")

        if not oa_id and (doi or arxiv_id):
            work = await self.openalex_client.get_work(doi=doi, arxiv_id=arxiv_id)
            if work:
                # Sanity-check the resolved work's title against our own
                # extracted title before trusting its identity. A DOI/arXiv ID
                # extracted from a PDF's header (whether via GROBID's header
                # consolidation cross-referencing CrossRef, or GROBID's raw
                # header model, or a naive regex match against the body text)
                # can be flatly wrong -- confirmed live against a real
                # anonymized preprint, where GROBID non-deterministically
                # resolved it to two DIFFERENT unrelated papers' DOIs across
                # separate calls. Trusting a wrong DOI here doesn't just add
                # noise: it makes the ENTIRE discovery pass search that other
                # paper's citation neighborhood instead of the seed paper's,
                # which is far worse than finding no candidates at all.
                if title and not fuzzy_title_match(
                    {"title": title, "publication_year": input_paper_meta.get("publication_year")},
                    work,
                    similarity_threshold=0.85
                ):
                    logger.warning(
                        f"Discarding resolved OpenAlex identity for doi={doi!r} arxiv_id={arxiv_id!r}: "
                        f"resolved work title {work.get('title')!r} does not match the input paper's own "
                        f"title {title!r}. Treating doi/arxiv_id as unreliable for this run rather than "
                        f"risk searching the wrong paper's citation neighborhood."
                    )
                    doi = None
                    arxiv_id = None
                else:
                    oa_id = work.get("openalex_id")

        discovery_cfg = getattr(self.config, "discovery", None)
        openalex_limit = getattr(discovery_cfg, "openalex_limit", 100) if discovery_cfg else 100
        s2_limit = getattr(discovery_cfg, "semantic_scholar_limit", 100) if discovery_cfg else 100

        # When there is no citation-graph anchor at all (no OpenAlex ID
        # resolved, e.g. a brand-new/anonymized preprint with no DOI, arXiv
        # ID, or citations yet), a bare title search is the ONLY discovery
        # signal available -- and confirmed live, the full title alone (which
        # usually contains the paper's own coined name/method, guaranteed to
        # match zero other papers under OpenAlex's AND-of-all-terms search)
        # surfaces generic globally-popular "hub" papers instead of real
        # neighbors. Build a short domain-keyword query from the abstract
        # instead in that situation, since the abstract describes the
        # problem using established terminology other papers actually share.
        abstract = input_paper_meta.get("abstract") or ""
        search_query = title
        if not oa_id and abstract:
            keyword_query = extract_search_keywords(abstract, max_terms=10)
            if keyword_query:
                search_query = keyword_query

        # Fetch OpenAlex candidates via related_to AND title/keyword search
        oa_candidates = []
        if oa_id:
            logger.info(f"Fetching OpenAlex related candidates for work {oa_id}...")
            oa_candidates = await self.openalex_client.find_related_candidates(oa_id, limit=openalex_limit)

        if search_query:
            logger.info(f"Fetching OpenAlex search candidates for query '{search_query[:60]}...'")
            search_cands = await self.openalex_client.search_candidates(search_query, limit=openalex_limit)
            oa_candidates.extend(search_cands)

        # Fetch from Semantic Scholar: prefer a recommendations lookup keyed
        # by a stable identifier, but fall back to a plain keyword search
        # when none exists -- without this, S2 contributed nothing at all to
        # discovery for a paper with no DOI/arXiv/S2 ID.
        s2_candidates = []
        s2_id = input_paper_meta.get("s2_id") or doi or arxiv_id
        if s2_id:
            logger.info(f"Fetching Semantic Scholar recommendations for paper {s2_id}...")
            s2_candidates = await self.semantic_scholar_client.get_paper_recommendations(s2_id, limit=s2_limit)
        elif search_query:
            logger.info(f"No stable identifier for Semantic Scholar; falling back to keyword search for '{search_query[:60]}...'")
            s2_candidates = await self.semantic_scholar_client.search_papers(search_query, limit=s2_limit)

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

        # NOTE: domain_relevance is deliberately NOT used as a hard drop threshold here.
        # A single dense-embedding cosine score against a short title+abstract is noisy
        # and was previously discarding otherwise-relevant candidates before the abstract
        # screener ever saw them. It already factors into rrf_score as a re-rank boost
        # above; the abstract screening stage (LLM or embedding-based) is the intended
        # relevance gate, so let it operate on the full deduplicated/excluded pool.
        filtered_candidates.sort(key=lambda x: x.get("rrf_score", 0.0), reverse=True)

        # Check screening configuration and overrides
        screen_config = getattr(getattr(self.config, "discovery", None), "screening", None)
        if screen_abstracts is False:
            do_screening = False
        elif screen_abstracts is True:
            do_screening = True
        else:
            do_screening = getattr(screen_config, "enabled", True) if screen_config else True

        method = screening_method or (getattr(screen_config, "method", "llm") if screen_config else "llm")
        pre_limit = getattr(screen_config, "pre_screening_limit", 50) if screen_config else 50
        min_score = min_relevance_score if min_relevance_score is not None else (getattr(screen_config, "min_relevance_score", 6.0) if screen_config else 6.0)
        # max_screened_candidates only narrows the pool when explicitly configured;
        # otherwise it defers to final_candidate_limit so screening doesn't silently
        # shrink the pool the downstream quota loop draws from.
        configured_max = getattr(screen_config, "max_screened_candidates", None) if screen_config else None
        max_candidates = configured_max if configured_max is not None else final_candidate_limit

        if do_screening:
            logger.info(f"Executing abstract screening stage (method='{method}', min_score={min_score}, pre_limit={pre_limit})...")
            pre_screen_pool = filtered_candidates[:pre_limit]
            filtered_candidates = await self.screener.screen_candidates(
                seed_paper_meta=input_paper_meta,
                candidates=pre_screen_pool,
                method=method,
                min_score=min_score,
                max_candidates=max_candidates
            )

        # Assign final rrf_rank
        for i, cand in enumerate(filtered_candidates, start=1):
            cand["rrf_rank"] = i

        return filtered_candidates[:max_candidates]

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
