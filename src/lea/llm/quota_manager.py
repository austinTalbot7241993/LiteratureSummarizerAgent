import asyncio
import os
import uuid
from typing import List, Dict, Any, Optional, Set
from lea.db.repository import LEARepository
from lea.discovery.graph_expansion import SecondaryGraphExpander
from lea.logging import logger
from lea.config import LEAConfig


class IterativeSourceManager:
    def __init__(
        self,
        config: Optional[LEAConfig] = None,
        target_sources: int = 20,
        min_relevance_score: float = 6.0,
        min_grounding_score: float = 7.0,
        max_search_depth: int = 100,
        enable_secondary_graph_expansion: bool = True,
        summarizer: Optional[Any] = None,
        downloader: Optional[Any] = None,
        chunker: Optional[Any] = None,
        embedder: Optional[Any] = None,
        hybrid_engine: Optional[Any] = None,
        expander: Optional[SecondaryGraphExpander] = None,
        fast_path_min_score: float = 3.0
    ):
        self.config = config
        self.target_sources = target_sources
        self.min_relevance_score = min_relevance_score
        self.min_grounding_score = min_grounding_score
        # Deliberately much lower than -- and independent of -- min_relevance_score.
        # AbstractScreener.screen_candidates() already decides which candidates
        # reach this queue at all, either by clearing its own min_score or, when
        # nothing does, via its designed threshold-relaxation fallback (it
        # retains the best available candidates rather than returning an empty
        # pool). Re-checking abstract_relevance_score against the *same*
        # min_relevance_score here as a "cheap prune" was redundant at best and,
        # whenever relaxation fired, actively self-defeating: every
        # relaxation-admitted candidate is by definition below that threshold
        # (that's why it needed relaxing), so this fast-path skip silently
        # rejected 100% of them before RAG summarization or self-critique ever
        # ran -- observed live: 20/20 candidates "evaluated" and rejected in
        # under half a second, with none of the expected PDF-download/LLM
        # latency, because every one carried a neutral placeholder score of 5.0
        # from a missing abstract and 5.0 < the (identical) 6.0 acceptance
        # threshold. This floor instead matches AbstractScreener's own
        # "irrelevant" tier boundary (score < 4.0), so only candidates the
        # screener itself would call obviously irrelevant are skipped without
        # ever reaching full evaluation.
        self.fast_path_min_score = fast_path_min_score
        self.max_search_depth = max_search_depth
        self.enable_secondary_graph_expansion = enable_secondary_graph_expansion
        self.summarizer = summarizer
        self.downloader = downloader
        self.chunker = chunker
        self.embedder = embedder
        self.hybrid_engine = hybrid_engine
        self.expander = expander or SecondaryGraphExpander(config=config)

    async def execute_quota_loop(
        self,
        repo: LEARepository,
        run_id: uuid.UUID,
        candidate_queue: List[Dict[str, Any]],
        input_paper_meta: Dict[str, Any],
        cited_references: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Executes candidate evaluation loop until target accepted source quota is satisfied."""
        accepted_candidates: List[Dict[str, Any]] = []
        seen_candidates: List[Dict[str, Any]] = []
        queue = list(candidate_queue)
        evaluated_count = 0
        # Tracks papers actually processed DURING this call, so a paper that
        # reappears later in the queue (e.g. rediscovered by secondary graph
        # expansion) is correctly skipped as a duplicate. This is deliberately
        # NOT the same check as "does a CandidatePaper row already exist for
        # this run" -- the caller (cli.py's `discover` command) pre-registers a
        # CandidatePaper row for every screened candidate before this loop ever
        # runs, so that row's mere existence is not evidence of evaluation.
        processed_paper_ids: Set[uuid.UUID] = set()

        logger.info(f"Starting Iterative Source Quota Loop (Target: {self.target_sources} sources, Max Depth: {self.max_search_depth})...")

        while len(accepted_candidates) < self.target_sources and queue and evaluated_count < self.max_search_depth:
            cand_dict = queue.pop(0)
            evaluated_count += 1
            seen_candidates.append(cand_dict)
            cand_title = cand_dict.get("title", "Untitled Candidate")

            logger.info(f"Evaluating candidate [{evaluated_count}/{self.max_search_depth}]: '{cand_title}'")

            # 1. Ensure candidate paper exists in DB. Resolve by stable external
            # identifiers FIRST -- candidates from discovery/graph-expansion never
            # carry a sha256_hash, so relying on that alone meant the same
            # real-world paper (e.g. a highly-cited "hub" paper repeatedly
            # rediscovered via secondary graph expansion) was always treated as
            # brand new and re-evaluated from scratch every time it reappeared.
            existing_paper = repo.find_paper_by_external_ids(
                doi=cand_dict.get("doi"),
                arxiv_id=cand_dict.get("arxiv_id"),
                openalex_id=cand_dict.get("openalex_id"),
                s2_id=cand_dict.get("s2_id")
            )
            if not existing_paper:
                c_sha = cand_dict.get("sha256_hash") or str(uuid.uuid4())[:32]
                existing_paper = repo.get_paper_by_hash(c_sha)

            if existing_paper and existing_paper.id in processed_paper_ids:
                # A genuine duplicate encountered WITHIN this loop run (e.g. the
                # same real-world paper rediscovered by secondary graph
                # expansion from two different accepted papers' neighborhoods).
                logger.info(f"Skipping duplicate candidate '{cand_title}' -- already evaluated earlier in this run.")
                continue

            if existing_paper:
                c_paper = existing_paper
            else:
                c_sha = cand_dict.get("sha256_hash") or str(uuid.uuid4())[:32]
                c_paper = repo.create_paper(
                    sha256_hash=c_sha,
                    title=cand_dict.get("title", "Untitled"),
                    authors=cand_dict.get("authors", []),
                    doi=cand_dict.get("doi"),
                    arxiv_id=cand_dict.get("arxiv_id"),
                    openalex_id=cand_dict.get("openalex_id"),
                    s2_id=cand_dict.get("s2_id"),
                    publication_year=cand_dict.get("publication_year"),
                    venue=cand_dict.get("venue"),
                    abstract=cand_dict.get("abstract"),
                    is_open_access=cand_dict.get("is_open_access", False),
                    oa_pdf_url=cand_dict.get("oa_pdf_url")
                )

            processed_paper_ids.add(c_paper.id)

            # Reuse a CandidatePaper row that the caller may have already
            # pre-registered for this run (cli.py's `discover` command inserts
            # one for every screened candidate before this loop starts) instead
            # of blindly inserting a second row for the same (run, paper) pair.
            cand_rec = repo.get_candidate_for_run_and_paper(run_id, c_paper.id)
            if cand_rec is None:
                cand_rec = repo.add_candidate_paper(
                    run_id=run_id,
                    paper_id=c_paper.id,
                    score=cand_dict.get("rrf_score", 0.0),
                    rrf_rank=cand_dict.get("rrf_rank", evaluated_count),
                    source_apis=cand_dict.get("source_apis", []),
                    open_access_url=cand_dict.get("oa_pdf_url"),
                    abstract_relevance_score=cand_dict.get("abstract_relevance_score"),
                    abstract_relevance_tier=cand_dict.get("abstract_relevance_tier"),
                    abstract_relevance_reasoning=cand_dict.get("abstract_relevance_reasoning")
                )

            # Fast-path abstract relevance check before PDF download & RAG summarization.
            # Uses fast_path_min_score (a low, fixed floor), NOT min_relevance_score
            # (the final acceptance gate) -- see the constructor comment for why
            # reusing the acceptance threshold here defeated AbstractScreener's
            # own threshold-relaxation fallback.
            abs_score = cand_dict.get("abstract_relevance_score")
            if abs_score is not None and abs_score < self.fast_path_min_score:
                logger.info(f"Skipping RAG summarization for '{cand_title}' - pre-screened abstract relevance ({abs_score}) < fast-path floor ({self.fast_path_min_score})")
                continue

            # 2. Acquire PDF if needed
            pdf_path = cand_dict.get("pdf_path")
            if not pdf_path and self.downloader:
                download_meta = {
                    "title": c_paper.title,
                    "doi": c_paper.doi,
                    "arxiv_id": c_paper.arxiv_id,
                    "s2_id": c_paper.s2_id,
                    "oa_pdf_url": cand_rec.open_access_url or c_paper.oa_pdf_url
                }
                try:
                    pdf_path = await self.downloader.download_candidate_pdf(download_meta)
                    if pdf_path:
                        cand_rec.pdf_path = pdf_path
                        cand_rec.is_downloaded = True
                except Exception as exc:
                    logger.warning(f"PDF download failed for '{c_paper.title}': {exc}")

            # 3. Index & Embed Chunks if downloader / chunker present
            if self.chunker and self.embedder:
                text_content = ""
                if pdf_path and os.path.exists(pdf_path):
                    try:
                        from lea.ingester.pdf_parser import PDFParser
                        text_content = PDFParser(pdf_path).extract_body_text()
                    except Exception as exc:
                        logger.warning(f"Failed to parse PDF text for {c_paper.title}: {exc}")

                if not text_content:
                    text_content = c_paper.abstract or c_paper.title

                if text_content:
                    try:
                        chunk_objs = self.chunker.chunk_text(text_content)
                        child_contents = [c["content"] for c in chunk_objs if c["chunk_type"] == "child"]
                        embeddings = self.embedder.embed_texts(child_contents) if child_contents else []

                        parent_id_map = {}
                        child_emb_idx = 0
                        for c in chunk_objs:
                            p_id = parent_id_map.get(c.get("parent_index")) if c["chunk_type"] == "child" else None
                            emb = embeddings[child_emb_idx] if (c["chunk_type"] == "child" and child_emb_idx < len(embeddings)) else None
                            if c["chunk_type"] == "child":
                                child_emb_idx += 1

                            chunk_rec = repo.add_chunk(
                                paper_id=c_paper.id,
                                run_id=run_id,
                                chunk_type=c["chunk_type"],
                                content=c["content"],
                                chunk_index=c["chunk_index"],
                                token_count=c["token_count"],
                                parent_id=p_id,
                                embedding=emb
                            )
                            if c["chunk_type"] == "parent":
                                parent_id_map[c["chunk_index"]] = chunk_rec.id
                    except Exception as exc:
                        logger.warning(f"Chunk indexing failed for '{c_paper.title}': {exc}")

            # 4. Hybrid Search + RAG Summarization & Self-Critique
            retrieved_chunks = []
            if self.hybrid_engine:
                try:
                    retrieved_chunks_objs = self.hybrid_engine.hybrid_search(
                        repo=repo,
                        run_id=run_id,
                        paper_id=c_paper.id,
                        query_text=f"Methodology algorithm model evaluation findings of {c_paper.title}",
                        dense_top_k=30,
                        sparse_top_k=30,
                        fused_top_k=8
                    )
                    retrieved_chunks = [item[0] for item in retrieved_chunks_objs]
                except Exception as exc:
                    logger.warning(f"Hybrid retrieval error for {c_paper.title}: {exc}")

            cand_meta = {
                "title": c_paper.title,
                "authors": c_paper.authors,
                "publication_year": c_paper.publication_year
            }

            if self.summarizer:
                try:
                    tech_summary, assessment = self.summarizer.summarize_candidate(
                        cand_meta,
                        retrieved_chunks,
                        availability_chunks=retrieved_chunks,
                        target_paper_meta=input_paper_meta
                    )

                    critique = getattr(tech_summary, "critique", None)
                    # Fail SAFE: a candidate with no critique at all (e.g. the
                    # summarizer implementation didn't perform one) has received
                    # no relevance judgment whatsoever, and must not be treated
                    # as accepted by default -- that was the same "fail open"
                    # pattern as the critique-failure fallback in
                    # TechnicalSummarizer.critique_candidate.
                    is_accepted = False
                    critique_verdict = None
                    rel_score = None
                    gnd_score = None
                    critique_rationale = None

                    if critique:
                        critique_verdict = critique.verdict
                        rel_score = critique.relevance_score
                        gnd_score = critique.factual_grounding_score
                        critique_rationale = critique.critique_rationale

                        is_accepted = (
                            critique.is_relevant_to_seed_topic and
                            critique_verdict in ["accepted", "marginal"] and
                            rel_score >= self.min_relevance_score and
                            gnd_score >= self.min_grounding_score
                        )

                    repo.add_summary(
                        run_id=run_id,
                        candidate_paper_id=cand_rec.id,
                        problem_formulation=tech_summary.problem_formulation,
                        methodological_novelty=tech_summary.methodological_novelty,
                        empirical_findings=tech_summary.empirical_findings,
                        paragraph_summary=tech_summary.paragraph_summary,
                        relationship_to_target=tech_summary.relationship_to_target,
                        data_availability=assessment.overall_status.value if hasattr(assessment.overall_status, "value") else str(assessment.overall_status),
                        data_location=tech_summary.data_location,
                        data_availability_assessment=assessment.model_dump(mode="json"),
                        model_name=getattr(self.config.llm, "model", "Qwen/Qwen2.5-7B-Instruct") if self.config else "Qwen/Qwen2.5-7B-Instruct",
                        self_critique_verdict=critique_verdict,
                        self_critique_relevance_score=rel_score,
                        self_critique_grounding_score=gnd_score,
                        self_critique_rationale=critique_rationale,
                        is_accepted=is_accepted
                    )

                    if is_accepted:
                        accepted_candidates.append(cand_dict)
                        pruned_count = evaluated_count - len(accepted_candidates)
                        progress_bar = "█" * len(accepted_candidates) + "░" * max(0, self.target_sources - len(accepted_candidates))
                        logger.info(
                            f"[PROGRESS] [{progress_bar}] {len(accepted_candidates)}/{self.target_sources} Valid Papers Accepted "
                            f"({pruned_count} Pruned as Irrelevant)"
                        )
                    else:
                        logger.warning(
                            f"Pruned candidate '{cand_title}' (Verdict={critique_verdict}, Rel={rel_score}, Grounding={gnd_score}): "
                            f"{critique_rationale}. Fetching replacement ({len(accepted_candidates)}/{self.target_sources} accepted)..."
                        )
                except Exception as exc:
                    logger.warning(f"Summarization error for '{cand_title}': {exc}")

            # 5. Secondary Graph Expansion if Queue Empty & Quota Unfilled
            if not queue and len(accepted_candidates) < self.target_sources and self.enable_secondary_graph_expansion:
                logger.info(f"Candidate queue exhausted; triggering 2nd-degree citation graph expansion ({len(accepted_candidates)}/{self.target_sources} accepted)...")
                new_candidates = await self.expander.expand_graph(
                    accepted_candidates=accepted_candidates,
                    input_paper_meta=input_paper_meta,
                    cited_references=cited_references,
                    already_seen_candidates=seen_candidates
                )
                queue.extend(new_candidates)

        logger.info(f"Iterative Quota Loop Completed. Accepted {len(accepted_candidates)} valid sources out of {evaluated_count} evaluated.")
        return accepted_candidates
