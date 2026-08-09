import uuid
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import select, text
from lea.db.models import (
    Paper, PaperReference, DiscoveryRun, CandidatePaper, TextChunk, TechnicalSummaryModel
)

class LEARepository:
    def __init__(self, session: Session):
        self.session = session

    # --- Paper Operations ---
    def get_paper_by_hash(self, sha256_hash: str) -> Optional[Paper]:
        return self.session.query(Paper).filter(Paper.sha256_hash == sha256_hash).first()

    def get_paper_by_id(self, paper_id: uuid.UUID) -> Optional[Paper]:
        return self.session.query(Paper).filter(Paper.id == paper_id).first()

    def get_paper_by_doi(self, doi: str) -> Optional[Paper]:
        return self.session.query(Paper).filter(Paper.doi == doi).first()

    def _clean_str(self, val: Any) -> Any:
        if isinstance(val, str):
            return val.replace("\x00", "").replace("\u0000", "")
        return val

    def create_paper(self, **kwargs) -> Paper:
        clean_kwargs = {k: self._clean_str(v) for k, v in kwargs.items()}
        paper = Paper(**clean_kwargs)
        self.session.add(paper)
        self.session.flush()
        return paper

    def update_paper(self, paper_id: uuid.UUID, **kwargs) -> Paper:
        paper = self.get_paper_by_id(paper_id)
        if paper:
            for k, v in kwargs.items():
                setattr(paper, k, self._clean_str(v))
            self.session.flush()
        return paper

    # --- Paper References ---
    def add_reference(self, source_paper_id: uuid.UUID, **kwargs) -> PaperReference:
        clean_kwargs = {k: self._clean_str(v) for k, v in kwargs.items()}
        ref = PaperReference(source_paper_id=source_paper_id, **clean_kwargs)
        self.session.add(ref)
        self.session.flush()
        return ref

    def get_references_for_paper(self, paper_id: uuid.UUID) -> List[PaperReference]:
        return self.session.query(PaperReference).filter(PaperReference.source_paper_id == paper_id).all()

    # --- Discovery Runs ---
    def create_discovery_run(self, input_paper_id: uuid.UUID, run_status: str = "initialized", exclusion_status: str = "complete") -> DiscoveryRun:
        run = DiscoveryRun(input_paper_id=input_paper_id, run_status=run_status, exclusion_status=exclusion_status)
        self.session.add(run)
        self.session.flush()
        return run

    def get_discovery_run(self, run_id: uuid.UUID) -> Optional[DiscoveryRun]:
        return self.session.query(DiscoveryRun).filter(DiscoveryRun.id == run_id).first()

    def update_discovery_run(self, run_id: uuid.UUID, run_status: Optional[str] = None, exclusion_status: Optional[str] = None) -> DiscoveryRun:
        run = self.get_discovery_run(run_id)
        if run:
            if run_status:
                run.run_status = run_status
            if exclusion_status:
                run.exclusion_status = exclusion_status
            self.session.flush()
        return run

    # --- Candidate Papers ---
    def add_candidate_paper(self, run_id: uuid.UUID, paper_id: uuid.UUID, score: float = 0.0, rrf_rank: Optional[int] = None, source_apis: Optional[List[str]] = None, open_access_url: Optional[str] = None, pdf_path: Optional[str] = None) -> CandidatePaper:
        candidate = CandidatePaper(
            run_id=run_id,
            paper_id=paper_id,
            score=score,
            rrf_rank=rrf_rank,
            source_apis=source_apis or [],
            open_access_url=self._clean_str(open_access_url),
            pdf_path=self._clean_str(pdf_path)
        )
        self.session.add(candidate)
        self.session.flush()
        return candidate

    def get_candidates_for_run(self, run_id: uuid.UUID) -> List[CandidatePaper]:
        return self.session.query(CandidatePaper).filter(CandidatePaper.run_id == run_id).order_by(CandidatePaper.rrf_rank.asc()).all()

    def get_candidate_by_id(self, candidate_id: uuid.UUID) -> Optional[CandidatePaper]:
        return self.session.query(CandidatePaper).filter(CandidatePaper.id == candidate_id).first()

    # --- Text Chunks ---
    def add_chunk(
        self,
        paper_id: uuid.UUID,
        run_id: uuid.UUID,
        chunk_type: str,
        content: str,
        chunk_index: int,
        token_count: int,
        parent_id: Optional[uuid.UUID] = None,
        embedding: Optional[List[float]] = None,
        section_title: Optional[str] = None,
        page_number: Optional[int] = None
    ) -> TextChunk:
        clean_content = self._clean_str(content)
        chunk = TextChunk(
            paper_id=paper_id,
            run_id=run_id,
            chunk_type=chunk_type,
            parent_id=parent_id,
            content=clean_content,
            chunk_index=chunk_index,
            token_count=token_count,
            section_title=section_title,
            page_number=page_number,
            embedding=embedding
        )
        self.session.add(chunk)
        self.session.flush()
        return chunk

    def get_chunks_for_run(self, run_id: uuid.UUID, chunk_type: Optional[str] = None) -> List[TextChunk]:
        query = self.session.query(TextChunk).filter(TextChunk.run_id == run_id)
        if chunk_type:
            query = query.filter(TextChunk.chunk_type == chunk_type)
        return query.order_by(TextChunk.chunk_index.asc()).all()

    def get_chunks_for_paper(self, paper_id: uuid.UUID, run_id: uuid.UUID, chunk_type: Optional[str] = None) -> List[TextChunk]:
        query = self.session.query(TextChunk).filter(TextChunk.paper_id == paper_id, TextChunk.run_id == run_id)
        if chunk_type:
            query = query.filter(TextChunk.chunk_type == chunk_type)
        return query.order_by(TextChunk.chunk_index.asc()).all()

    def search_dense_vector(
        self,
        run_id: uuid.UUID,
        paper_id: uuid.UUID,
        query_embedding: List[float],
        top_k: int = 30
    ) -> List[TextChunk]:
        # Perform cosine distance vector search using pgvector operator (<=>)
        # Fallback to python distance calculation if not pgvector engine
        try:
            stmt = select(TextChunk).where(
                TextChunk.run_id == run_id,
                TextChunk.paper_id == paper_id,
                TextChunk.chunk_type == "child",
                TextChunk.embedding.isnot(None)
            ).order_by(TextChunk.embedding.cosine_distance(query_embedding)).limit(top_k)
            return self.session.scalars(stmt).all()
        except Exception:
            # Python fallback if pgvector operator fails (e.g. SQLite test mode)
            chunks = self.get_chunks_for_paper(paper_id, run_id, chunk_type="child")
            # Filter chunks with embeddings
            valid_chunks = [c for c in chunks if c.embedding is not None]
            if not valid_chunks:
                return []
            import numpy as np
            q_vec = np.array(query_embedding)
            def sim(c):
                c_vec = np.array(c.embedding)
                denom = (np.linalg.norm(q_vec) * np.linalg.norm(c_vec))
                return float(np.dot(q_vec, c_vec) / denom) if denom > 0 else 0.0
            valid_chunks.sort(key=sim, reverse=True)
            return valid_chunks[:top_k]

    # --- Technical Summaries ---
    def add_summary(
        self,
        run_id: uuid.UUID,
        candidate_paper_id: uuid.UUID,
        problem_formulation: str,
        methodological_novelty: str,
        empirical_findings: str,
        paragraph_summary: str,
        model_name: str,
        data_availability: str,
        relationship_to_target: Optional[str] = None,
        data_location: Optional[str] = None,
        data_availability_assessment: Optional[Dict[str, Any]] = None
    ) -> TechnicalSummaryModel:
        summary = TechnicalSummaryModel(
            run_id=run_id,
            candidate_paper_id=candidate_paper_id,
            problem_formulation=problem_formulation,
            methodological_novelty=methodological_novelty,
            empirical_findings=empirical_findings,
            paragraph_summary=paragraph_summary,
            relationship_to_target=relationship_to_target,
            data_availability=data_availability,
            data_location=data_location,
            data_availability_assessment=data_availability_assessment,
            model_name=model_name
        )
        self.session.add(summary)
        self.session.flush()
        return summary

    def get_summaries_for_run(self, run_id: uuid.UUID) -> List[TechnicalSummaryModel]:
        return self.session.query(TechnicalSummaryModel).filter(TechnicalSummaryModel.run_id == run_id).all()
