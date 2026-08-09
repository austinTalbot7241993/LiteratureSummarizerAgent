import uuid
from datetime import datetime
from typing import List, Optional, Any
from sqlalchemy import (
    Column, String, Text, Integer, Float, Boolean, DateTime, ForeignKey, text, JSON
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from sqlalchemy.orm import declarative_base, relationship
from pgvector.sqlalchemy import Vector

Base = declarative_base()

# Cross-dialect JSONB / JSON type
JSONType = JSON().with_variant(JSONB, "postgresql")

class Paper(Base):
    __tablename__ = "papers"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    sha256_hash = Column(String(64), unique=True, nullable=False)
    title = Column(Text, nullable=False)
    authors = Column(JSONType, default=list)
    doi = Column(String(255), index=True, nullable=True)
    arxiv_id = Column(String(255), index=True, nullable=True)
    openalex_id = Column(String(255), index=True, nullable=True)
    s2_id = Column(String(255), index=True, nullable=True)
    publication_year = Column(Integer, nullable=True)
    venue = Column(Text, nullable=True)
    abstract = Column(Text, nullable=True)
    pdf_path = Column(Text, nullable=True)
    is_open_access = Column(Boolean, default=False)
    oa_pdf_url = Column(Text, nullable=True)
    raw_bibtex = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    references = relationship("PaperReference", foreign_keys="[PaperReference.source_paper_id]", back_populates="source_paper", cascade="all, delete-orphan")
    discovery_runs = relationship("DiscoveryRun", back_populates="input_paper", cascade="all, delete-orphan")


class PaperReference(Base):
    __tablename__ = "paper_references"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_paper_id = Column(PG_UUID(as_uuid=True), ForeignKey("papers.id", ondelete="CASCADE"), nullable=False)
    raw_citation = Column(Text, nullable=True)
    title = Column(Text, nullable=True)
    authors = Column(JSONType, default=list)
    doi = Column(String(255), index=True, nullable=True)
    arxiv_id = Column(String(255), index=True, nullable=True)
    openalex_id = Column(String(255), nullable=True)
    s2_id = Column(String(255), nullable=True)
    year = Column(Integer, nullable=True)
    target_paper_id = Column(PG_UUID(as_uuid=True), ForeignKey("papers.id", ondelete="SET NULL"), nullable=True)
    extraction_method = Column(String(50), nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    source_paper = relationship("Paper", foreign_keys=[source_paper_id], back_populates="references")


class DiscoveryRun(Base):
    __tablename__ = "discovery_runs"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    input_paper_id = Column(PG_UUID(as_uuid=True), ForeignKey("papers.id", ondelete="CASCADE"), nullable=False)
    run_status = Column(String(50), nullable=False, default="initialized")
    exclusion_status = Column(String(50), nullable=False, default="complete")
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    input_paper = relationship("Paper", back_populates="discovery_runs")
    candidates = relationship("CandidatePaper", back_populates="run", cascade="all, delete-orphan")
    chunks = relationship("TextChunk", back_populates="run", cascade="all, delete-orphan")
    summaries = relationship("TechnicalSummaryModel", back_populates="run", cascade="all, delete-orphan")


class CandidatePaper(Base):
    __tablename__ = "candidate_papers"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id = Column(PG_UUID(as_uuid=True), ForeignKey("discovery_runs.id", ondelete="CASCADE"), nullable=False)
    paper_id = Column(PG_UUID(as_uuid=True), ForeignKey("papers.id", ondelete="CASCADE"), nullable=False)
    score = Column(Float, default=0.0)
    rrf_rank = Column(Integer, nullable=True)
    source_apis = Column(JSONType, default=list)
    open_access_url = Column(Text, nullable=True)
    pdf_path = Column(Text, nullable=True)
    is_downloaded = Column(Boolean, default=False)
    abstract_relevance_score = Column(Float, nullable=True)
    abstract_relevance_tier = Column(String(20), nullable=True)
    abstract_relevance_reasoning = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    run = relationship("DiscoveryRun", back_populates="candidates")
    paper = relationship("Paper")
    summary = relationship("TechnicalSummaryModel", back_populates="candidate_paper", uselist=False, cascade="all, delete-orphan")


class TextChunk(Base):
    __tablename__ = "text_chunks"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    paper_id = Column(PG_UUID(as_uuid=True), ForeignKey("papers.id", ondelete="CASCADE"), nullable=False)
    run_id = Column(PG_UUID(as_uuid=True), ForeignKey("discovery_runs.id", ondelete="CASCADE"), nullable=False)
    chunk_type = Column(String(20), nullable=False)
    parent_id = Column(PG_UUID(as_uuid=True), ForeignKey("text_chunks.id", ondelete="CASCADE"), nullable=True)
    content = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    token_count = Column(Integer, nullable=False)
    section_title = Column(Text, nullable=True)
    page_number = Column(Integer, nullable=True)
    embedding = Column(Vector(1024), nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    run = relationship("DiscoveryRun", back_populates="chunks")
    paper = relationship("Paper")


class TechnicalSummaryModel(Base):
    __tablename__ = "technical_summaries"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id = Column(PG_UUID(as_uuid=True), ForeignKey("discovery_runs.id", ondelete="CASCADE"), nullable=False)
    candidate_paper_id = Column(PG_UUID(as_uuid=True), ForeignKey("candidate_papers.id", ondelete="CASCADE"), nullable=False)
    problem_formulation = Column(Text, nullable=False)
    methodological_novelty = Column(Text, nullable=False)
    empirical_findings = Column(Text, nullable=False)
    paragraph_summary = Column(Text, nullable=False)
    relationship_to_target = Column(Text, nullable=True)
    data_availability = Column(String(50), nullable=False)
    data_location = Column(Text, nullable=True)
    data_availability_assessment = Column(JSONType, nullable=True)
    model_name = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    run = relationship("DiscoveryRun", back_populates="summaries")
    candidate_paper = relationship("CandidatePaper", back_populates="summary")

