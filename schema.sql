CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS papers (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    sha256_hash VARCHAR(64) UNIQUE NOT NULL,
    title TEXT NOT NULL,
    authors JSONB DEFAULT '[]'::jsonb,
    doi VARCHAR(255),
    arxiv_id VARCHAR(255),
    openalex_id VARCHAR(255),
    s2_id VARCHAR(255),
    publication_year INT,
    venue TEXT,
    abstract TEXT,
    pdf_path TEXT,
    is_open_access BOOLEAN DEFAULT FALSE,
    oa_pdf_url TEXT,
    raw_bibtex TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_papers_doi ON papers (doi);
CREATE INDEX IF NOT EXISTS idx_papers_arxiv_id ON papers (arxiv_id);
CREATE INDEX IF NOT EXISTS idx_papers_openalex_id ON papers (openalex_id);
CREATE INDEX IF NOT EXISTS idx_papers_s2_id ON papers (s2_id);

CREATE TABLE IF NOT EXISTS paper_references (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_paper_id UUID NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    raw_citation TEXT,
    title TEXT,
    authors JSONB DEFAULT '[]'::jsonb,
    doi VARCHAR(255),
    arxiv_id VARCHAR(255),
    openalex_id VARCHAR(255),
    s2_id VARCHAR(255),
    year INT,
    target_paper_id UUID REFERENCES papers(id) ON DELETE SET NULL,
    extraction_method VARCHAR(50) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_references_source_paper ON paper_references (source_paper_id);
CREATE INDEX IF NOT EXISTS idx_references_doi ON paper_references (doi);
CREATE INDEX IF NOT EXISTS idx_references_arxiv_id ON paper_references (arxiv_id);

CREATE TABLE IF NOT EXISTS discovery_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    input_paper_id UUID NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    run_status VARCHAR(50) NOT NULL DEFAULT 'initialized',
    exclusion_status VARCHAR(50) NOT NULL DEFAULT 'complete',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS candidate_papers (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id UUID NOT NULL REFERENCES discovery_runs(id) ON DELETE CASCADE,
    paper_id UUID NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    score FLOAT DEFAULT 0.0,
    rrf_rank INT,
    source_apis JSONB DEFAULT '[]'::jsonb,
    open_access_url TEXT,
    pdf_path TEXT,
    is_downloaded BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_candidates_run_id ON candidate_papers (run_id);

CREATE TABLE IF NOT EXISTS text_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    paper_id UUID NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    run_id UUID NOT NULL REFERENCES discovery_runs(id) ON DELETE CASCADE,
    chunk_type VARCHAR(20) NOT NULL, -- 'parent' or 'child'
    parent_id UUID REFERENCES text_chunks(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    chunk_index INT NOT NULL,
    token_count INT NOT NULL,
    embedding vector(1024),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_chunks_run_id ON text_chunks (run_id);
CREATE INDEX IF NOT EXISTS idx_chunks_paper_id ON text_chunks (paper_id);

CREATE TABLE IF NOT EXISTS technical_summaries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id UUID NOT NULL REFERENCES discovery_runs(id) ON DELETE CASCADE,
    candidate_paper_id UUID NOT NULL REFERENCES candidate_papers(id) ON DELETE CASCADE,
    problem_formulation TEXT NOT NULL,
    methodological_novelty TEXT NOT NULL,
    empirical_findings TEXT NOT NULL,
    paragraph_summary TEXT NOT NULL,
    model_name VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_summaries_run_id ON technical_summaries (run_id);
