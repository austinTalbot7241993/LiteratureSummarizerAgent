import sqlite3
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from lea.db.session import create_tables
from lea.db.models import TechnicalSummaryModel, Base
from lea.db.repository import LEARepository
from lea.llm.schemas import TechnicalSummary, PaperAvailabilityStatus


def test_case22_legacy_proprietary_db_rows_migrate_to_unclear(tmp_path):
    db_path = tmp_path / "legacy_test.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create legacy table schema with data_availability column containing 'proprietary'
    cursor.execute("""
        CREATE TABLE technical_summaries (
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            candidate_paper_id TEXT NOT NULL,
            problem_formulation TEXT NOT NULL,
            methodological_novelty TEXT NOT NULL,
            empirical_findings TEXT NOT NULL,
            paragraph_summary TEXT NOT NULL,
            data_availability TEXT DEFAULT 'proprietary',
            data_location TEXT,
            model_name TEXT NOT NULL,
            created_at TEXT
        );
    """)
    cursor.execute("""
        INSERT INTO technical_summaries (id, run_id, candidate_paper_id, problem_formulation, methodological_novelty, empirical_findings, paragraph_summary, data_availability, model_name)
        VALUES ('sum-1', 'run-1', 'cand-1', 'P', 'M', 'E', 'S', 'proprietary', 'test-model');
    """)
    cursor.execute("""
        INSERT INTO technical_summaries (id, run_id, candidate_paper_id, problem_formulation, methodological_novelty, empirical_findings, paragraph_summary, data_availability, model_name)
        VALUES ('sum-2', 'run-1', 'cand-2', 'P', 'M', 'E', 'S', 'publicly_available', 'test-model');
    """)
    conn.commit()
    conn.close()

    # Run create_tables on legacy DB engine
    engine = create_engine(f"sqlite:///{db_path}")
    create_tables(engine)

    with engine.connect() as check_conn:
        res1 = check_conn.execute(text("SELECT data_availability FROM technical_summaries WHERE id = 'sum-1';")).scalar()
        res2 = check_conn.execute(text("SELECT data_availability FROM technical_summaries WHERE id = 'sum-2';")).scalar()

        # Legacy 'proprietary' migrated to 'unclear'
        assert res1 == "unclear"
        # Legacy 'publicly_available' preserved
        assert res2 == "publicly_available"


def test_case23_no_substantive_proprietary_defaults():
    # TechnicalSummary schema requires explicit status
    with pytest.raises(ValueError):
        TechnicalSummary(
            problem_formulation="P",
            methodological_novelty="M",
            empirical_findings="E",
            paragraph_summary="S"
        )

    # Database model TechnicalSummaryModel column has no default='proprietary'
    col_default = TechnicalSummaryModel.data_availability.property.columns[0].default
    assert col_default is None, "data_availability Column must not have a default='proprietary'"


def test_missing_text_chunks_columns_auto_migrate(tmp_path):
    import uuid
    from lea.db.session import init_db, get_db_session
    db_path = tmp_path / "legacy_chunks.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE papers (
            id TEXT PRIMARY KEY,
            sha256_hash TEXT UNIQUE NOT NULL,
            title TEXT NOT NULL,
            authors TEXT,
            doi TEXT,
            arxiv_id TEXT,
            openalex_id TEXT,
            s2_id TEXT,
            publication_year INTEGER,
            venue TEXT,
            abstract TEXT,
            pdf_path TEXT,
            is_open_access INTEGER,
            oa_pdf_url TEXT,
            raw_bibtex TEXT,
            created_at TEXT,
            updated_at TEXT
        );
    """)
    cursor.execute("""
        CREATE TABLE discovery_runs (
            id TEXT PRIMARY KEY,
            input_paper_id TEXT NOT NULL,
            run_status TEXT NOT NULL,
            exclusion_status TEXT NOT NULL,
            created_at TEXT
        );
    """)
    # Legacy text_chunks table without section_title or page_number
    cursor.execute("""
        CREATE TABLE text_chunks (
            id TEXT PRIMARY KEY,
            paper_id TEXT NOT NULL,
            run_id TEXT NOT NULL,
            chunk_type TEXT NOT NULL,
            parent_id TEXT,
            content TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            token_count INTEGER NOT NULL,
            embedding BLOB,
            created_at TEXT
        );
    """)
    conn.commit()
    conn.close()

    # Initialize DB connection
    db_url = f"sqlite:///{db_path}"
    engine = init_db(db_url)

    with get_db_session() as session:
        repo = LEARepository(session)
        p = repo.create_paper(sha256_hash="hash_c", title="Paper C")
        r = repo.create_discovery_run(input_paper_id=p.id)
        chunk = repo.add_chunk(
            paper_id=p.id,
            run_id=r.id,
            chunk_type="child",
            content="Sample text content.",
            chunk_index=0,
            token_count=5,
            section_title="Data Availability",
            page_number=4
        )
        assert chunk.id is not None
        assert chunk.section_title == "Data Availability"
        assert chunk.page_number == 4


def test_missing_candidate_papers_columns_auto_migrate(tmp_path):
    from lea.db.session import init_db, get_db_session
    db_path = tmp_path / "legacy_candidates.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE papers (
            id TEXT PRIMARY KEY,
            sha256_hash TEXT UNIQUE NOT NULL,
            title TEXT NOT NULL,
            authors TEXT,
            doi TEXT,
            arxiv_id TEXT,
            openalex_id TEXT,
            s2_id TEXT,
            publication_year INTEGER,
            venue TEXT,
            abstract TEXT,
            pdf_path TEXT,
            is_open_access INTEGER,
            oa_pdf_url TEXT,
            raw_bibtex TEXT,
            created_at TEXT,
            updated_at TEXT
        );
    """)
    cursor.execute("""
        CREATE TABLE discovery_runs (
            id TEXT PRIMARY KEY,
            input_paper_id TEXT NOT NULL,
            run_status TEXT NOT NULL,
            exclusion_status TEXT NOT NULL,
            created_at TEXT
        );
    """)
    # Legacy candidate_papers table without abstract_relevance columns
    cursor.execute("""
        CREATE TABLE candidate_papers (
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            paper_id TEXT NOT NULL,
            score REAL,
            rrf_rank INTEGER,
            source_apis TEXT,
            open_access_url TEXT,
            pdf_path TEXT,
            is_downloaded INTEGER,
            created_at TEXT
        );
    """)
    conn.commit()
    conn.close()

    # Initialize DB connection and run auto-migration
    db_url = f"sqlite:///{db_path}"
    engine = init_db(db_url)

    with get_db_session() as session:
        repo = LEARepository(session)
        p1 = repo.create_paper(sha256_hash="hash_p1", title="Paper 1")
        p2 = repo.create_paper(sha256_hash="hash_p2", title="Paper 2")
        r = repo.create_discovery_run(input_paper_id=p1.id)

        cand = repo.add_candidate_paper(
            run_id=r.id,
            paper_id=p2.id,
            score=0.8,
            rrf_rank=1,
            abstract_relevance_score=8.5,
            abstract_relevance_tier="high",
            abstract_relevance_reasoning="Highly relevant abstract."
        )

        assert cand.id is not None
        assert cand.abstract_relevance_score == 8.5
        assert cand.abstract_relevance_tier == "high"
        assert cand.abstract_relevance_reasoning == "Highly relevant abstract."

