import pytest
import uuid
from lea.db.session import init_db, create_tables, get_db_session
from lea.db.repository import LEARepository

@pytest.fixture
def test_db_session(tmp_path):
    db_file = tmp_path / "test_lea.db"
    db_url = f"sqlite:///{db_file}"
    engine = init_db(db_url)
    create_tables(engine)
    with get_db_session() as session:
        yield session

def test_paper_crud_operations(test_db_session):
    repo = LEARepository(test_db_session)
    paper = repo.create_paper(
        sha256_hash="hash123",
        title="Test Ingested Paper",
        authors=["Alice", "Bob"],
        doi="10.1000/test"
    )

    assert paper.id is not None
    fetched = repo.get_paper_by_hash("hash123")
    assert fetched.title == "Test Ingested Paper"

    ref = repo.add_reference(source_paper_id=paper.id, title="Ref Paper 1", extraction_method="grobid")
    assert ref.id is not None

    refs = repo.get_references_for_paper(paper.id)
    assert len(refs) == 1

def test_discovery_run_flow(test_db_session):
    repo = LEARepository(test_db_session)
    paper = repo.create_paper(sha256_hash="hash456", title="Input Paper")
    run = repo.create_discovery_run(input_paper_id=paper.id)

    cand_paper = repo.create_paper(sha256_hash="hash789", title="Candidate Paper")
    candidate = repo.add_candidate_paper(run_id=run.id, paper_id=cand_paper.id, score=0.95, rrf_rank=1)

    assert candidate.id is not None
    cands = repo.get_candidates_for_run(run.id)
    assert len(cands) == 1
    assert cands[0].paper.title == "Candidate Paper"

def test_existing_database_auto_migrates_missing_columns(tmp_path):
    from sqlalchemy import text, create_engine
    from sqlalchemy.orm import sessionmaker

    db_file = tmp_path / "legacy_lea.db"
    db_url = f"sqlite:///{db_file}"
    engine = create_engine(db_url)

    # 1. Create a legacy technical_summaries table missing data_availability and data_location columns
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE technical_summaries (
                id VARCHAR(36) PRIMARY KEY,
                run_id VARCHAR(36) NOT NULL,
                candidate_paper_id VARCHAR(36) NOT NULL,
                problem_formulation TEXT NOT NULL,
                methodological_novelty TEXT NOT NULL,
                empirical_findings TEXT NOT NULL,
                paragraph_summary TEXT NOT NULL,
                model_name VARCHAR(255) NOT NULL,
                created_at TIMESTAMP
            );
        """))
        conn.commit()

    # 2. Initialize DB session and run create_tables auto-sync
    init_engine = init_db(db_url)
    create_tables(init_engine)

    # 3. Verify inserting summary with data_availability succeeds without column error
    SessionLocal = sessionmaker(bind=init_engine)
    with SessionLocal() as session:
        repo = LEARepository(session)
        # Create minimal foreign key parent entries
        paper = repo.create_paper(sha256_hash="legacy_hash", title="Legacy Paper")
        run = repo.create_discovery_run(input_paper_id=paper.id)
        cand = repo.add_candidate_paper(run_id=run.id, paper_id=paper.id)

        summary = repo.add_summary(
            run_id=run.id,
            candidate_paper_id=cand.id,
            problem_formulation="Legacy problem",
            methodological_novelty="Legacy novelty",
            empirical_findings="Legacy findings",
            paragraph_summary="Legacy summary paragraph",
            data_availability="restricted",
            data_location="UK Biobank App 123",
            model_name="mock"
        )
        session.commit()

        assert summary.id is not None
        assert summary.data_availability == "restricted"
        assert summary.data_location == "UK Biobank App 123"

