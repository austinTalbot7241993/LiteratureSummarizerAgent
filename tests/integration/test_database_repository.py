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
