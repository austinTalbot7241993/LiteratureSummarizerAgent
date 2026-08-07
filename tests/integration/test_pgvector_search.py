import pytest
import numpy as np
from lea.db.session import init_db, create_tables, get_db_session
from lea.db.repository import LEARepository

@pytest.fixture
def test_db_session(tmp_path):
    db_file = tmp_path / "test_vector.db"
    db_url = f"sqlite:///{db_file}"
    engine = init_db(db_url)
    create_tables(engine)
    with get_db_session() as session:
        yield session

def test_dense_vector_search(test_db_session):
    repo = LEARepository(test_db_session)
    paper = repo.create_paper(sha256_hash="vec_hash", title="Vector Paper")
    run = repo.create_discovery_run(input_paper_id=paper.id)

    # Generate dummy normalized embeddings
    v1 = (np.ones(1024) / np.sqrt(1024)).tolist()
    v2 = (-np.ones(1024) / np.sqrt(1024)).tolist()

    chunk1 = repo.add_chunk(paper_id=paper.id, run_id=run.id, chunk_type="child", content="Positive vector chunk", chunk_index=1, token_count=10, embedding=v1)
    chunk2 = repo.add_chunk(paper_id=paper.id, run_id=run.id, chunk_type="child", content="Negative vector chunk", chunk_index=2, token_count=10, embedding=v2)

    results = repo.search_dense_vector(run.id, paper.id, v1, top_k=2)
    assert len(results) == 2
    assert results[0].content == "Positive vector chunk"
