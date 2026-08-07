import uuid
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from lea.db.models import Base, Paper, DiscoveryRun, TextChunk
from lea.db.repository import LEARepository
from lea.rag.dense_search import DenseSearchEngine
from lea.rag.hybrid_search import HybridSearchEngine


class DummyEmbedder:
    def embed_texts(self, texts):
        return [[0.1] * 1024 for _ in texts]


def test_case16_and_17_paper_isolated_search():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    repo = LEARepository(session)

    p_input = repo.create_paper(sha256_hash="input123", title="Input Paper")
    run = repo.create_discovery_run(input_paper_id=p_input.id)

    p_a = repo.create_paper(sha256_hash="hash_a", title="Paper A")
    p_b = repo.create_paper(sha256_hash="hash_b", title="Paper B")

    # Paper A chunk: publicly available
    chunk_a = repo.add_chunk(
        paper_id=p_a.id,
        run_id=run.id,
        chunk_type="child",
        content="Paper A data are freely downloadable from https://example.org/data.",
        chunk_index=0,
        token_count=10,
        embedding=[0.9] * 1024
    )

    # Paper B chunk: cannot be shared
    chunk_b = repo.add_chunk(
        paper_id=p_b.id,
        run_id=run.id,
        chunk_type="child",
        content="Paper B participant data cannot be shared due to privacy.",
        chunk_index=0,
        token_count=10,
        embedding=[0.95] * 1024  # Higher similarity to query vector!
    )

    # 1. Direct repo.search_dense_vector for Paper A
    query_vec = [0.9] * 1024
    results_a = repo.search_dense_vector(run_id=run.id, paper_id=p_a.id, query_embedding=query_vec, top_k=10)
    
    assert len(results_a) == 1
    assert results_a[0].id == chunk_a.id
    assert results_a[0].paper_id == p_a.id
    assert "Paper A" in results_a[0].content
    assert "Paper B" not in results_a[0].content

    # 2. Search for Paper B
    results_b = repo.search_dense_vector(run_id=run.id, paper_id=p_b.id, query_embedding=query_vec, top_k=10)
    assert len(results_b) == 1
    assert results_b[0].id == chunk_b.id
    assert results_b[0].paper_id == p_b.id
    assert "Paper B" in results_b[0].content
    assert "Paper A" not in results_b[0].content

    # 3. DenseSearchEngine paper isolation
    dense_engine = DenseSearchEngine(embedder=DummyEmbedder())
    dense_out_a = dense_engine.search(repo=repo, run_id=run.id, paper_id=p_a.id, query_text="data download", top_k=10)
    assert len(dense_out_a) == 1
    assert dense_out_a[0][0]["paper_id"] == str(p_a.id)

    # 4. HybridSearchEngine paper isolation
    hybrid_engine = HybridSearchEngine(dense_engine=dense_engine)
    hybrid_out_a = hybrid_engine.hybrid_search(
        repo=repo,
        run_id=run.id,
        paper_id=p_a.id,
        query_text="data availability download"
    )
    for chunk_dict, score in hybrid_out_a:
        assert chunk_dict["paper_id"] == str(p_a.id)
