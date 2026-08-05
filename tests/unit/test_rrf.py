import pytest
from lea.rag.hybrid_search import compute_rrf_fusion

def test_compute_rrf_fusion():
    c1 = {"id": "chunk_1", "content": "First chunk"}
    c2 = {"id": "chunk_2", "content": "Second chunk"}
    c3 = {"id": "chunk_3", "content": "Third chunk"}

    dense_results = [(c1, 0.9), (c2, 0.8)]
    sparse_results = [(c2, 5.0), (c3, 4.0)]

    fused = compute_rrf_fusion(dense_results, sparse_results, rrf_k=60, fused_top_k=3)
    assert len(fused) == 3
    # c2 is in both dense and sparse, so it should have the highest RRF score
    assert fused[0][0]["id"] == "chunk_2"
