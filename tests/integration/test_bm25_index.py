import pytest
from lea.rag.bm25_index import BM25Index

def test_bm25_indexing_and_search():
    chunks = [
        {"id": "c1", "content": "Convolutional Neural Networks for computer vision image recognition."},
        {"id": "c2", "content": "Transformer self-attention models for natural language processing and translation."},
        {"id": "c3", "content": "Reinforcement learning for policy optimization in game playing environments."}
    ]

    index = BM25Index()
    index.index_chunks(chunks)

    results = index.search("transformer self-attention", top_k=2)
    assert len(results) > 0
    top_chunk, score = results[0]
    assert top_chunk["id"] == "c2"
