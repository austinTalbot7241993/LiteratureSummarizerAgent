import pytest
from lea.rag.chunker import HierarchicalChunker

def test_hierarchical_chunking():
    chunker = HierarchicalChunker(parent_tokens=50, parent_overlap_tokens=10, child_tokens=20, child_overlap_tokens=5)
    text = "Word " * 120 # 120 words

    chunks = chunker.chunk_text(text)
    assert len(chunks) > 0

    parents = [c for c in chunks if c["chunk_type"] == "parent"]
    children = [c for c in chunks if c["chunk_type"] == "child"]

    assert len(parents) > 0
    assert len(children) > 0
    for child in children:
        assert child["parent_index"] is not None
