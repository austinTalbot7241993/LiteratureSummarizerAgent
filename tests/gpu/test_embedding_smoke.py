import pytest
from lea.rag.embedder import BGEEmbedder

def test_gpu_embedding_smoke():
    try:
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA GPU not available")
    except ImportError:
        pytest.skip("PyTorch not installed")

    embedder = BGEEmbedder(model_name="BAAI/bge-m3", use_subprocess=True)
    vecs = embedder.embed_texts(["Test paper query for GPU smoke test."])
    assert len(vecs) == 1
    assert len(vecs[0]) == 1024
