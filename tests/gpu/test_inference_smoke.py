import pytest
from lea.llm.backends import TransformersPeftBackend

def test_gpu_inference_smoke():
    try:
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA GPU not available")
    except ImportError:
        pytest.skip("PyTorch not installed")

    backend = TransformersPeftBackend(model_name="Qwen/Qwen2.5-7B-Instruct")
    summary = backend.generate_summary(
        system_prompt="Output JSON TechnicalSummary",
        user_prompt="Paper Title: Sample\nRetrieved Context: Test context."
    )
    assert len(summary.paragraph_summary) > 0
