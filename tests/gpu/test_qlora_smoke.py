import pytest
from lea.llm.qlora_trainer import train_qlora

def test_gpu_qlora_smoke(tmp_path):
    try:
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA GPU not available")
    except ImportError:
        pytest.skip("PyTorch not installed")

    out_dir = str(tmp_path / "adapters")
    dataset = str(tmp_path / "train.jsonl")
    with open(dataset, "w") as f:
        f.write('{"text": "Sample training text for QLoRA"}\n')

    train_qlora(dataset_path=dataset, output_dir=out_dir)
