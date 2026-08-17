import pytest
from lea.llm.backends import TransformersPeftBackend
from lea.rag.prompts import SUMMARY_SYSTEM_PROMPT, SUMMARY_USER_PROMPT_TEMPLATE

pytestmark = pytest.mark.gpu

def test_gpu_inference_smoke():
    backend = TransformersPeftBackend(model_name="Qwen/Qwen2.5-7B-Instruct")
    user_prompt = SUMMARY_USER_PROMPT_TEMPLATE.format(
        title="Sample Paper",
        authors="Sample Author",
        year=2024,
        target_title="Target Input Paper",
        target_abstract="This is a sample abstract describing the target paper's topic.",
        context_text="This paper introduces a novel approach for literature exploration."
    )
    summary = backend.generate_summary(
        system_prompt=SUMMARY_SYSTEM_PROMPT,
        user_prompt=user_prompt
    )
    assert len(summary.paragraph_summary) > 0

