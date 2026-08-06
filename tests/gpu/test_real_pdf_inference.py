from pathlib import Path
import pytest
from lea.ingester.pdf_parser import PDFParser
from lea.rag.chunker import HierarchicalChunker
from lea.rag.prompts import SUMMARY_SYSTEM_PROMPT, SUMMARY_USER_PROMPT_TEMPLATE

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "sample_real_paper.pdf"

def test_real_pdf_extraction_and_prompt_formatting():
    assert FIXTURE_PATH.exists(), "Real sample PDF fixture missing"

    pdf = PDFParser(str(FIXTURE_PATH))
    body_text = pdf.extract_body_text()
    assert len(body_text) > 500, "Failed to extract body text from real PDF"

    # Verify back-matter section stripping
    assert "s3-us-west-2.amazonaws.com" not in body_text

    chunker = HierarchicalChunker(tokenizer_model="BAAI/bge-m3")
    chunks = chunker.chunk_text(body_text)
    assert len(chunks) > 0, "Failed to chunk body text from real PDF"

    child_contents = [c["content"] for c in chunks if c["chunk_type"] == "child"]
    context_text = "\n\n".join(child_contents[:4])

    user_prompt = SUMMARY_USER_PROMPT_TEMPLATE.format(
        title="Scalable and rare-variant aware genome inference across the 1kGP cohort",
        authors="Genome Research Group",
        year=2024,
        context_text=context_text
    )

    assert "PROBLEM FORMULATION:" in SUMMARY_SYSTEM_PROMPT
    assert "METHODOLOGICAL NOVELTY:" in SUMMARY_SYSTEM_PROMPT
    assert "EMPIRICAL FINDINGS:" in SUMMARY_SYSTEM_PROMPT
    assert "TECHNICAL SYNTHESIS:" in SUMMARY_SYSTEM_PROMPT
    assert len(user_prompt) > 200

def test_real_gpu_llm_inference():
    pdf = PDFParser(str(FIXTURE_PATH))
    body_text = pdf.extract_body_text()
    chunker = HierarchicalChunker(tokenizer_model="BAAI/bge-m3")
    chunks = chunker.chunk_text(body_text)
    child_contents = [c["content"] for c in chunks if c["chunk_type"] == "child"]
    context_text = "\n\n".join(child_contents[:4])

    user_prompt = SUMMARY_USER_PROMPT_TEMPLATE.format(
        title="Scalable and rare-variant aware genome inference across the 1kGP cohort",
        authors="Genome Research Group",
        year=2024,
        context_text=context_text
    )

    from lea.llm.backends import TransformersPeftBackend
    backend = TransformersPeftBackend(model_name="Qwen/Qwen2.5-7B-Instruct")
    summary = backend.generate_summary(
        system_prompt=SUMMARY_SYSTEM_PROMPT,
        user_prompt=user_prompt
    )

    # Verify all 4 sections were generated and parsed cleanly from Qwen
    assert len(summary.problem_formulation) > 15
    assert len(summary.methodological_novelty) > 15
    assert len(summary.empirical_findings) > 15
    assert len(summary.paragraph_summary) > 15

    # Ensure zero hardcoded placeholder strings exist in summary
    assert "Formulates problem statement for" not in summary.problem_formulation
    assert "Introduces technical framework presented" not in summary.methodological_novelty
    assert "Reports empirical evaluation results as detailed" not in summary.empirical_findings
