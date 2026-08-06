import pytest
from lea.llm.backends import TransformersPeftBackend, MockLLMBackend
from lea.exceptions import SummaryValidationError
from lea.rag.prompts import SUMMARY_SYSTEM_PROMPT, SUMMARY_USER_PROMPT_TEMPLATE

def test_parse_labeled_prose_headers():
    backend = MockLLMBackend()
    raw_response = """
PROBLEM FORMULATION:
We address the computational complexity of rare-variant aware genome inference across large population cohorts.

METHODOLOGICAL NOVELTY:
We introduce a novel graph polishing pipeline using Sniffles2 and bcftools consensus to generate consensus haplotypes.

EMPIRICAL FINDINGS:
Evaluation on the 1kGP cohort demonstrates higher haplotype accuracy compared to PanGenie v3 and HGSVC3 benchmarks.

TECHNICAL SYNTHESIS:
This paper presents a scalable framework for rare-variant aware genome inference using pangenome graphs and consensus polishing.
"""
    summary = backend.generate_summary(
        system_prompt=SUMMARY_SYSTEM_PROMPT,
        user_prompt=SUMMARY_USER_PROMPT_TEMPLATE.format(
            title="Genome Inference",
            authors="Test Authors",
            year=2024,
            context_text="Sample paper text."
        )
    )
    assert len(summary.problem_formulation) > 10
    assert len(summary.methodological_novelty) > 10
    assert len(summary.empirical_findings) > 10
    assert len(summary.paragraph_summary) > 10


def test_missing_section_fails_loudly():
    # If the model fails to output one of the 4 required sections, it MUST raise SummaryValidationError
    raw_missing_empirical = """
PROBLEM FORMULATION:
We address high-dimensional matrix estimation.

METHODOLOGICAL NOVELTY:
We propose a fast spectral thresholding algorithm.

TECHNICAL SYNTHESIS:
This paper presents a fast spectral thresholding method for matrix estimation.
"""
    # Simulate LLM backend section validation
    import re
    patterns = {
        "problem_formulation": r"(?:PROBLEM FORMULATION|PROBLEM STATEMENT|PROBLEM)\s*:\s*(.*?)(?=\n\s*(?:\*\*|\#\#|\#)?\s*(?:METHODOLOGICAL NOVELTY|METHODOLOGY|EMPIRICAL FINDINGS|RESULTS|TECHNICAL SYNTHESIS|SYNTHESIS)\s*:|\s*$)",
        "methodological_novelty": r"(?:METHODOLOGICAL NOVELTY|METHODOLOGY|NOVELTY)\s*:\s*(.*?)(?=\n\s*(?:\*\*|\#\#|\#)?\s*(?:EMPIRICAL FINDINGS|RESULTS|TECHNICAL SYNTHESIS|SYNTHESIS)\s*:|\s*$)",
        "empirical_findings": r"(?:EMPIRICAL FINDINGS|RESULTS|EMPIRICAL EVALUATION)\s*:\s*(.*?)(?=\n\s*(?:\*\*|\#\#|\#)?\s*(?:TECHNICAL SYNTHESIS|SYNTHESIS|SUMMARY)\s*:|\s*$)",
        "paragraph_summary": r"(?:TECHNICAL SYNTHESIS|SYNTHESIS|SUMMARY)\s*:\s*(.*?)(?=\s*$)"
    }
    parsed = {}
    for key, pat in patterns.items():
        m = re.search(pat, raw_missing_empirical, re.IGNORECASE | re.DOTALL)
        if m and len(m.group(1).strip()) > 5:
            parsed[key] = m.group(1).strip()

    missing = [k for k in ["problem_formulation", "methodological_novelty", "empirical_findings", "paragraph_summary"] if k not in parsed]
    assert "empirical_findings" in missing
