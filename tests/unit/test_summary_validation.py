import pytest
from lea.llm.schemas import TechnicalSummary

def test_valid_technical_summary():
    summary = TechnicalSummary(
        problem_formulation="Statistical estimation of sparse matrices",
        methodological_novelty="First algorithm with O(n log n) complexity guarantee",
        empirical_findings="Tested on 5 benchmarks showing 3x speedup",
        paragraph_summary="This paper presents a novel algorithmic solution to sparse matrix estimation."
    )
    assert summary.paragraph_summary.startswith("This paper")

def test_summary_fails_if_multiple_paragraphs():
    with pytest.raises(ValueError, match="single paragraph"):
        TechnicalSummary(
            problem_formulation="P",
            methodological_novelty="M",
            empirical_findings="E",
            paragraph_summary="First paragraph.\n\nSecond paragraph."
        )

def test_summary_fails_if_exceeds_word_limit():
    long_text = "word " * 305
    with pytest.raises(ValueError, match="at most 300 words"):
        TechnicalSummary(
            problem_formulation="P",
            methodological_novelty="M",
            empirical_findings="E",
            paragraph_summary=long_text
        )
