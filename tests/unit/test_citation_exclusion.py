import pytest
from lea.discovery.exclusion import ExclusionEngine
from lea.exceptions import ExclusionViolationError

def test_citation_exclusion_filters_cited_and_self():
    input_paper = {"title": "Target Transformer Work", "doi": "10.1000/1"}
    cited_refs = [
        {"title": "Attention Is All You Need", "doi": "10.1000/2"},
        {"title": "BERT Pre-training of Deep Bidirectional Transformers", "arxiv_id": "1810.04805"}
    ]

    candidates = [
        {"title": "Target Transformer Work", "doi": "10.1000/1"}, # Self
        {"title": "Attention Is All You Need", "doi": "10.1000/2"}, # Cited
        {"title": "RoBERTa Robustly Optimized BERT Pretraining Approach", "doi": "10.1000/3"} # Valid candidate
    ]

    engine = ExclusionEngine(input_paper_meta=input_paper, cited_references=cited_refs, exclusion_status="complete")
    filtered = engine.filter_candidates(candidates)

    assert len(filtered) == 1
    assert filtered[0]["title"] == "RoBERTa Robustly Optimized BERT Pretraining Approach"

def test_citation_exclusion_fails_closed_when_incomplete():
    input_paper = {"title": "Target Paper"}
    engine = ExclusionEngine(
        input_paper_meta=input_paper,
        cited_references=[],
        exclusion_status="incomplete",
        allow_incomplete_citation_exclusion=False
    )

    with pytest.raises(ExclusionViolationError):
        engine.filter_candidates([{"title": "Some Candidate"}])
