import pytest
from typing import List, Dict, Any
from lea.llm.inference import TechnicalSummarizer
from lea.llm.backends import BaseLLMBackend
from lea.llm.schemas import TechnicalSummary
from lea.rag.prompts import SUMMARY_USER_PROMPT_TEMPLATE
from lea.exceptions import SummaryValidationError

class AlwaysFailingCritiqueBackend(BaseLLMBackend):
    def generate_summary(self, system_prompt: str, user_prompt: str) -> TechnicalSummary:
        return TechnicalSummary(
            problem_formulation="Valid problem statement.",
            methodological_novelty="Valid methodological novelty.",
            empirical_findings="Valid empirical findings.",
            paragraph_summary="Valid single paragraph summary.",
            data_availability="publicly_available"
        )

    def generate_data_availability(self, system_prompt: str, user_prompt: str):
        from lea.llm.schemas import DataAvailabilityAssessment, PaperAvailabilityStatus
        return DataAvailabilityAssessment(
            overall_status=PaperAvailabilityStatus.NOT_REPORTED,
            datasets=[],
            rationale="Not reported"
        )

    def generate_critique(self, system_prompt: str, user_prompt: str):
        raise SummaryValidationError("Simulated permanent critique failure (e.g. malformed JSON)")

class FailingFirstAttemptBackend(BaseLLMBackend):
    def __init__(self):
        self.call_history = []

    def generate_summary(self, system_prompt: str, user_prompt: str) -> TechnicalSummary:
        self.call_history.append((system_prompt, user_prompt))
        if len(self.call_history) == 1:
            raise SummaryValidationError("Simulated attempt 1 failure: missing section headers")
        return TechnicalSummary(
            problem_formulation="Valid problem statement.",
            methodological_novelty="Valid methodological novelty.",
            empirical_findings="Valid empirical findings.",
            paragraph_summary="Valid single paragraph summary.",
            data_availability="publicly_available"
        )

    def generate_data_availability(self, system_prompt: str, user_prompt: str):
        from lea.llm.schemas import DataAvailabilityAssessment, PaperAvailabilityStatus
        return DataAvailabilityAssessment(
            overall_status=PaperAvailabilityStatus.NOT_REPORTED,
            datasets=[],
            rationale="Not reported"
        )

def test_user_prompt_template_places_directives_before_context():
    formatted = SUMMARY_USER_PROMPT_TEMPLATE.format(
        title="Test Title",
        authors="Test Author",
        year="2024",
        target_title="Target Input Paper",
        target_abstract="Sample target abstract",
        context_text="Sample context"
    )
    format_idx = formatted.index("REQUIRED FORMAT:")
    context_idx = formatted.index("Retrieved Context Chunks:")
    assert format_idx < context_idx, "REQUIRED FORMAT directive must appear BEFORE context chunks"
    assert "PROBLEM FORMULATION:" in formatted
    assert "METHODOLOGICAL NOVELTY:" in formatted
    assert "EMPIRICAL FINDINGS:" in formatted
    assert "TECHNICAL SYNTHESIS:" in formatted
    assert "RELATIONSHIP TO TARGET PAPER:" in formatted

def test_summarizer_retry_reduces_context_and_adds_directive():
    backend = FailingFirstAttemptBackend()
    summarizer = TechnicalSummarizer(backend=backend, max_attempts=2)

    chunks = [{"content": f"Chunk content {i}"} for i in range(10)]
    candidate_meta = {"title": "Paper on Scalable Sequencing", "authors": ["Alice"], "year": 2024}

    summary, assessment = summarizer.summarize_candidate(candidate_meta, chunks)

    assert summary.paragraph_summary == "Valid single paragraph summary."
    assert len(backend.call_history) == 2

    first_user_prompt = backend.call_history[0][1]
    second_user_prompt = backend.call_history[1][1]

    # Verification: Attempt 2 must add retry directive and reduce chunks
    assert "CRITICAL RETRY DIRECTIVE" in second_user_prompt
    assert "Chunk content 9" in first_user_prompt
    assert "Chunk content 9" not in second_user_prompt  # Context was truncated on retry

def test_summarizer_handles_extremely_large_context():
    backend = FailingFirstAttemptBackend()
    summarizer = TechnicalSummarizer(backend=backend, max_attempts=2)

    # 50 large context chunks
    large_chunks = [{"content": "Word " * 1000} for _ in range(50)]
    candidate_meta = {"title": "Large Context Paper", "authors": ["Bob"], "year": 2024}

    summary, assessment = summarizer.summarize_candidate(candidate_meta, large_chunks)
    assert summary is not None

def test_backends_module_imports_gc_and_os():
    import lea.llm.backends as backends_mod
    assert hasattr(backends_mod, "gc")
    assert hasattr(backends_mod, "os")

class AlwaysFailingAvailBackend(BaseLLMBackend):
    def generate_summary(self, system_prompt: str, user_prompt: str) -> TechnicalSummary:
        return TechnicalSummary(
            problem_formulation="Valid problem statement.",
            methodological_novelty="Valid methodological novelty.",
            empirical_findings="Valid empirical findings.",
            paragraph_summary="Valid single paragraph summary.",
            data_availability="publicly_available"
        )

    def generate_data_availability(self, system_prompt: str, user_prompt: str):
        raise SummaryValidationError("Simulated permanent validation failure")

def test_extract_data_availability_fallback_on_failure():
    backend = AlwaysFailingAvailBackend()
    summarizer = TechnicalSummarizer(backend=backend, max_attempts=2)
    candidate_meta = {"title": "Failing Avail Paper", "authors": ["Charlie"], "year": 2024}

    assessment = summarizer.extract_data_availability(candidate_meta, [{"content": "sample text"}])
    from lea.llm.schemas import PaperAvailabilityStatus
    assert assessment.overall_status == PaperAvailabilityStatus.NOT_REPORTED
    assert "Defaulted to not_reported" in assessment.rationale


def test_critique_candidate_fails_safe_not_open_on_repeated_failure():
    """Regression test: when the self-critique LLM call fails on every retry
    attempt, critique_candidate() must default to REJECTING the candidate, not
    fabricating a passing score. The old fallback returned
    (is_relevant_to_seed_topic=True, verdict="marginal", 6.0/6.0), which meant a
    candidate with zero real relevance judgment could still pass
    IterativeSourceManager's acceptance threshold in quota_manager.py --
    exactly what let an irrelevant paper (about fusion-reactor camera
    shielding) get accepted into a real run against a genomics paper, purely
    because its critique call happened to fail to parse.
    """
    backend = AlwaysFailingCritiqueBackend()
    summarizer = TechnicalSummarizer(backend=backend, max_attempts=2)

    candidate_meta = {"title": "Unrelated Candidate Paper", "authors": ["Someone"], "year": 2024}
    summary, _ = summarizer.summarize_candidate(candidate_meta, [{"content": "sample text"}])

    critique = summarizer.critique_candidate(
        candidate_meta,
        summary,
        [{"content": "sample text"}],
        target_paper_meta={"title": "Seed Paper", "abstract": "Seed paper abstract."}
    )

    assert critique.is_relevant_to_seed_topic is False
    assert critique.verdict == "rejected"
    assert critique.relevance_score == 0.0
    assert critique.factual_grounding_score == 0.0

