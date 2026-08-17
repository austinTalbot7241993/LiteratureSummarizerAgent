import pytest
import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock
from lea.llm.quota_manager import IterativeSourceManager
from lea.discovery.graph_expansion import SecondaryGraphExpander
from lea.llm.schemas import TechnicalSummary, SelfCritiqueAssessment, DataAvailabilityAssessment, PaperAvailabilityStatus
from lea.config import LEAConfig


@pytest.mark.asyncio
async def test_quota_loop_reaches_target():
    # Mock summarizer that accepts every candidate
    mock_summarizer = MagicMock()
    tech_summary = TechnicalSummary(
        problem_formulation="Formulation",
        methodological_novelty="Novelty",
        empirical_findings="Findings",
        paragraph_summary="Summary text.",
        data_availability=PaperAvailabilityStatus.PUBLICLY_AVAILABLE,
        critique=SelfCritiqueAssessment(
            is_relevant_to_seed_topic=True,
            relevance_score=8.5,
            factual_grounding_score=9.0,
            critique_rationale="Accepted",
            verdict="accepted"
        )
    )
    assessment = DataAvailabilityAssessment(
        overall_status=PaperAvailabilityStatus.PUBLICLY_AVAILABLE,
        datasets=[],
        rationale="Public data"
    )
    mock_summarizer.summarize_candidate.return_value = (tech_summary, assessment)

    mock_repo = MagicMock()
    mock_repo.get_paper_by_hash.return_value = None
    # Unconfigured MagicMock attributes return a truthy MagicMock, which would
    # make every candidate look like a pre-existing duplicate against the new
    # dedup checks in execute_quota_loop. Make the mock behave like a repo with
    # no prior papers/candidates, matching a real fresh discovery run.
    mock_repo.find_paper_by_external_ids.return_value = None
    mock_repo.get_candidate_for_run_and_paper.return_value = None

    manager = IterativeSourceManager(
        target_sources=3,
        summarizer=mock_summarizer,
        enable_secondary_graph_expansion=False
    )

    candidates = [
        {"title": f"Paper {i}", "doi": f"10.1000/p.{i}"} for i in range(1, 10)
    ]

    accepted = await manager.execute_quota_loop(
        repo=mock_repo,
        run_id=uuid.uuid4(),
        candidate_queue=candidates,
        input_paper_meta={"title": "Seed Paper"},
        cited_references=[]
    )

    # Loop stopped after getting exactly 3 accepted sources
    assert len(accepted) == 3


@pytest.mark.asyncio
async def test_quota_loop_prunes_rejected_and_fetches_replacement():
    # Mock summarizer that rejects candidate #1 and accepts others
    mock_summarizer = MagicMock()

    accepted_critique = SelfCritiqueAssessment(
        is_relevant_to_seed_topic=True,
        relevance_score=8.0,
        factual_grounding_score=8.0,
        critique_rationale="Accepted",
        verdict="accepted"
    )

    rejected_critique = SelfCritiqueAssessment(
        is_relevant_to_seed_topic=False,
        relevance_score=3.0,
        factual_grounding_score=4.0,
        critique_rationale="Off-topic relative to seed paper",
        verdict="rejected"
    )

    summary_accepted = TechnicalSummary(
        problem_formulation="P", methodological_novelty="M", empirical_findings="E", paragraph_summary="S",
        data_availability=PaperAvailabilityStatus.PUBLICLY_AVAILABLE,
        critique=accepted_critique
    )
    summary_rejected = TechnicalSummary(
        problem_formulation="P", methodological_novelty="M", empirical_findings="E", paragraph_summary="S",
        data_availability=PaperAvailabilityStatus.PUBLICLY_AVAILABLE,
        critique=rejected_critique
    )
    assessment = DataAvailabilityAssessment(overall_status=PaperAvailabilityStatus.PUBLICLY_AVAILABLE, datasets=[], rationale="Public")

    # Candidate 1 rejected, Candidates 2 & 3 accepted
    mock_summarizer.summarize_candidate.side_effect = [
        (summary_rejected, assessment),
        (summary_accepted, assessment),
        (summary_accepted, assessment)
    ]

    mock_repo = MagicMock()
    mock_repo.get_paper_by_hash.return_value = None
    # Unconfigured MagicMock attributes return a truthy MagicMock, which would
    # make every candidate look like a pre-existing duplicate against the new
    # dedup checks in execute_quota_loop. Make the mock behave like a repo with
    # no prior papers/candidates, matching a real fresh discovery run.
    mock_repo.find_paper_by_external_ids.return_value = None
    mock_repo.get_candidate_for_run_and_paper.return_value = None

    manager = IterativeSourceManager(
        target_sources=2,
        summarizer=mock_summarizer,
        enable_secondary_graph_expansion=False
    )

    candidates = [
        {"title": "Off-topic Candidate", "doi": "10.1000/reject"},
        {"title": "Valid Candidate 1", "doi": "10.1000/valid1"},
        {"title": "Valid Candidate 2", "doi": "10.1000/valid2"}
    ]

    accepted = await manager.execute_quota_loop(
        repo=mock_repo,
        run_id=uuid.uuid4(),
        candidate_queue=candidates,
        input_paper_meta={"title": "Seed Paper"},
        cited_references=[]
    )

    assert len(accepted) == 2
    assert accepted[0]["title"] == "Valid Candidate 1"
    assert accepted[1]["title"] == "Valid Candidate 2"


@pytest.mark.asyncio
async def test_secondary_graph_expansion_trigger():
    mock_oa_client = AsyncMock()
    mock_s2_client = AsyncMock()

    mock_oa_client.find_related_candidates.return_value = [
        {"title": "Graph Expanded Paper 1", "doi": "10.1000/exp1"}
    ]
    mock_s2_client.get_paper_recommendations.return_value = []

    expander = SecondaryGraphExpander(
        openalex_client=mock_oa_client,
        semantic_scholar_client=mock_s2_client
    )

    mock_summarizer = MagicMock()
    summary = TechnicalSummary(
        problem_formulation="P", methodological_novelty="M", empirical_findings="E", paragraph_summary="S",
        data_availability=PaperAvailabilityStatus.PUBLICLY_AVAILABLE,
        critique=SelfCritiqueAssessment(
            is_relevant_to_seed_topic=True, relevance_score=8.0, factual_grounding_score=8.0,
            critique_rationale="Accepted", verdict="accepted"
        )
    )
    assessment = DataAvailabilityAssessment(overall_status=PaperAvailabilityStatus.PUBLICLY_AVAILABLE, datasets=[], rationale="Public")
    mock_summarizer.summarize_candidate.return_value = (summary, assessment)

    mock_repo = MagicMock()
    mock_repo.get_paper_by_hash.return_value = None
    # Unconfigured MagicMock attributes return a truthy MagicMock, which would
    # make every candidate look like a pre-existing duplicate against the new
    # dedup checks in execute_quota_loop. Make the mock behave like a repo with
    # no prior papers/candidates, matching a real fresh discovery run.
    mock_repo.find_paper_by_external_ids.return_value = None
    mock_repo.get_candidate_for_run_and_paper.return_value = None

    manager = IterativeSourceManager(
        target_sources=2,
        summarizer=mock_summarizer,
        expander=expander,
        enable_secondary_graph_expansion=True
    )

    initial_queue = [
        {"title": "Initial Candidate 1", "doi": "10.1000/init1", "openalex_id": "W111"}
    ]

    accepted = await manager.execute_quota_loop(
        repo=mock_repo,
        run_id=uuid.uuid4(),
        candidate_queue=initial_queue,
        input_paper_meta={"title": "Seed Paper"},
        cited_references=[]
    )

    # Initial candidate + graph expanded candidate = 2 accepted sources
    assert len(accepted) == 2
    assert accepted[0]["title"] == "Initial Candidate 1"
    assert accepted[1]["title"] == "Graph Expanded Paper 1"
