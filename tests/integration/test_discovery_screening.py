import pytest
import asyncio
from typing import List, Dict, Any
from unittest.mock import AsyncMock
from lea.discovery.candidate_builder import CandidateBuilder
from lea.discovery.abstract_screener import AbstractScreener
from lea.llm.backends import MockLLMBackend
from lea.config import LEAConfig, DiscoveryConfig, ScreeningConfig


@pytest.mark.asyncio
async def test_candidate_builder_abstract_screening_integration():
    # Mock OpenAlex and Semantic Scholar clients returning candidate pools
    mock_oa_client = AsyncMock()
    mock_s2_client = AsyncMock()

    mock_oa_client.get_work.return_value = {"openalex_id": "W123456"}
    mock_oa_client.find_related_candidates.return_value = [
        {
            "title": f"Candidate OpenAlex {i}",
            "abstract": f"Abstract text for OpenAlex candidate {i}",
            "doi": f"10.1000/oa.{i}",
            "publication_year": 2023
        }
        for i in range(1, 15)
    ]
    mock_oa_client.search_candidates.return_value = []
    mock_s2_client.get_paper_recommendations.return_value = [
        {
            "title": f"Candidate S2 {j}",
            "abstract": f"Abstract text for S2 candidate {j}",
            "doi": f"10.1000/s2.{j}",
            "publication_year": 2023
        }
        for j in range(1, 15)
    ]

    mock_llm = MockLLMBackend()
    config = LEAConfig(
        discovery=DiscoveryConfig(
            final_candidate_limit=10,
            screening=ScreeningConfig(
                enabled=True,
                method="llm",
                pre_screening_limit=20,
                min_relevance_score=6.0,
                max_screened_candidates=5
            )
        )
    )

    builder = CandidateBuilder(
        openalex_client=mock_oa_client,
        semantic_scholar_client=mock_s2_client,
        config=config,
        llm_backend=mock_llm
    )

    seed_paper_meta = {
        "title": "Seed Paper on Hybrid Retrieval Agents",
        "abstract": "We present LEA, an autonomous literature summarization agent.",
        "openalex_id": "W123456"
    }

    candidates = await builder.build_candidates(
        input_paper_meta=seed_paper_meta,
        cited_references=[],
        exclusion_status="complete",
        final_candidate_limit=5
    )

    assert len(candidates) <= 5
    assert len(candidates) > 0
    for cand in candidates:
        assert "abstract_relevance_score" in cand
        assert "abstract_relevance_tier" in cand
        assert "abstract_relevance_reasoning" in cand
        assert cand["abstract_relevance_score"] >= 6.0
        assert cand["abstract_relevance_tier"] in ["high", "moderate"]


@pytest.mark.asyncio
async def test_candidate_builder_discards_doi_resolved_identity_on_title_mismatch():
    """Regression test for a real production bug: a DOI extracted from a
    PDF's header can resolve to a completely unrelated OpenAlex work.
    Confirmed live against a real anonymized preprint: GROBID's header
    extraction non-deterministically resolved it to two DIFFERENT unrelated
    papers' DOIs across separate calls (one via CrossRef consolidation
    matching against Besag et al.'s classic spatial-statistics paper, another
    even with consolidation disabled). Trusting that wrong identity made the
    ENTIRE discovery pass search the wrong paper's citation neighborhood
    instead of the seed paper's -- confirmed live: a run corrupted this way
    returned candidates about random matrix theory and point processes for a
    genomics linkage-disequilibrium paper. CandidateBuilder must sanity-check
    the resolved work's title against the input paper's own title before
    trusting it, and fall back to a plain title search instead of the wrong
    citation-graph neighborhood.
    """
    mock_oa_client = AsyncMock()
    mock_s2_client = AsyncMock()

    # The (wrong) DOI resolves to a real but completely unrelated paper.
    mock_oa_client.get_work.return_value = {
        "openalex_id": "W_WRONG_PAPER",
        "title": "Statistical Analysis of Non-Lattice Data"
    }
    mock_oa_client.find_related_candidates.return_value = [
        {"title": "Should never be fetched", "doi": "10.1000/wrong-neighbor", "publication_year": 2020}
    ]
    mock_oa_client.search_candidates.return_value = [
        {"title": "Genuine LD Estimation Follow-up", "doi": "10.1000/real-neighbor", "publication_year": 2024}
    ]
    mock_s2_client.get_paper_recommendations.return_value = []

    builder = CandidateBuilder(
        openalex_client=mock_oa_client,
        semantic_scholar_client=mock_s2_client,
        config=None,
        llm_backend=None
    )

    seed_paper_meta = {
        "title": "HapSpin: Regularized Linkage Disequilibrium Matrix Estimation",
        "doi": "10.2307/2987782",  # the wrong DOI, as actually extracted from the PDF header
        "abstract": "Linkage disequilibrium matrices underpin GWAS methods."
    }

    candidates = await builder.build_candidates(
        input_paper_meta=seed_paper_meta,
        cited_references=[],
        exclusion_status="complete",
        final_candidate_limit=10,
        screen_abstracts=False
    )

    mock_oa_client.find_related_candidates.assert_not_called()
    mock_oa_client.search_candidates.assert_called_once()
    titles = [c.get("title") for c in candidates]
    assert "Should never be fetched" not in titles
    assert "Genuine LD Estimation Follow-up" in titles
