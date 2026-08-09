import pytest
import asyncio
from typing import List, Dict, Any
from lea.discovery.abstract_screener import AbstractScreener, AbstractRelevanceSchema, compute_relevance_tier
from lea.llm.backends import MockLLMBackend
from lea.config import LEAConfig, DiscoveryConfig, ScreeningConfig


class DummyEmbedder:
    def __init__(self, seed_sim_map: Dict[str, float]):
        self.seed_sim_map = seed_sim_map

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        # Returns 2D vector for seed text [1.0, 0.0] and candidates according to similarity
        # [cos(theta), sin(theta)] gives cosine similarity cos(theta) with [1.0, 0.0]
        vectors = []
        # First text is seed
        vectors.append([1.0, 0.0])
        for text in texts[1:]:
            sim = self.seed_sim_map.get(text, 0.5)
            sim = max(-1.0, min(1.0, sim))
            sin_val = (1.0 - sim ** 2) ** 0.5
            vectors.append([sim, sin_val])
        return vectors


@pytest.mark.asyncio
async def test_abstract_screener_llm_mode():
    mock_backend = MockLLMBackend()
    screener = AbstractScreener(llm_backend=mock_backend)

    seed_paper = {
        "title": "Attention Is All You Need",
        "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks."
    }

    candidates = [
        {
            "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
            "abstract": "We introduce a new language representation model called BERT."
        },
        {
            "title": "GPT-3: Language Models are Few-Shot Learners",
            "abstract": "We show that scaling up language models greatly improves task-agnostic, few-shot performance."
        }
    ]

    screened = await screener.screen_candidates(
        seed_paper_meta=seed_paper,
        candidates=candidates,
        method="llm",
        min_score=6.0,
        max_candidates=10
    )

    assert len(screened) == 2
    for cand in screened:
        assert "abstract_relevance_score" in cand
        assert cand["abstract_relevance_score"] == 8.5
        assert cand["abstract_relevance_tier"] == "high"
        assert "abstract_relevance_reasoning" in cand


@pytest.mark.asyncio
async def test_abstract_screener_embedding_mode():
    cand1_text = "BERT: Pre-training of Deep Bidirectional Transformers We introduce BERT."
    cand2_text = "Irrelevant Gardening Guide A book about growing tomatoes."

    embedder = DummyEmbedder({
        cand1_text: 0.85,
        cand2_text: 0.20
    })

    screener = AbstractScreener(embedder=embedder)

    seed_paper = {
        "title": "Attention Is All You Need",
        "abstract": "Transformer sequence models."
    }

    candidates = [
        {
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "abstract": "We introduce BERT."
        },
        {
            "title": "Irrelevant Gardening Guide",
            "abstract": "A book about growing tomatoes."
        }
    ]

    screened = await screener.screen_candidates(
        seed_paper_meta=seed_paper,
        candidates=candidates,
        method="embedding",
        min_score=6.0,
        max_candidates=10
    )

    assert len(screened) == 1
    assert screened[0]["title"] == "BERT: Pre-training of Deep Bidirectional Transformers"
    assert screened[0]["abstract_relevance_score"] == 8.5
    assert screened[0]["abstract_relevance_tier"] == "high"


@pytest.mark.asyncio
async def test_missing_abstract_fallback_pass():
    config = LEAConfig(
        discovery=DiscoveryConfig(
            screening=ScreeningConfig(fallback_on_missing_abstract="pass")
        )
    )
    mock_backend = MockLLMBackend()
    screener = AbstractScreener(config=config, llm_backend=mock_backend)

    seed_paper = {"title": "Seed Paper Title", "abstract": "Seed abstract."}
    candidates = [
        {"title": "Paper Without Abstract", "abstract": ""},
        {"title": "Paper With Abstract", "abstract": "Valid abstract text."}
    ]

    screened = await screener.screen_candidates(
        seed_paper_meta=seed_paper,
        candidates=candidates,
        method="llm",
        min_score=4.0,
        max_candidates=10
    )

    assert len(screened) == 2
    no_abs_cand = next(c for c in screened if c["title"] == "Paper Without Abstract")
    assert no_abs_cand["abstract_relevance_score"] == 5.0
    assert no_abs_cand["abstract_relevance_tier"] == "moderate"


@pytest.mark.asyncio
async def test_missing_abstract_fallback_drop():
    config = LEAConfig(
        discovery=DiscoveryConfig(
            screening=ScreeningConfig(fallback_on_missing_abstract="drop")
        )
    )
    mock_backend = MockLLMBackend()
    screener = AbstractScreener(config=config, llm_backend=mock_backend)

    seed_paper = {"title": "Seed Paper Title", "abstract": "Seed abstract."}
    candidates = [
        {"title": "Paper Without Abstract", "abstract": ""},
        {"title": "Paper With Abstract", "abstract": "Valid abstract text."}
    ]

    screened = await screener.screen_candidates(
        seed_paper_meta=seed_paper,
        candidates=candidates,
        method="llm",
        min_score=6.0,
        max_candidates=10
    )

    assert len(screened) == 1
    assert screened[0]["title"] == "Paper With Abstract"


@pytest.mark.asyncio
async def test_threshold_relaxation_when_no_candidates_pass():
    class LowScoreBackend:
        def generate_abstract_relevance(self, sys_prompt, user_prompt):
            return {
                "relevance_score": 2.0,
                "relevance_tier": "irrelevant",
                "reasoning": "Low relevance score test."
            }

    screener = AbstractScreener(llm_backend=LowScoreBackend())
    seed_paper = {"title": "Seed Paper", "abstract": "Seed abstract"}
    candidates = [
        {"title": "Candidate A", "abstract": "Abstract A"},
        {"title": "Candidate B", "abstract": "Abstract B"}
    ]

    screened = await screener.screen_candidates(
        seed_paper_meta=seed_paper,
        candidates=candidates,
        method="llm",
        min_score=8.0,  # No candidate achieves 8.0
        max_candidates=1
    )

    # Threshold relaxation retains top candidate
    assert len(screened) == 1
    assert screened[0]["abstract_relevance_score"] == 2.0


@pytest.mark.asyncio
async def test_llm_failure_falls_back_to_embedding():
    class FailingLLMBackend:
        def generate_abstract_relevance(self, sys_prompt, user_prompt):
            raise RuntimeError("LLM Service Outage")

    cand_text = "Candidate C Abstract C"
    embedder = DummyEmbedder({cand_text: 0.70})
    screener = AbstractScreener(llm_backend=FailingLLMBackend(), embedder=embedder)

    seed_paper = {"title": "Seed Paper", "abstract": "Seed abstract"}
    candidates = [
        {"title": "Candidate C", "abstract": "Abstract C"}
    ]

    screened = await screener.screen_candidates(
        seed_paper_meta=seed_paper,
        candidates=candidates,
        method="llm",
        min_score=5.0,
        max_candidates=10
    )

    assert len(screened) == 1
    assert screened[0]["abstract_relevance_score"] == 7.0
    assert "Dense embedding cosine similarity score" in screened[0]["abstract_relevance_reasoning"]
