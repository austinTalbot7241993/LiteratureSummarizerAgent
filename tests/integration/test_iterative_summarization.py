import pytest
import asyncio
import uuid
from lea.llm.quota_manager import IterativeSourceManager
from lea.llm.inference import TechnicalSummarizer
from lea.llm.backends import MockLLMBackend
from lea.db.session import init_db, get_db_session
from lea.db.repository import LEARepository


@pytest.mark.asyncio
async def test_iterative_summarization_integration(tmp_path):
    db_path = tmp_path / "iterative_test.db"
    db_url = f"sqlite:///{db_path}"
    engine = init_db(db_url)

    mock_llm = MockLLMBackend()
    summarizer = TechnicalSummarizer(backend=mock_llm)

    manager = IterativeSourceManager(
        target_sources=4,
        min_relevance_score=6.0,
        min_grounding_score=7.0,
        enable_secondary_graph_expansion=False,
        summarizer=summarizer
    )

    with get_db_session() as session:
        repo = LEARepository(session)
        seed_paper = repo.create_paper(sha256_hash="seed_hash", title="Seed Paper on Deep Learning")
        run = repo.create_discovery_run(input_paper_id=seed_paper.id)

        candidates = [
            {"title": f"Discovered Candidate {i}", "doi": f"10.1000/disc.{i}", "abstract": f"Abstract for candidate {i}"}
            for i in range(1, 10)
        ]

        input_meta = {"title": seed_paper.title}

        accepted = await manager.execute_quota_loop(
            repo=repo,
            run_id=run.id,
            candidate_queue=candidates,
            input_paper_meta=input_meta,
            cited_references=[]
        )

        assert len(accepted) == 4

        # Verify summaries in DB
        summaries = repo.get_summaries_for_run(run.id, accepted_only=True)
        assert len(summaries) == 4
        for sum_obj in summaries:
            assert sum_obj.is_accepted is True
            assert sum_obj.self_critique_verdict == "accepted"
            assert sum_obj.self_critique_relevance_score == 8.5
            assert sum_obj.self_critique_grounding_score == 9.0


@pytest.mark.asyncio
async def test_quota_loop_deduplicates_rediscovered_candidate_against_real_db(tmp_path):
    """Regression test for a real production bug: the same real-world paper
    (identified by a stable DOI) rediscovered twice in one run -- exactly what
    happens when secondary graph expansion repeatedly surfaces the same
    highly-cited "hub" paper from multiple accepted candidates' neighborhoods
    -- must be evaluated ONCE, not summarized/critiqued/stored twice.

    This deliberately uses a real sqlite-backed LEARepository rather than a
    MagicMock, because the previous test suite mocked `repo.get_paper_by_hash`
    to always return None -- which hid the fact that candidates from discovery
    never carry a sha256_hash at all, so that lookup could never have matched
    a duplicate even with a real database.
    """
    db_path = tmp_path / "dedup_test.db"
    db_url = f"sqlite:///{db_path}"
    init_db(db_url)

    mock_llm = MockLLMBackend()
    summarizer = TechnicalSummarizer(backend=mock_llm)

    manager = IterativeSourceManager(
        target_sources=10,
        min_relevance_score=6.0,
        min_grounding_score=7.0,
        enable_secondary_graph_expansion=False,
        summarizer=summarizer
    )

    with get_db_session() as session:
        repo = LEARepository(session)
        seed_paper = repo.create_paper(sha256_hash="seed_hash_2", title="Seed Paper on Deep Learning")
        run = repo.create_discovery_run(input_paper_id=seed_paper.id)

        # The same DOI appears twice in the queue -- as if the same paper were
        # returned once by initial discovery and again by graph expansion.
        duplicate_doi = "10.1000/hub-paper"
        candidates = [
            {"title": "Hub Paper", "doi": duplicate_doi, "abstract": "A widely-cited hub paper."},
            {"title": "Distinct Candidate", "doi": "10.1000/distinct", "abstract": "A genuinely different paper."},
            {"title": "Hub Paper", "doi": duplicate_doi, "abstract": "A widely-cited hub paper."},
        ]

        input_meta = {"title": seed_paper.title}

        accepted = await manager.execute_quota_loop(
            repo=repo,
            run_id=run.id,
            candidate_queue=candidates,
            input_paper_meta=input_meta,
            cited_references=[]
        )

        # Both distinct DOIs accepted, but the duplicate must not double-count.
        assert len(accepted) == 2

        candidate_rows = repo.get_candidates_for_run(run.id)
        assert len(candidate_rows) == 2, (
            f"Expected exactly 2 candidate_paper rows (one per distinct DOI), got "
            f"{len(candidate_rows)} -- the duplicate 'Hub Paper' entry was evaluated "
            f"and stored more than once."
        )

        hub_papers = [c for c in candidate_rows if c.paper.doi == duplicate_doi]
        assert len(hub_papers) == 1, "The rediscovered hub paper produced more than one Paper/CandidatePaper row."

        summaries = repo.get_summaries_for_run(run.id, accepted_only=False)
        assert len(summaries) == 2, (
            f"Expected exactly 2 summaries generated, got {len(summaries)} -- the "
            f"summarizer was invoked more than once for the same rediscovered paper."
        )


@pytest.mark.asyncio
async def test_quota_loop_still_evaluates_candidates_pre_registered_by_discover(tmp_path):
    """Regression test for a real production bug introduced by the fix above:
    cli.py's `discover` command pre-registers a CandidatePaper row for EVERY
    screened candidate before the quota loop ever runs (see discover()'s
    `repo.add_candidate_paper(...)` loop). The first version of the dedup fix
    checked "does a CandidatePaper row already exist for (run, paper)" and
    skipped the candidate if so -- which meant EVERY candidate looked like an
    already-evaluated duplicate on its very first encounter, since discover()
    had just registered a row for all of them moments earlier. That produced
    "Discovered Related Literature (0 Papers)" on a real `lea run` invocation
    despite every unit test passing, because no existing test modeled this
    pre-registration step before calling execute_quota_loop().

    This test reproduces that exact call sequence: pre-register CandidatePaper
    rows the way `discover()` does, THEN run the quota loop on those same
    candidates, and assert they are actually downloaded/summarized/accepted
    rather than being skipped as "duplicates."
    """
    db_path = tmp_path / "prereg_test.db"
    db_url = f"sqlite:///{db_path}"
    init_db(db_url)

    mock_llm = MockLLMBackend()
    summarizer = TechnicalSummarizer(backend=mock_llm)

    manager = IterativeSourceManager(
        target_sources=3,
        min_relevance_score=6.0,
        min_grounding_score=7.0,
        enable_secondary_graph_expansion=False,
        summarizer=summarizer
    )

    with get_db_session() as session:
        repo = LEARepository(session)
        seed_paper = repo.create_paper(sha256_hash="seed_hash_3", title="Seed Paper on Deep Learning")
        run = repo.create_discovery_run(input_paper_id=seed_paper.id)

        candidates = [
            {"title": f"Screened Candidate {i}", "doi": f"10.1000/prereg.{i}", "abstract": f"Abstract {i}"}
            for i in range(1, 4)
        ]

        # Mirror discover()'s pre-registration: a Paper + CandidatePaper row is
        # created for every screened candidate BEFORE the quota loop runs.
        for cand in candidates:
            c_paper = repo.create_paper(
                sha256_hash=f"prereg-{cand['doi']}",
                title=cand["title"],
                doi=cand["doi"],
                abstract=cand["abstract"]
            )
            repo.add_candidate_paper(run_id=run.id, paper_id=c_paper.id, score=1.0)

        input_meta = {"title": seed_paper.title}

        accepted = await manager.execute_quota_loop(
            repo=repo,
            run_id=run.id,
            candidate_queue=candidates,
            input_paper_meta=input_meta,
            cited_references=[]
        )

        assert len(accepted) == 3, (
            "All 3 pre-registered candidates should have been evaluated and accepted; "
            "got fewer, meaning pre-registered-but-unevaluated candidates were wrongly "
            "skipped as duplicates."
        )

        summaries = repo.get_summaries_for_run(run.id, accepted_only=False)
        assert len(summaries) == 3, (
            f"Expected 3 summaries (one per pre-registered candidate), got {len(summaries)}."
        )
