import uuid
from unittest.mock import patch
import pytest
from typer.testing import CliRunner

from lea.cli import app
from lea.db.session import get_db_session
from lea.db.repository import LEARepository

runner = CliRunner()


def _seed_paper(db_file, monkeypatch) -> uuid.UUID:
    monkeypatch.setenv("LEA_DATABASE_URL", f"sqlite:///{db_file}")
    # Force a fresh engine bound to this test's DB file.
    result = runner.invoke(app, ["db", "init"])
    assert result.exit_code == 0, result.stdout

    with get_db_session() as session:
        repo = LEARepository(session)
        paper = repo.create_paper(
            sha256_hash=f"hash-{db_file.name}",
            title="Seed Paper on Hybrid Retrieval Agents",
            authors=["A. Author"],
            abstract="We present a seed paper abstract for backend wiring tests."
        )
        return paper.id


class _SpyCandidateBuilder:
    """Stand-in for CandidateBuilder that records how it was constructed."""

    captured_kwargs = None

    def __init__(self, *args, **kwargs):
        type(self).captured_kwargs = kwargs

    async def build_candidates(self, *args, **kwargs):
        return []


def test_discover_wires_llm_backend_for_default_llm_screening(tmp_path, monkeypatch):
    """Regression test: `discover` must construct and pass a real LLM backend into
    CandidateBuilder whenever LLM-based abstract screening is active (the default).

    Previously `discover()` called `CandidateBuilder(config=config)` with no
    `llm_backend` at all. Every unit test mocked CandidateBuilder or the LLM
    backend directly, so nothing ever exercised this wiring -- and in practice
    abstract screening silently fell back to embedding-only cosine similarity on
    every real run, with no error and no warning visible to a user, because
    AbstractScreener treats a missing backend as an expected fallback path.
    """
    db_file = tmp_path / "wiring_llm.db"
    paper_id = _seed_paper(db_file, monkeypatch)

    _SpyCandidateBuilder.captured_kwargs = None
    with patch("lea.discovery.candidate_builder.CandidateBuilder", _SpyCandidateBuilder):
        result = runner.invoke(app, ["discover", str(paper_id), "--mock"])

    assert result.exit_code == 0, result.stdout
    assert _SpyCandidateBuilder.captured_kwargs is not None, "CandidateBuilder was never constructed"
    llm_backend = _SpyCandidateBuilder.captured_kwargs.get("llm_backend")
    assert llm_backend is not None, (
        "discover() must pass a constructed llm_backend into CandidateBuilder when "
        "screening.method='llm' (the default) -- otherwise abstract screening silently "
        "degrades to embedding-only cosine similarity."
    )


def test_discover_skips_llm_backend_when_screening_disabled(tmp_path, monkeypatch):
    """When abstract screening is turned off entirely, no LLM backend should be
    constructed at all -- building one would mean an unnecessary model load
    (potentially several GB of VRAM and tens of seconds) for a feature the user
    explicitly disabled.
    """
    db_file = tmp_path / "wiring_disabled.db"
    paper_id = _seed_paper(db_file, monkeypatch)

    _SpyCandidateBuilder.captured_kwargs = None
    with patch("lea.discovery.candidate_builder.CandidateBuilder", _SpyCandidateBuilder):
        result = runner.invoke(app, ["discover", str(paper_id), "--no-screen-abstracts", "--mock"])

    assert result.exit_code == 0, result.stdout
    llm_backend = _SpyCandidateBuilder.captured_kwargs.get("llm_backend")
    assert llm_backend is None


class _DuplicateYieldingCandidateBuilder:
    """Stand-in for CandidateBuilder that returns the same real-world paper
    (by DOI) twice, as if an upstream merge/dedup step in CandidateBuilder
    missed a duplicate.
    """

    def __init__(self, *args, **kwargs):
        pass

    async def build_candidates(self, *args, **kwargs):
        return [
            {"title": "Hub Paper", "doi": "10.1000/hub-paper", "abstract": "A widely-cited hub paper.", "rrf_score": 0.05},
            {"title": "Distinct Candidate", "doi": "10.1000/distinct", "abstract": "A different paper.", "rrf_score": 0.04},
            {"title": "Hub Paper", "doi": "10.1000/hub-paper", "abstract": "A widely-cited hub paper.", "rrf_score": 0.05},
        ]


def test_discover_deduplicates_candidates_sharing_an_external_id(tmp_path, monkeypatch):
    """Regression test for a real production bug: `discover()`'s bulk-insert
    loop had no deduplication at all (unlike the quota loop, which was fixed
    separately) -- if CandidateBuilder's own internal merge/dedup ever missed
    a duplicate (confirmed happening on a real `lea run` against a real PDF:
    the same DOI appeared in two separate Paper/CandidatePaper rows within
    one discovery run), `discover()` would blindly create a second Paper row
    with a fresh random UUID hash for it, inflating the candidate count and
    surfacing the same paper twice in the final report.
    """
    db_file = tmp_path / "discover_dedup.db"
    paper_id = _seed_paper(db_file, monkeypatch)

    with patch("lea.discovery.candidate_builder.CandidateBuilder", _DuplicateYieldingCandidateBuilder):
        result = runner.invoke(app, ["discover", str(paper_id), "--mock"])

    assert result.exit_code == 0, result.stdout

    with get_db_session() as session:
        repo = LEARepository(session)
        from lea.db.models import DiscoveryRun
        run = session.query(DiscoveryRun).filter(DiscoveryRun.input_paper_id == paper_id).order_by(DiscoveryRun.created_at.desc()).first()
        assert run is not None

        candidates = repo.get_candidates_for_run(run.id)
        assert len(candidates) == 2, (
            f"Expected exactly 2 candidate_paper rows (one per distinct DOI), got "
            f"{len(candidates)} -- the duplicate 'Hub Paper' entry was stored more than once."
        )
        hub_rows = [c for c in candidates if c.paper.doi == "10.1000/hub-paper"]
        assert len(hub_rows) == 1


def test_discover_skips_llm_backend_when_method_is_embedding(tmp_path, monkeypatch):
    """When screening is explicitly configured to use the embedding method, no
    LLM backend should be constructed either -- same reasoning as above.
    """
    db_file = tmp_path / "wiring_embedding.db"
    paper_id = _seed_paper(db_file, monkeypatch)

    _SpyCandidateBuilder.captured_kwargs = None
    with patch("lea.discovery.candidate_builder.CandidateBuilder", _SpyCandidateBuilder):
        result = runner.invoke(app, [
            "discover", str(paper_id),
            "--screening-method", "embedding",
            "--mock"
        ])

    assert result.exit_code == 0, result.stdout
    llm_backend = _SpyCandidateBuilder.captured_kwargs.get("llm_backend")
    assert llm_backend is None
