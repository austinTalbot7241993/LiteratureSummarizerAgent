"""Real, (deliberately) un-mocked end-to-end pipeline test.

Every other CLI-level test mocks the LLM backend and/or CandidateBuilder. That's
correct for fast unit tests, but it also means those tests can't catch bugs in
the exact code they replace -- which is precisely how the discover()-never-
wires-an-llm_backend regression (see tests/unit/test_cli_discover_backend_wiring.py)
went unnoticed while "all tests passed": the mock stood in for the very call
that was broken.

This test runs `lea run` for real: real network discovery APIs (OpenAlex /
Semantic Scholar), real BGE-M3 embeddings, and the real quantized Qwen 2.5 7B
backend on GPU. No `--mock`. It requires a CUDA GPU and network access, so it's
kept out of the default fast test run (tests/unit, tests/integration) under
tests/gpu/ and skipped automatically when no GPU is present.

Run explicitly with: pytest tests/gpu/test_real_end_to_end_pipeline.py -q
"""
import uuid
from pathlib import Path

import pytest
import torch
from typer.testing import CliRunner

from lea.cli import app
from lea.db.session import get_db_session
from lea.db.repository import LEARepository
from lea.db.models import Paper, DiscoveryRun

pytestmark = pytest.mark.gpu

runner = CliRunner()

FIXTURE_PDF = Path(__file__).resolve().parents[1] / "fixtures" / "sample_paper.pdf"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA GPU to load the real Qwen backend")
def test_real_end_to_end_pipeline_on_sample_paper(tmp_path, monkeypatch):
    db_file = tmp_path / "real_e2e.db"
    output_report = tmp_path / "real_report.html"
    monkeypatch.setenv("LEA_DATABASE_URL", f"sqlite:///{db_file}")

    result = runner.invoke(app, [
        "run",
        str(FIXTURE_PDF),
        "--output", str(output_report),
        "--target-sources", "2",
        # Force at least one candidate past screening/critique so real
        # summarization actually executes on GPU rather than short-circuiting.
        "--min-relevance", "0",
        "--min-grounding", "0",
    ])

    assert result.exit_code == 0, f"Real end-to-end run failed:\n{result.stdout}"
    assert output_report.exists() and output_report.stat().st_size > 0, "No report was written"

    with get_db_session() as session:
        paper = session.query(Paper).filter(Paper.pdf_path == str(FIXTURE_PDF)).first()
        assert paper is not None, "Input paper was never ingested into the database"

        repo = LEARepository(session)
        references = repo.get_references_for_paper(paper.id)
        assert len(references) > 0, "No bibliography references were extracted from the sample PDF"

        run_obj = (
            session.query(DiscoveryRun)
            .filter(DiscoveryRun.input_paper_id == paper.id)
            .order_by(DiscoveryRun.created_at.desc())
            .first()
        )
        assert run_obj is not None, "No discovery run was created"

        candidates = repo.get_candidates_for_run(run_obj.id)
        assert len(candidates) > 0, (
            "No candidate papers were discovered at all -- discovery/exclusion/"
            "screening produced an empty pool against real APIs"
        )

        screened = [c for c in candidates if c.abstract_relevance_score is not None]
        assert screened, "No candidates carry an abstract relevance score; screening never ran"

        # Regression guard: if LLM-based screening silently degrades back to the
        # embedding-cosine fallback (the bug this suite was written to catch),
        # every reasoning string will read as the templated fallback text.
        reasonings = [c.abstract_relevance_reasoning or "" for c in screened]
        all_fallback = all("cosine similarity" in r.lower() for r in reasonings if r)
        assert not all_fallback, (
            "Every screened candidate used the embedding-cosine fallback reasoning "
            "template; LLM-based abstract screening (screening.method='llm') never "
            "actually ran despite being the configured default."
        )

        summaries = repo.get_summaries_for_run(run_obj.id, accepted_only=False)
        assert len(summaries) > 0, (
            "No candidate reached real summarization -- the quota loop never invoked "
            "the LLM summarizer on GPU"
        )
