import pytest
from typer.testing import CliRunner
from lea.cli import app
from lea.db.session import get_db_session
from lea.db.repository import LEARepository
from lea.db.models import Paper, DiscoveryRun

runner = CliRunner()

def test_cli_doctor():
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "System Environment Check" in result.stdout

def test_cli_db_init(tmp_path, monkeypatch):
    db_file = tmp_path / "cli_test.db"
    monkeypatch.setenv("LEA_DATABASE_URL", f"sqlite:///{db_file}")
    result = runner.invoke(app, ["db", "init"])
    assert result.exit_code == 0
    assert "Database schema successfully initialized" in result.stdout


def test_cli_run_sample_paper_command(tmp_path, monkeypatch):
    """Unit test for: python -m lea run tests/fixtures/sample_paper.pdf --output report_1p11.html"""
    db_file = tmp_path / "cli_run_sample.db"
    output_report = tmp_path / "report_1p11.html"
    monkeypatch.setenv("LEA_DATABASE_URL", f"sqlite:///{db_file}")

    result = runner.invoke(app, [
        "run",
        "tests/fixtures/sample_paper.pdf",
        "--output", str(output_report),
        "--mock",
        # MockLLMBackend.generate_abstract_relevance always scores 8.5/10, so
        # every screened candidate passes and proceeds to real PDF acquisition
        # network calls. Bound the quota so this stays a fast sanity check
        # rather than sequentially downloading dozens of real PDFs; the
        # un-mocked tests/gpu suite is where full-depth behavior is verified.
        "--target-sources", "2"
    ])

    assert result.exit_code == 0, f"CLI run command failed with output:\n{result.stdout}"
    assert output_report.exists(), "Output report file was not generated"
    assert output_report.stat().st_size > 0

    content = output_report.read_text(encoding="utf-8")
    assert "Target Input Paper" in content or "Literature Exploration" in content

    # An exit code of 0 and a non-empty report are necessary but not sufficient:
    # a pipeline that silently discovers/screens/accepts zero candidates also
    # exits 0 and writes a (near-empty) report. Verify real pipeline output
    # actually made it into the database at every stage.
    with get_db_session() as session:
        paper = session.query(Paper).filter(Paper.pdf_path == "tests/fixtures/sample_paper.pdf").first()
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
        assert len(candidates) > 0, "No candidate papers were discovered against the real discovery APIs"
