import pytest
from typer.testing import CliRunner
from lea.cli import app

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
        "--mock"
    ])

    assert result.exit_code == 0, f"CLI run command failed with output:\n{result.stdout}"
    assert output_report.exists(), "Output report file was not generated"
    assert output_report.stat().st_size > 0

    content = output_report.read_text(encoding="utf-8")
    assert "Target Input Paper" in content or "Literature Exploration" in content
