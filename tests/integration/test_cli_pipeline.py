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
