import tempfile
import pytest
from pathlib import Path
from lea.exporter.html_builder import HTMLReportExporter
from lea.db.session import init_db, get_db_session
from lea.db.repository import LEARepository

def test_html_report_rendering(tmp_path):
    exporter = HTMLReportExporter()
    # Test Jinja2 template loading and rendering directly
    template = exporter.env.get_template("report.html.j2")
    html = template.render(
        report_title="Test Exploration Report",
        input_paper={"title": "Base Input Paper", "doi": "10.1000/base"},
        exclusion_status="complete",
        candidates=[
            {
                "candidate": {"rrf_rank": 1, "score": 0.033},
                "paper": {
                    "title": "Discovered Candidate Paper",
                    "authors": ["John Doe"],
                    "publication_year": 2023,
                    "venue": "ICML",
                    "is_open_access": True,
                    "oa_pdf_url": "https://example.com/paper.pdf"
                },
                "summary": {
                    "problem_formulation": "Problem statement",
                    "methodological_novelty": "Novel algorithm",
                    "empirical_findings": "High performance",
                    "paragraph_summary": "Single paragraph technical synthesis.",
                    "relationship_to_target": "Direct extension of base input paper."
                },
                "bibtex": "@article{doe2023discovered, title={Discovered Candidate Paper}}"
            }
        ]
    )

    assert "Test Exploration Report" in html
    assert "Discovered Candidate Paper" in html
    assert "Single paragraph technical synthesis." in html
    assert "Relationship to Target Paper" in html
    assert "Direct extension of base input paper." in html
    assert "@article{doe2023discovered" in html


def test_export_report_includes_accepted_summary_without_downloaded_pdf(tmp_path):
    """Regression test for a real production bug: export_report() previously
    required cand.is_downloaded to be True before a candidate would appear in
    the report at all. But summarization deliberately falls back to
    title/abstract-only context when no open-access PDF could be acquired
    (most candidates simply aren't open access), and self-critique can still
    legitimately accept a paper on that basis. That meant a fully successful
    run whose accepted candidates all happened to lack a downloaded PDF
    rendered "Discovered Related Literature (0 Papers)" -- with real, accepted
    summaries sitting in the database the whole time.

    The previous test in this file only rendered the Jinja2 template directly
    with hand-built data, bypassing export_report()'s own DB-querying and
    filtering logic entirely -- exactly the layer that was broken. This test
    exercises export_report() itself against a real database.
    """
    db_path = tmp_path / "export_test.db"
    init_db(f"sqlite:///{db_path}")

    with get_db_session() as session:
        repo = LEARepository(session)
        input_paper = repo.create_paper(sha256_hash="input_hash", title="Input Paper")
        run = repo.create_discovery_run(input_paper_id=input_paper.id)

        # Candidate whose PDF was never downloaded (e.g. not open access) but
        # was still summarized from abstract-only context and accepted.
        undownloaded_paper = repo.create_paper(
            sha256_hash="undownloaded_hash",
            title="Accepted But Never Downloaded",
            abstract="An abstract-only candidate that was still accepted."
        )
        undownloaded_cand = repo.add_candidate_paper(run_id=run.id, paper_id=undownloaded_paper.id, score=1.0)
        assert undownloaded_cand.is_downloaded is False
        repo.add_summary(
            run_id=run.id,
            candidate_paper_id=undownloaded_cand.id,
            problem_formulation="P", methodological_novelty="M", empirical_findings="E",
            paragraph_summary="Accepted without a downloaded PDF.",
            model_name="test-model", data_availability="not_reported",
            self_critique_verdict="accepted", self_critique_relevance_score=8.0,
            self_critique_grounding_score=8.0, is_accepted=True
        )

        # Candidate that was pre-registered by discover() but never reached the
        # quota loop's deep-evaluation stage -- no summary exists at all.
        unscreened_paper = repo.create_paper(sha256_hash="unscreened_hash", title="Never Evaluated")
        repo.add_candidate_paper(run_id=run.id, paper_id=unscreened_paper.id, score=0.5)

        # Candidate that was evaluated and explicitly rejected.
        rejected_paper = repo.create_paper(sha256_hash="rejected_hash", title="Explicitly Rejected")
        rejected_cand = repo.add_candidate_paper(run_id=run.id, paper_id=rejected_paper.id, score=0.5)
        repo.add_summary(
            run_id=run.id,
            candidate_paper_id=rejected_cand.id,
            problem_formulation="P", methodological_novelty="M", empirical_findings="E",
            paragraph_summary="Rejected as irrelevant.",
            model_name="test-model", data_availability="not_reported",
            self_critique_verdict="rejected", self_critique_relevance_score=1.0,
            self_critique_grounding_score=1.0, is_accepted=False
        )

        exporter = HTMLReportExporter()
        out_path = exporter.export_report(repo, run.id, str(tmp_path / "report.html"))
        html = Path(out_path).read_text(encoding="utf-8")

        assert "Accepted But Never Downloaded" in html, (
            "An accepted, summarized candidate must appear in the report even "
            "without a downloaded PDF."
        )
        assert "Never Evaluated" not in html, "A candidate with no summary at all should not appear by default."
        assert "Explicitly Rejected" not in html, "A candidate whose summary was explicitly rejected should not appear by default."
        assert "Discovered Related Literature (1 Papers)" in html
