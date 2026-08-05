import tempfile
import pytest
from pathlib import Path
from lea.exporter.html_builder import HTMLReportExporter

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
                    "paragraph_summary": "Single paragraph technical synthesis."
                },
                "bibtex": "@article{doe2023discovered, title={Discovered Candidate Paper}}"
            }
        ]
    )

    assert "Test Exploration Report" in html
    assert "Discovered Candidate Paper" in html
    assert "Single paragraph technical synthesis." in html
    assert "@article{doe2023discovered" in html
