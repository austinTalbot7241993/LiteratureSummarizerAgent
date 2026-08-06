import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from jinja2 import Environment, FileSystemLoader, select_autoescape
from lea.bibliography.bibtex import generate_bibtex
from lea.db.repository import LEARepository

class HTMLReportExporter:
    def __init__(self, template_dir: Optional[str] = None):
        if template_dir is None:
            template_dir = str(Path(__file__).parent / "templates")

        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape(["html", "xml"])
        )

    def export_report(
        self,
        repo: LEARepository,
        run_id,
        output_filepath: str,
        title: str = "Literature Exploration Report",
        include_abstract_only: bool = False
    ) -> str:
        run = repo.get_discovery_run(run_id)
        if not run:
            raise ValueError(f"Discovery run {run_id} not found.")

        input_paper = run.input_paper
        candidates = repo.get_candidates_for_run(run_id)

        items = []
        for cand in candidates:
            paper = cand.paper
            summary = cand.summary

            if not cand.is_downloaded and not include_abstract_only:
                continue

            paper_dict = {
                "id": str(paper.id),
                "title": paper.title,
                "authors": paper.authors or [],
                "doi": paper.doi,
                "arxiv_id": paper.arxiv_id,
                "publication_year": paper.publication_year,
                "venue": paper.venue,
                "abstract": paper.abstract,
                "is_open_access": paper.is_open_access,
                "oa_pdf_url": paper.oa_pdf_url or cand.open_access_url
            }

            bibtex_str = generate_bibtex(paper_dict)

            summary_dict = None
            if summary:
                summary_dict = {
                    "problem_formulation": summary.problem_formulation,
                    "methodological_novelty": summary.methodological_novelty,
                    "empirical_findings": summary.empirical_findings,
                    "paragraph_summary": summary.paragraph_summary,
                    "data_availability": getattr(summary, "data_availability", "proprietary"),
                    "data_location": getattr(summary, "data_location", None)
                }

            items.append({
                "candidate": {
                    "score": cand.score,
                    "rrf_rank": cand.rrf_rank,
                    "is_downloaded": cand.is_downloaded
                },
                "paper": paper_dict,
                "summary": summary_dict,
                "bibtex": bibtex_str
            })

        input_paper_dict = {
            "title": input_paper.title,
            "doi": input_paper.doi,
            "arxiv_id": input_paper.arxiv_id,
            "publication_year": input_paper.publication_year
        }

        template = self.env.get_template("report.html.j2")
        html_out = template.render(
            report_title=title,
            input_paper=input_paper_dict,
            exclusion_status=run.exclusion_status,
            candidates=items
        )

        out_path = Path(output_filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html_out)

        return str(out_path)
