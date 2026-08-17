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

            # A candidate with a real, accepted summary is a genuine pipeline
            # result and must be shown regardless of whether a full-text PDF
            # was ever downloaded -- summarization deliberately falls back to
            # title/abstract-only context when no PDF is available (most
            # candidates are not open access), and self-critique can still
            # legitimately accept a paper on that basis. Requiring
            # is_downloaded here meant a fully successful run whose accepted
            # papers all happened to lack an open-access PDF rendered an empty
            # "0 Papers" report even though real, accepted summaries existed.
            # Only candidates with NO summary at all (never reached the quota
            # loop's deep-evaluation stage, e.g. screened by `discover` but
            # not reached before the quota loop hit its target/depth cap) are
            # gated behind include_abstract_only.
            if not cand.summary and not include_abstract_only:
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
                if getattr(summary, "is_accepted", True) is False and not include_abstract_only:
                    continue

                summary_dict = {
                    "problem_formulation": summary.problem_formulation,
                    "methodological_novelty": summary.methodological_novelty,
                    "empirical_findings": summary.empirical_findings,
                    "paragraph_summary": summary.paragraph_summary,
                    "relationship_to_target": getattr(summary, "relationship_to_target", None),
                    "data_availability": getattr(summary, "data_availability", "unclear"),
                    "data_location": getattr(summary, "data_location", None),
                    "data_availability_assessment": getattr(summary, "data_availability_assessment", None),
                    "self_critique_verdict": getattr(summary, "self_critique_verdict", None),
                    "self_critique_relevance_score": getattr(summary, "self_critique_relevance_score", None),
                    "self_critique_grounding_score": getattr(summary, "self_critique_grounding_score", None),
                    "self_critique_rationale": getattr(summary, "self_critique_rationale", None),
                    "is_accepted": getattr(summary, "is_accepted", True)
                }

            items.append({
                "candidate": {
                    "score": cand.score,
                    "rrf_rank": cand.rrf_rank,
                    "is_downloaded": cand.is_downloaded,
                    "abstract_relevance_score": getattr(cand, "abstract_relevance_score", None),
                    "abstract_relevance_tier": getattr(cand, "abstract_relevance_tier", None),
                    "abstract_relevance_reasoning": getattr(cand, "abstract_relevance_reasoning", None)
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
