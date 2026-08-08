from typing import List, Dict, Any, Optional, Tuple
from lea.llm.schemas import TechnicalSummary, DataAvailabilityAssessment, PaperAvailabilityStatus, VerificationStatus
from lea.llm.backends import BaseLLMBackend, MockLLMBackend
from lea.rag.prompts import (
    SUMMARY_SYSTEM_PROMPT,
    SUMMARY_USER_PROMPT_TEMPLATE,
    DATA_AVAILABILITY_SYSTEM_PROMPT,
    DATA_AVAILABILITY_USER_PROMPT_TEMPLATE
)
from lea.rag.availability_retrieval import format_availability_context
from lea.llm.aggregation import compute_overall_paper_status
from lea.exceptions import SummaryValidationError
from lea.logging import logger


class TechnicalSummarizer:
    def __init__(self, backend: Optional[BaseLLMBackend] = None, max_attempts: int = 2):
        self.backend = backend or MockLLMBackend()
        self.max_attempts = max_attempts

    def extract_data_availability(
        self,
        candidate_meta: Dict[str, Any],
        availability_chunks: List[Dict[str, Any]]
    ) -> DataAvailabilityAssessment:
        title = candidate_meta.get("title", "Untitled")
        authors = ", ".join(candidate_meta.get("authors", [])) or "Unknown"
        year = str(candidate_meta.get("publication_year") or candidate_meta.get("year") or "N/A")

        context_text = format_availability_context(availability_chunks)

        user_prompt = DATA_AVAILABILITY_USER_PROMPT_TEMPLATE.format(
            title=title,
            authors=authors,
            year=year,
            context_text=context_text
        )

        last_error = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                logger.info(f"Extracting data availability assessment for '{title}' (Attempt {attempt}/{self.max_attempts})...")
                assessment = self.backend.generate_data_availability(DATA_AVAILABILITY_SYSTEM_PROMPT, user_prompt)
                
                # Deterministic override/validation of paper-level overall status
                computed_overall = compute_overall_paper_status(assessment.datasets, has_evidence=bool(availability_chunks))
                if assessment.overall_status != computed_overall:
                    logger.info(f"Deterministic aggregation updated overall status from {assessment.overall_status} to {computed_overall}")
                    assessment.overall_status = computed_overall

                return assessment
            except SummaryValidationError as exc:
                logger.warning(f"Data availability validation failed on attempt {attempt}: {exc}")
                last_error = exc
            except Exception as exc:
                logger.warning(f"Unexpected error during data availability extraction: {exc}")
                last_error = exc

        logger.warning(
            f"Failed to generate valid DataAvailabilityAssessment for '{title}' after {self.max_attempts} attempts: {last_error}. "
            f"Falling back to NOT_REPORTED assessment."
        )
        return DataAvailabilityAssessment(
            overall_status=PaperAvailabilityStatus.NOT_REPORTED,
            datasets=[],
            rationale=f"Data availability extraction failed validation after {self.max_attempts} attempts ({last_error}). Defaulted to not_reported.",
            verification_status=VerificationStatus.NOT_CHECKED
        )

    def summarize_candidate(
        self,
        candidate_meta: Dict[str, Any],
        retrieved_chunks: List[Dict[str, Any]],
        availability_chunks: Optional[List[Dict[str, Any]]] = None,
        target_paper_meta: Optional[Dict[str, Any]] = None
    ) -> Tuple[TechnicalSummary, DataAvailabilityAssessment]:
        title = candidate_meta.get("title", "Untitled")
        authors = ", ".join(candidate_meta.get("authors", [])) or "Unknown"
        year = str(candidate_meta.get("publication_year") or candidate_meta.get("year") or "N/A")
        target_title = (target_paper_meta or {}).get("title") or candidate_meta.get("target_title") or "Target Input Paper"

        chunk_texts = [
            f"[CHUNK id={c.get('id', i+1)} index={c.get('chunk_index', i+1)}]\n{c.get('content')}"
            for i, c in enumerate(retrieved_chunks)
        ]
        context_text = "\n\n".join(chunk_texts) if chunk_texts else "No full text chunks available. Title and abstract only."

        user_prompt = SUMMARY_USER_PROMPT_TEMPLATE.format(
            title=title,
            authors=authors,
            year=year,
            target_title=target_title,
            context_text=context_text
        )

        last_error = None
        summary = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                if attempt > 1:
                    truncated_chunks = retrieved_chunks[: max(1, len(retrieved_chunks) // 2)]
                    c_texts = [
                        f"[CHUNK id={c.get('id', i+1)} index={c.get('chunk_index', i+1)}]\n{c.get('content')}"
                        for i, c in enumerate(truncated_chunks)
                    ]
                    c_str = "\n\n".join(c_texts) if c_texts else "No full text chunks available."
                    user_prompt = SUMMARY_USER_PROMPT_TEMPLATE.format(
                        title=title,
                        authors=authors,
                        year=year,
                        target_title=target_title,
                        context_text=c_str
                    ) + "\n\nCRITICAL RETRY DIRECTIVE: You MUST output all 5 section headers: PROBLEM FORMULATION:, METHODOLOGICAL NOVELTY:, EMPIRICAL FINDINGS:, TECHNICAL SYNTHESIS:, RELATIONSHIP TO TARGET PAPER:."

                logger.info(f"Generating summary for '{title}' (Attempt {attempt}/{self.max_attempts})...")
                summary = self.backend.generate_summary(SUMMARY_SYSTEM_PROMPT, user_prompt)
                break
            except SummaryValidationError as exc:
                logger.warning(f"Summary validation failed on attempt {attempt}: {exc}")
                last_error = exc
            except Exception as exc:
                logger.warning(f"Unexpected error during summary generation: {exc}")
                last_error = exc

        if summary is None:
            raise SummaryValidationError(f"Failed to generate valid TechnicalSummary after {self.max_attempts} attempts: {last_error}")

        # Extract data availability assessment
        avail_chunks = availability_chunks if availability_chunks is not None else retrieved_chunks
        assessment = self.extract_data_availability(candidate_meta, avail_chunks)

        # Derive compatibility fields on TechnicalSummary
        summary.data_availability = assessment.overall_status
        location = None
        for d in assessment.datasets:
            if d.url:
                location = d.url
                break
            elif d.accession:
                location = f"{d.repository or 'Accession'}: {d.accession}"
                break
            elif d.access_conditions:
                location = d.access_conditions
                break
        summary.data_location = location

        return summary, assessment
