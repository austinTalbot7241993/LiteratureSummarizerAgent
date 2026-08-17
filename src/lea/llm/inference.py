from typing import List, Dict, Any, Optional, Tuple
from lea.llm.schemas import TechnicalSummary, DataAvailabilityAssessment, SelfCritiqueAssessment, PaperAvailabilityStatus, VerificationStatus
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
        target_paper_meta: Optional[Dict[str, Any]] = None,
        perform_critique: bool = True
    ) -> Tuple[TechnicalSummary, DataAvailabilityAssessment]:
        title = candidate_meta.get("title", "Untitled")
        authors = ", ".join(candidate_meta.get("authors", [])) or "Unknown"
        year = str(candidate_meta.get("publication_year") or candidate_meta.get("year") or "N/A")
        target_title = (target_paper_meta or {}).get("title") or candidate_meta.get("target_title") or "Target Input Paper"
        # The model can only judge "relationship to target paper" if it actually
        # knows what the target paper is about; a bare title is not enough
        # (especially for narrowly-named tools/methods a base model won't
        # recognize from pretraining alone), so pass the target's abstract too.
        target_abstract = (target_paper_meta or {}).get("abstract") or "No abstract available."

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
            target_abstract=target_abstract,
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
                        target_abstract=target_abstract,
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

        if perform_critique:
            try:
                summary.critique = self.critique_candidate(candidate_meta, summary, retrieved_chunks, target_paper_meta)
            except Exception as exc:
                logger.warning(f"Self-critique generation failed: {exc}")

        return summary, assessment

    def critique_candidate(
        self,
        candidate_meta: Dict[str, Any],
        summary: TechnicalSummary,
        retrieved_chunks: List[Dict[str, Any]],
        target_paper_meta: Optional[Dict[str, Any]] = None
    ) -> SelfCritiqueAssessment:
        title = candidate_meta.get("title", "Untitled")
        target_title = (target_paper_meta or {}).get("title") or "Target Input Paper"
        # Same reasoning as summarize_candidate(): the model can't judge whether a
        # candidate is relevant to a seed paper it only knows by name. Include the
        # seed abstract here too, matching what AbstractScreener already does.
        target_abstract = (target_paper_meta or {}).get("abstract") or "No abstract available."

        chunk_texts = [
            f"[CHUNK id={c.get('id', i+1)} index={c.get('chunk_index', i+1)}]\n{c.get('content')}"
            for i, c in enumerate(retrieved_chunks)
        ]
        context_text = "\n\n".join(chunk_texts) if chunk_texts else "No full text chunks available."

        system_prompt = (
            "You are a rigorous academic literature evaluator performing self-critique on candidate paper summaries and retrieved context."
        )
        user_prompt = (
            f"SEED PAPER TOPIC:\nTitle: {target_title}\nAbstract: {target_abstract}\n\n"
            f"CANDIDATE PAPER:\nTitle: {title}\n"
            f"Summary:\nProblem Formulation: {summary.problem_formulation}\n"
            f"Methodological Novelty: {summary.methodological_novelty}\n"
            f"Empirical Findings: {summary.empirical_findings}\n\n"
            f"RETRIEVED CONTEXT CHUNKS:\n{context_text}\n\n"
            f"Task: Evaluate how relevant this candidate paper is to the seed paper topic and how strictly the summary claims are supported by verbatim text chunks.\n\n"
            f"Output JSON conforming strictly to schema:\n"
            f"{{\n"
            f'  "is_relevant_to_seed_topic": boolean,\n'
            f'  "relevance_score": float (0.0 to 10.0),\n'
            f'  "factual_grounding_score": float (0.0 to 10.0),\n'
            f'  "critique_rationale": "1-2 sentence justification",\n'
            f'  "verdict": "accepted" | "marginal" | "rejected"\n'
            f"}}\n"
        )

        for attempt in range(1, self.max_attempts + 1):
            try:
                logger.info(f"Generating self-critique for '{title}' (Attempt {attempt}/{self.max_attempts})...")
                critique = self.backend.generate_critique(system_prompt, user_prompt)
                return critique
            except Exception as exc:
                logger.warning(f"Self-critique generation failed on attempt {attempt}: {exc}")

        # Fail SAFE, not open: if the critique step itself never produced a real
        # judgment after all attempts, that is not evidence of relevance. The
        # previous default (is_relevant_to_seed_topic=True, verdict="marginal",
        # 6.0/6.0) fabricated a passing score whenever the LLM's critique output
        # merely failed to parse -- which let candidates with zero actual
        # relevance judgment slip through is_accepted's threshold check in
        # llm/quota_manager.py. A failed critique must reject the candidate so
        # the quota loop fetches a genuine replacement instead.
        logger.warning(
            f"Self-critique generation failed for '{title}' after {self.max_attempts} attempts; "
            f"defaulting to rejected rather than fabricating a passing score."
        )
        return SelfCritiqueAssessment(
            is_relevant_to_seed_topic=False,
            relevance_score=0.0,
            factual_grounding_score=0.0,
            critique_rationale=f"Self-critique generation failed after {self.max_attempts} attempts; defaulted to rejected.",
            verdict="rejected"
        )
