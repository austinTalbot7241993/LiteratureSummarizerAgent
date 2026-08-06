from typing import List, Dict, Any, Optional
from lea.llm.schemas import TechnicalSummary
from lea.llm.backends import BaseLLMBackend, MockLLMBackend, TransformersPeftBackend
from lea.rag.prompts import SUMMARY_SYSTEM_PROMPT, SUMMARY_USER_PROMPT_TEMPLATE
from lea.exceptions import SummaryValidationError
from lea.logging import logger

class TechnicalSummarizer:
    def __init__(self, backend: Optional[BaseLLMBackend] = None, max_attempts: int = 2):
        self.backend = backend or MockLLMBackend()
        self.max_attempts = max_attempts

    def summarize_candidate(
        self,
        candidate_meta: Dict[str, Any],
        retrieved_chunks: List[Dict[str, Any]]
    ) -> TechnicalSummary:
        title = candidate_meta.get("title", "Untitled")
        authors = ", ".join(candidate_meta.get("authors", [])) or "Unknown"
        year = str(candidate_meta.get("publication_year") or candidate_meta.get("year") or "N/A")

        chunk_texts = [f"[{i+1}] {c.get('content')}" for i, c in enumerate(retrieved_chunks)]
        context_text = "\n\n".join(chunk_texts) if chunk_texts else "No full text chunks available. Title and abstract only."

        user_prompt = SUMMARY_USER_PROMPT_TEMPLATE.format(
            title=title,
            authors=authors,
            year=year,
            context_text=context_text
        )

        last_error = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                if attempt > 1:
                    # On retry, reduce context size and append explicit formatting directive
                    truncated_chunks = retrieved_chunks[: max(1, len(retrieved_chunks) // 2)]
                    c_texts = [f"[{i+1}] {c.get('content')}" for i, c in enumerate(truncated_chunks)]
                    c_str = "\n\n".join(c_texts) if c_texts else "No full text chunks available."
                    user_prompt = SUMMARY_USER_PROMPT_TEMPLATE.format(
                        title=title,
                        authors=authors,
                        year=year,
                        context_text=c_str
                    ) + "\n\nCRITICAL RETRY DIRECTIVE: You MUST output all 4 section headers: PROBLEM FORMULATION:, METHODOLOGICAL NOVELTY:, EMPIRICAL FINDINGS:, TECHNICAL SYNTHESIS:."

                logger.info(f"Generating summary for '{title}' (Attempt {attempt}/{self.max_attempts})...")
                summary = self.backend.generate_summary(SUMMARY_SYSTEM_PROMPT, user_prompt)
                return summary
            except SummaryValidationError as exc:
                logger.warning(f"Summary validation failed on attempt {attempt}: {exc}")
                last_error = exc
            except Exception as exc:
                logger.warning(f"Unexpected error during summary generation: {exc}")
                last_error = exc

        raise SummaryValidationError(f"Failed to generate valid TechnicalSummary after {self.max_attempts} attempts: {last_error}")
