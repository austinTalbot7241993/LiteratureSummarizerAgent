import json
import numpy as np
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ValidationError

from lea.logging import logger
from lea.config import LEAConfig


class AbstractRelevanceSchema(BaseModel):
    relevance_score: float = Field(..., ge=0.0, le=10.0)
    relevance_tier: str  # "high" | "moderate" | "low" | "irrelevant"
    reasoning: str


def compute_relevance_tier(score: float) -> str:
    if score >= 8.0:
        return "high"
    elif score >= 6.0:
        return "moderate"
    elif score >= 4.0:
        return "low"
    else:
        return "irrelevant"


class AbstractScreener:
    def __init__(
        self,
        config: Optional[LEAConfig] = None,
        llm_backend: Optional[Any] = None,
        embedder: Optional[Any] = None
    ):
        self.config = config
        self.llm_backend = llm_backend
        self.embedder = embedder

    async def screen_candidates(
        self,
        seed_paper_meta: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        method: str = "llm",
        min_score: float = 6.0,
        max_candidates: int = 10
    ) -> List[Dict[str, Any]]:
        """Screens candidate papers by abstract relevance against seed paper using LLM or Embeddings."""
        if not candidates:
            return []

        seed_title = seed_paper_meta.get("title", "")
        seed_abstract = seed_paper_meta.get("abstract", "")

        fallback_policy = "pass"
        if self.config and hasattr(self.config, "discovery") and hasattr(self.config.discovery, "screening"):
            fallback_policy = getattr(self.config.discovery.screening, "fallback_on_missing_abstract", "pass")

        screened_pool: List[Dict[str, Any]] = []

        # Handle candidates missing abstract
        candidates_to_score: List[Dict[str, Any]] = []
        for cand in candidates:
            c_copy = dict(cand)
            abstract_text = c_copy.get("abstract", "") or ""
            if not abstract_text.strip():
                if fallback_policy == "drop":
                    logger.info(f"Dropping candidate '{c_copy.get('title')}' due to missing abstract (fallback_policy='drop')")
                    c_copy["abstract_relevance_score"] = 0.0
                    c_copy["abstract_relevance_tier"] = "irrelevant"
                    c_copy["abstract_relevance_reasoning"] = "Candidate abstract missing; dropped by fallback strategy."
                    # We still track it so it's excluded by score threshold unless relaxation is needed
                    screened_pool.append(c_copy)
                else:
                    logger.info(f"Retaining candidate '{c_copy.get('title')}' with neutral score due to missing abstract (fallback_policy='pass')")
                    c_copy["abstract_relevance_score"] = 5.0
                    c_copy["abstract_relevance_tier"] = "moderate"
                    c_copy["abstract_relevance_reasoning"] = "Candidate abstract unavailable; retained by fallback strategy."
                    screened_pool.append(c_copy)
            else:
                candidates_to_score.append(c_copy)

        if not candidates_to_score:
            return self._finalize_selection(screened_pool, min_score, max_candidates)

        # Attempt scoring method
        scored_candidates: List[Dict[str, Any]] = []
        use_embedding_fallback = False

        if method == "llm":
            if self.llm_backend is not None:
                try:
                    for cand in candidates_to_score:
                        scored_cand = await self._score_with_llm(seed_title, seed_abstract, cand)
                        scored_candidates.append(scored_cand)
                except Exception as exc:
                    logger.warning(f"LLM abstract screening error: {exc}. Falling back to embedding mode.")
                    use_embedding_fallback = True
            else:
                logger.info("No LLM backend provided for LLM abstract screening. Falling back to embedding mode.")
                use_embedding_fallback = True
        else:
            use_embedding_fallback = True

        if use_embedding_fallback or method == "embedding":
            scored_candidates = await self._score_with_embedding(seed_title, seed_abstract, candidates_to_score)

        screened_pool.extend(scored_candidates)
        return self._finalize_selection(screened_pool, min_score, max_candidates)

    async def _score_with_llm(self, seed_title: str, seed_abstract: str, cand: Dict[str, Any]) -> Dict[str, Any]:
        system_prompt = "You are a strict academic literature relevance evaluator."
        user_prompt = (
            f"SEED PAPER TOPIC:\n"
            f"Title: {seed_title}\n"
            f"Abstract: {seed_abstract}\n\n"
            f"CANDIDATE PAPER TO EVALUATE:\n"
            f"Title: {cand.get('title', '')}\n"
            f"Abstract: {cand.get('abstract', '')}\n\n"
            f"Task: Evaluate how relevant the candidate paper is to researching or advancing the seed paper's scientific problem.\n\n"
            f"JSON Output Schema:\n"
            f"{{\n"
            f'  "relevance_score": float (0.0 to 10.0),\n'
            f'  "relevance_tier": "high" | "moderate" | "low" | "irrelevant",\n'
            f'  "reasoning": "1-2 sentence justification"\n'
            f"}}\n"
        )

        res_dict = None
        if hasattr(self.llm_backend, "generate_abstract_relevance"):
            res_dict = self.llm_backend.generate_abstract_relevance(system_prompt, user_prompt)
        elif hasattr(self.llm_backend, "generate"):
            raw_res = self.llm_backend.generate(system_prompt, user_prompt)
            from lea.llm.backends import clean_json_response
            cleaned = clean_json_response(raw_res)
            res_dict = json.loads(cleaned)

        if not res_dict or not isinstance(res_dict, dict):
            raise ValueError("LLM failed to return a valid dictionary for abstract relevance.")

        validated = AbstractRelevanceSchema(**res_dict)
        cand["abstract_relevance_score"] = float(validated.relevance_score)
        cand["abstract_relevance_tier"] = str(validated.relevance_tier)
        cand["abstract_relevance_reasoning"] = str(validated.reasoning)
        return cand

    async def _score_with_embedding(
        self,
        seed_title: str,
        seed_abstract: str,
        candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        seed_text = f"{seed_title} {seed_abstract}".strip()
        cand_texts = [f"{c.get('title', '')} {c.get('abstract', '') or ''}".strip() for c in candidates]

        if not self.embedder:
            from lea.rag.embedder import BGEEmbedder
            model_name = "BAAI/bge-m3"
            if self.config and hasattr(self.config, "embedding"):
                model_name = getattr(self.config.embedding, "model", "BAAI/bge-m3")
            self.embedder = BGEEmbedder(model_name=model_name)

        all_texts = [seed_text] + cand_texts
        all_embs = self.embedder.embed_texts(all_texts)

        if not all_embs or len(all_embs) != len(candidates) + 1:
            # Fallback if embedding fails
            for c in candidates:
                c["abstract_relevance_score"] = 5.0
                c["abstract_relevance_tier"] = "moderate"
                c["abstract_relevance_reasoning"] = "Embedding generation failed; defaulted to neutral score."
            return candidates

        seed_vec = np.array(all_embs[0])
        norm_seed = np.linalg.norm(seed_vec)

        for idx, cand in enumerate(candidates):
            c_vec = np.array(all_embs[idx + 1])
            norm_c = np.linalg.norm(c_vec)
            if norm_seed > 0 and norm_c > 0:
                similarity = float(np.dot(seed_vec, c_vec) / (norm_seed * norm_c))
            else:
                similarity = 0.0

            score = round(max(0.0, min(10.0, similarity * 10.0)), 2)
            tier = compute_relevance_tier(score)
            reasoning = f"Dense embedding cosine similarity score: {similarity:.3f} (scaled to {score}/10.0)."

            cand["abstract_relevance_score"] = score
            cand["abstract_relevance_tier"] = tier
            cand["abstract_relevance_reasoning"] = reasoning

        return candidates

    def _finalize_selection(
        self,
        candidates: List[Dict[str, Any]],
        min_score: float,
        max_candidates: int
    ) -> List[Dict[str, Any]]:
        passing_candidates = [
            c for c in candidates
            if c.get("abstract_relevance_score", 0.0) >= min_score
        ]

        if not passing_candidates:
            logger.warning(
                f"No candidates passed minimum relevance threshold {min_score}. "
                f"Relaxing threshold to retain top scoring candidates."
            )
            # Sort all candidates by score descending and take top max_candidates
            candidates.sort(key=lambda x: x.get("abstract_relevance_score", 0.0), reverse=True)
            return candidates[:max_candidates]

        passing_candidates.sort(key=lambda x: x.get("abstract_relevance_score", 0.0), reverse=True)
        return passing_candidates[:max_candidates]
