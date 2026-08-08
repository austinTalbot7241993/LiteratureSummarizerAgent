import os
import gc
import json
import re
from typing import Dict, Any, Optional, List
from lea.llm.schemas import (
    TechnicalSummary,
    DataAvailabilityAssessment,
    PaperAvailabilityStatus,
    DatasetAvailability,
    DatasetAvailabilityStatus,
    AvailabilityEvidence,
    DatasetRole,
    DatasetOwnership,
    VerificationStatus
)
from lea.exceptions import SummaryValidationError
from lea.logging import logger


def clean_json_response(raw_text: str) -> str:
    cleaned = raw_text.strip()
    # Remove markdown code blocks if present
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE)

    # Extract JSON object substring if model surrounded JSON with prose
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        return match.group(0).strip()
    return cleaned.strip()


def extract_chunks_from_prompt(user_prompt: str) -> List[Dict[str, Any]]:
    """Parses [CHUNK id=<uuid> index=<int> ...] blocks from the formatted user prompt for evidence quote validation."""
    chunk_blocks = re.findall(r"\[CHUNK\s+id=([^\s\]]+)[^\]]*\]\n([\s\S]*?)(?=\n\n\[CHUNK|\s*$)", user_prompt)
    chunks = []
    for cid, content in chunk_blocks:
        chunks.append({
            "id": cid.strip(),
            "content": content.strip()
        })
    return chunks


def validate_evidence_quotes(assessment: DataAvailabilityAssessment, retrieved_chunks: List[Dict[str, Any]]) -> None:
    if not retrieved_chunks:
        return

    chunk_map = {str(c.get("id")): c.get("content", "") for c in retrieved_chunks if c.get("id")}
    
    for dataset in assessment.datasets:
        for ev in dataset.evidence:
            cid = str(ev.source_chunk_id)
            if cid not in chunk_map:
                raise ValueError(f"Evidence quote refers to unknown source_chunk_id '{cid}'. Valid chunk IDs: {list(chunk_map.keys())}")
            
            source_content = chunk_map[cid]
            norm_quote = " ".join(ev.quote.strip().split()).lower()
            norm_source = " ".join(source_content.strip().split()).lower()
            
            if norm_quote not in norm_source:
                raise ValueError(f"Evidence quote '{ev.quote}' was not found in source chunk '{cid}'.")


class BaseLLMBackend:
    def generate_summary(self, system_prompt: str, user_prompt: str) -> TechnicalSummary:
        raise NotImplementedError

    def generate_data_availability(self, system_prompt: str, user_prompt: str) -> DataAvailabilityAssessment:
        raise NotImplementedError


class MockLLMBackend(BaseLLMBackend):
    def __init__(
        self,
        preset: str = "publicly_available",
        custom_summary: Optional[TechnicalSummary] = None,
        custom_assessment: Optional[DataAvailabilityAssessment] = None
    ):
        self.preset = preset
        self.custom_summary = custom_summary
        self.custom_assessment = custom_assessment

    def generate_summary(self, system_prompt: str, user_prompt: str) -> TechnicalSummary:
        if self.custom_summary:
            return self.custom_summary

        match = re.search(r"Paper Title:\s*(.+)", user_prompt)
        title = match.group(1).strip() if match else "Target Paper"

        status_map = {
            "publicly_available": PaperAvailabilityStatus.PUBLICLY_AVAILABLE,
            "public": PaperAvailabilityStatus.PUBLICLY_AVAILABLE,
            "restricted": PaperAvailabilityStatus.RESTRICTED,
            "not_available": PaperAvailabilityStatus.NOT_AVAILABLE,
            "not_reported": PaperAvailabilityStatus.NOT_REPORTED,
            "mixed": PaperAvailabilityStatus.MIXED,
            "unclear": PaperAvailabilityStatus.UNCLEAR
        }
        status = status_map.get(self.preset, PaperAvailabilityStatus.PUBLICLY_AVAILABLE)
        location = "https://www.internationalgenome.org" if status == PaperAvailabilityStatus.PUBLICLY_AVAILABLE else None

        match_target = re.search(r"Target Input Paper:\s*(.+)", user_prompt)
        target_title = match_target.group(1).strip() if match_target else "Target Input Paper"

        return TechnicalSummary(
            problem_formulation=f"Formulates scientific evaluation for paper '{title}'.",
            methodological_novelty="Introduces a novel hybrid retrieval and citation exclusion methodology.",
            empirical_findings="Achieves quantifiable empirical performance improvements across benchmarks.",
            paragraph_summary=f"This paper '{title}' proposes a novel technical methodology addressing key challenges in literature synthesis.",
            relationship_to_target=f"Compares with target input paper '{target_title}' as a complementary approach and benchmark reference.",
            data_availability=status,
            data_location=location
        )

    def generate_data_availability(self, system_prompt: str, user_prompt: str) -> DataAvailabilityAssessment:
        if self.custom_assessment:
            return self.custom_assessment

        if self.preset == "malformed":
            raise SummaryValidationError("Mock backend configured to simulate malformed JSON output.")

        chunks = extract_chunks_from_prompt(user_prompt)
        cid = chunks[0]["id"] if chunks else "chunk-1"
        ccontent = chunks[0]["content"] if chunks else "Data are available."
        quote = ccontent[:min(50, len(ccontent))] if ccontent else "Data are available."

        if self.preset in ("publicly_available", "public"):
            return DataAvailabilityAssessment(
                overall_status=PaperAvailabilityStatus.PUBLICLY_AVAILABLE,
                datasets=[
                    DatasetAvailability(
                        dataset_name="Primary Dataset",
                        role=DatasetRole.PRIMARY,
                        status=DatasetAvailabilityStatus.PUBLICLY_AVAILABLE,
                        url="https://www.internationalgenome.org",
                        repository="1000 Genomes",
                        evidence=[
                            AvailabilityEvidence(
                                source_chunk_id=cid,
                                quote=quote
                            )
                        ]
                    )
                ],
                rationale="Data deposited in public repository."
            )
        elif self.preset == "restricted":
            return DataAvailabilityAssessment(
                overall_status=PaperAvailabilityStatus.RESTRICTED,
                datasets=[
                    DatasetAvailability(
                        dataset_name="Clinical Cohort",
                        role=DatasetRole.PRIMARY,
                        status=DatasetAvailabilityStatus.RESTRICTED,
                        access_conditions="Available upon reasonable request to corresponding author.",
                        evidence=[
                            AvailabilityEvidence(
                                source_chunk_id=cid,
                                quote=quote
                            )
                        ]
                    )
                ],
                rationale="Restricted access upon reasonable request."
            )
        elif self.preset == "not_available":
            return DataAvailabilityAssessment(
                overall_status=PaperAvailabilityStatus.NOT_AVAILABLE,
                datasets=[
                    DatasetAvailability(
                        dataset_name="Patient Records",
                        role=DatasetRole.PRIMARY,
                        status=DatasetAvailabilityStatus.NOT_AVAILABLE,
                        access_conditions="Data cannot be shared due to patient privacy.",
                        evidence=[
                            AvailabilityEvidence(
                                source_chunk_id=cid,
                                quote=quote if ("cannot" in quote.lower() or "not" in quote.lower()) else "Data cannot be shared due to privacy."
                            )
                        ]
                    )
                ],
                rationale="Patient privacy prevents data sharing."
            )
        elif self.preset == "not_reported":
            return DataAvailabilityAssessment(
                overall_status=PaperAvailabilityStatus.NOT_REPORTED,
                datasets=[],
                rationale="No data availability statement reported in paper."
            )
        elif self.preset == "mixed":
            return DataAvailabilityAssessment(
                overall_status=PaperAvailabilityStatus.MIXED,
                datasets=[
                    DatasetAvailability(
                        dataset_name="Genomic Data",
                        role=DatasetRole.PRIMARY,
                        status=DatasetAvailabilityStatus.PUBLICLY_AVAILABLE,
                        url="https://www.ncbi.nlm.nih.gov/geo/GSE12345",
                        accession="GSE12345",
                        evidence=[
                            AvailabilityEvidence(
                                source_chunk_id=cid,
                                quote=quote
                            )
                        ]
                    ),
                    DatasetAvailability(
                        dataset_name="Clinical Records",
                        role=DatasetRole.VALIDATION,
                        status=DatasetAvailabilityStatus.RESTRICTED,
                        access_conditions="Available after institutional approval and DUA.",
                        evidence=[
                            AvailabilityEvidence(
                                source_chunk_id=cid,
                                quote=quote
                            )
                        ]
                    )
                ],
                rationale="Genomic data public, clinical records restricted."
            )
        else:
            return DataAvailabilityAssessment(
                overall_status=PaperAvailabilityStatus.RESTRICTED,
                datasets=[
                    DatasetAvailability(
                        dataset_name="Study Dataset",
                        role=DatasetRole.PRIMARY,
                        status=DatasetAvailabilityStatus.RESTRICTED,
                        access_conditions="Available from the corresponding author upon reasonable request.",
                        evidence=[
                            AvailabilityEvidence(
                                source_chunk_id=cid,
                                quote=quote
                            )
                        ]
                    )
                ],
                rationale="Data available upon reasonable request."
            )


class TransformersPeftBackend(BaseLLMBackend):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        adapter_path: Optional[str] = None,
        max_context_tokens: int = 1800
    ):
        self.model_name = model_name
        self.adapter_path = adapter_path
        self.max_context_tokens = max_context_tokens
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        if self._model is not None:
            return

        import os
        import gc
        import torch
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        logger.info(f"Loading {self.model_name} in 4-bit NF4 precision (FP16 compute)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            llm_int8_enable_fp32_cpu_offload=True
        )

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )

        if self.adapter_path:
            from peft import PeftModel
            logger.info(f"Loading LoRA adapter weights from {self.adapter_path}...")
            self._model = PeftModel.from_pretrained(self._model, self.adapter_path)

    def _trim_user_prompt_chunks(self, user_prompt: str, max_user_tokens: int) -> str:
        """Trims context in user prompt strictly at complete chunk boundaries."""
        user_tokens = self._tokenizer.encode(user_prompt, add_special_tokens=False)
        if len(user_tokens) <= max_user_tokens:
            return user_prompt

        # Split user prompt into header/instruction parts and chunk list
        parts = user_prompt.split("Retrieved Context Chunks:")
        if len(parts) != 2:
            parts = user_prompt.split("Retrieved Evidence Chunks:")

        if len(parts) == 2:
            header_prefix = parts[0] + ("Retrieved Context Chunks:\n" if "Retrieved Context Chunks:" in user_prompt else "Retrieved Evidence Chunks:\n")
            chunks_text = parts[1]
            raw_chunks = chunks_text.split("\n\n[CHUNK")
            kept_chunks = []
            
            for i, chunk_block in enumerate(raw_chunks):
                candidate_block = ("[CHUNK" if i > 0 else "") + chunk_block
                test_prompt = header_prefix + "\n\n".join(kept_chunks + [candidate_block])
                if len(self._tokenizer.encode(test_prompt, add_special_tokens=False)) > max_user_tokens:
                    break
                kept_chunks.append(candidate_block)
            
            if kept_chunks:
                return header_prefix + "\n\n".join(kept_chunks)

        # Fallback if structure is non-standard
        return self._tokenizer.decode(user_tokens[:max_user_tokens], skip_special_tokens=True)

    def generate_summary(self, system_prompt: str, user_prompt: str) -> TechnicalSummary:
        self._load_model()
        import torch

        system_tokens = self._tokenizer.encode(system_prompt, add_special_tokens=False)
        max_user_tokens = max(300, self.max_context_tokens - len(system_tokens) - 100)
        user_prompt = self._trim_user_prompt_chunks(user_prompt, max_user_tokens)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        text_input = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(text_input, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=600,
                do_sample=False
            )

        output_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        raw_response = self._tokenizer.decode(output_tokens, skip_special_tokens=True)
        del inputs, outputs
        torch.cuda.empty_cache()
        gc.collect()
        logger.info(f"Raw Technical Summary LLM Output ({len(raw_response)} chars):\n{raw_response}")

        match_title = re.search(r"Paper Title:\s*(.+)", user_prompt)
        t_str = match_title.group(1).strip() if match_title else "the paper"

        cleaned = clean_json_response(raw_response)
        try:
            data = json.loads(cleaned)
            if isinstance(data, dict) and all(k in data for k in ["problem_formulation", "methodological_novelty", "empirical_findings", "paragraph_summary", "relationship_to_target"]):
                return TechnicalSummary(**data)
        except Exception:
            pass

        patterns = {
            "problem_formulation": r"(?:PROBLEM FORMULATION|PROBLEM STATEMENT|PROBLEM)\s*:\s*(.*?)(?=\n\s*(?:\*\*|\#\#|\#)?\s*(?:METHODOLOGICAL NOVELTY|METHODOLOGY|EMPIRICAL FINDINGS|RESULTS|TECHNICAL SYNTHESIS|SYNTHESIS|RELATIONSHIP TO TARGET PAPER|RELATIONSHIP TO TARGET)\s*:|\s*$)",
            "methodological_novelty": r"(?:METHODOLOGICAL NOVELTY|METHODOLOGY|NOVELTY)\s*:\s*(.*?)(?=\n\s*(?:\*\*|\#\#|\#)?\s*(?:EMPIRICAL FINDINGS|RESULTS|TECHNICAL SYNTHESIS|SYNTHESIS|RELATIONSHIP TO TARGET PAPER|RELATIONSHIP TO TARGET)\s*:|\s*$)",
            "empirical_findings": r"(?:EMPIRICAL FINDINGS|RESULTS|EMPIRICAL EVALUATION)\s*:\s*(.*?)(?=\n\s*(?:\*\*|\#\#|\#)?\s*(?:TECHNICAL SYNTHESIS|SYNTHESIS|SUMMARY|RELATIONSHIP TO TARGET PAPER|RELATIONSHIP TO TARGET)\s*:|\s*$)",
            "paragraph_summary": r"(?:TECHNICAL SYNTHESIS|SYNTHESIS|SUMMARY)\s*:\s*(.*?)(?=\n\s*(?:\*\*|\#\#|\#)?\s*(?:RELATIONSHIP TO TARGET PAPER|RELATIONSHIP TO TARGET|RELATIONSHIP TO INPUT PAPER|RELATIONSHIP)\s*:|\s*$)",
            "relationship_to_target": r"(?:RELATIONSHIP TO TARGET PAPER|RELATIONSHIP TO TARGET|RELATIONSHIP TO INPUT PAPER|RELATIONSHIP)\s*:\s*(.*?)(?=\s*$)"
        }

        parsed_fields = {}
        for key, pat in patterns.items():
            m = re.search(pat, raw_response, re.IGNORECASE | re.DOTALL)
            if m:
                val = m.group(1).strip()
                val_clean = " ".join(val.split())
                if len(val_clean) > 5:
                    parsed_fields[key] = val_clean

        missing = [k for k in ["problem_formulation", "methodological_novelty", "empirical_findings", "paragraph_summary", "relationship_to_target"] if k not in parsed_fields]
        if missing:
            raise SummaryValidationError(
                f"LLM output failed to produce required section(s) {missing} for paper '{t_str}'. "
                f"Raw response generated by model:\n{raw_response[:600]}"
            )

        words = parsed_fields["paragraph_summary"].split()
        if len(words) > 300:
            parsed_fields["paragraph_summary"] = " ".join(words[:290])

        # Note: data_availability will be provided via dedicated generate_data_availability call
        parsed_fields["data_availability"] = PaperAvailabilityStatus.UNCLEAR

        return TechnicalSummary(**parsed_fields)

    def generate_data_availability(self, system_prompt: str, user_prompt: str) -> DataAvailabilityAssessment:
        self._load_model()
        import torch

        retrieved_chunks = extract_chunks_from_prompt(user_prompt)

        system_tokens = self._tokenizer.encode(system_prompt, add_special_tokens=False)
        max_user_tokens = max(300, self.max_context_tokens - len(system_tokens) - 100)
        trimmed_user_prompt = self._trim_user_prompt_chunks(user_prompt, max_user_tokens)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": trimmed_user_prompt}
        ]

        text_input = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(text_input, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=800,
                do_sample=False
            )

        output_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        raw_response = self._tokenizer.decode(output_tokens, skip_special_tokens=True)
        del inputs, outputs
        torch.cuda.empty_cache()
        logger.info(f"Raw Data Availability LLM Output ({len(raw_response)} chars):\n{raw_response}")

        cleaned = clean_json_response(raw_response)
        
        # 1st attempt: parse JSON and validate Pydantic & quotes
        try:
            data = json.loads(cleaned)
            assessment = DataAvailabilityAssessment(**data)
            validate_evidence_quotes(assessment, retrieved_chunks)
            return assessment
        except Exception as exc1:
            logger.warning(f"Data availability extraction validation failed on initial attempt: {exc1}. Attempting 1-step repair retry...")

            # 1-step repair retry request
            repair_user_prompt = (
                f"{trimmed_user_prompt}\n\n"
                f"CRITICAL REPAIR DIRECTIVE: Your previous output was invalid.\n"
                f"Validation Error: {exc1}\n"
                f"Previous Raw Output: {raw_response[:400]}\n"
                f"RULES TO FIX VALIDATION ERROR:\n"
                f"- If an accession number (e.g. SRA:SRP004777, GSE12345) or URL is present, dataset status MUST be 'publicly_available'.\n"
                f"- 'not_available' status requires explicit refusal words (e.g., 'cannot', 'privacy', 'restricted', 'not available').\n"
                f"- 'not_reported' status must NOT have evidence with accession numbers or data availability mentions.\n"
                f"Return ONLY a single valid JSON object adhering strictly to DataAvailabilityAssessment schema."
            )

            repair_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": repair_user_prompt}
            ]

            repair_input = self._tokenizer.apply_chat_template(repair_messages, tokenize=False, add_generation_prompt=True)
            repair_inputs = self._tokenizer(repair_input, return_tensors="pt").to("cuda")

            with torch.no_grad():
                repair_outputs = self._model.generate(
                    **repair_inputs,
                    max_new_tokens=800,
                    do_sample=False
                )

            repair_tokens = repair_outputs[0][repair_inputs["input_ids"].shape[1]:]
            repair_response = self._tokenizer.decode(repair_tokens, skip_special_tokens=True)
            del repair_inputs, repair_outputs
            torch.cuda.empty_cache()
            gc.collect()
            logger.info(f"Raw Repair LLM Output ({len(repair_response)} chars):\n{repair_response}")

            cleaned_repair = clean_json_response(repair_response)
            try:
                repair_data = json.loads(cleaned_repair)
                assessment = DataAvailabilityAssessment(**repair_data)
                validate_evidence_quotes(assessment, retrieved_chunks)
                return assessment
            except Exception as exc2:
                raise SummaryValidationError(
                    f"Data availability extraction failed after repair retry: {exc2}. Raw repair output:\n{repair_response[:600]}"
                )
