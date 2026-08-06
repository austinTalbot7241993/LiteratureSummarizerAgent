import json
import re
from typing import Dict, Any, Optional
from lea.llm.schemas import TechnicalSummary
from lea.exceptions import SummaryValidationError
from lea.logging import logger

def parse_data_availability_statement(da_text: str) -> tuple:
    if not da_text:
        return "proprietary", None

    da_lower = da_text.lower()
    first_line = da_text.strip().splitlines()[0].lower() if da_text.strip() else ""

    # Check for negations first (e.g., "not publicly available", "preventing public access")
    negation_patterns = [
        r"not\s+(?:publicly\s+)?available",
        r"no\s+public",
        r"preventing?\s+public",
        r"non[- ]public",
        r"unavailable",
        r"cannot\s+be\s+shared"
    ]
    is_negated = any(re.search(pat, da_lower) for pat in negation_patterns)

    # Entity-based domain rules
    restricted_keywords = [
        "uk biobank", "ukbiobank", "dbgap", "ega", "european genome-phenome archive",
        "biovu", "all of us", "controlled access", "data access committee", "dac",
        "permission", "application required", "managed access", "authorization",
        "access control", "restricted access", "restricted"
    ]

    public_keywords = [
        "1000 genomes", "1kgp", "geo", "gse", "sra", "srp", "srr", "ena",
        "genbank", "github", "huggingface", "zenodo", "figshare", "open science framework",
        "osf", "ncbi", "ebi", "publicly_available", "publicly available"
    ]

    status = "proprietary"

    if any(term in da_lower for term in restricted_keywords) or "restrict" in first_line:
        status = "restricted"
    elif not is_negated and (any(term in da_lower for term in public_keywords) or "public" in first_line):
        status = "publicly_available"
    elif "propriet" in da_lower or "private" in da_lower or is_negated:
        status = "proprietary"
    else:
        status = "proprietary"

    # Location extraction (ONLY if publicly_available)
    location = None
    if status == "publicly_available":
        url_match = re.search(r"https?://\S+", da_text)
        if url_match:
            location = url_match.group(0).rstrip(".,;)")
        else:
            accession_match = re.search(r"\b(?:GSE|SRP|SRR|PRJN|PRJEB|PRJD|ERP)\d+\b", da_text, re.IGNORECASE)
            if accession_match:
                location = accession_match.group(0)
            else:
                loc_match = re.search(r"(?:location|accession|repository|repo|at|via|url)\s*[:=]\s*([^.,;\n]+)", da_text, re.IGNORECASE)
                if loc_match and len(loc_match.group(1).strip()) > 3:
                    location = loc_match.group(1).strip()
                elif "1000 genomes" in da_lower:
                    location = "1000 Genomes Project (https://www.internationalgenome.org)"
                elif "github" in da_lower:
                    location = "GitHub"

    return status, location

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

class BaseLLMBackend:
    def generate_summary(self, system_prompt: str, user_prompt: str) -> TechnicalSummary:
        raise NotImplementedError

class MockLLMBackend(BaseLLMBackend):
    def generate_summary(self, system_prompt: str, user_prompt: str) -> TechnicalSummary:
        # Extract title from prompt if possible
        match = re.search(r"Paper Title:\s*(.+)", user_prompt)
        title = match.group(1).strip() if match else "Target Paper"

        return TechnicalSummary(
            problem_formulation=f"Formulates scientific evaluation for paper '{title}'.",
            methodological_novelty="Introduces a novel hybrid retrieval and citation exclusion methodology.",
            empirical_findings="Achieves quantifiable empirical performance improvements across benchmarks.",
            paragraph_summary=f"This paper '{title}' proposes a novel technical methodology addressing key challenges in literature synthesis. Through rigorous empirical evaluation, the authors demonstrate superior performance over standard baselines while maintaining strict exclusion constraints.",
            data_availability="publicly_available",
            data_location="https://www.internationalgenome.org"
        )

class TransformersPeftBackend(BaseLLMBackend):
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct", adapter_path: Optional[str] = None, max_context_tokens: int = 1800):
        self.model_name = model_name
        self.adapter_path = adapter_path
        self.max_context_tokens = max_context_tokens
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        if self._model is not None:
            return

        import torch
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

    def generate_summary(self, system_prompt: str, user_prompt: str) -> TechnicalSummary:
        self._load_model()
        import torch

        system_tokens = self._tokenizer.encode(system_prompt, add_special_tokens=False)
        max_user_tokens = max(200, self.max_context_tokens - len(system_tokens) - 100)
        user_tokens = self._tokenizer.encode(user_prompt, add_special_tokens=False)
        if len(user_tokens) > max_user_tokens:
            user_prompt = self._tokenizer.decode(user_tokens[:max_user_tokens], skip_special_tokens=True)

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
                temperature=0.1,
                top_p=0.9,
                do_sample=True
            )

        output_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        raw_response = self._tokenizer.decode(output_tokens, skip_special_tokens=True)
        logger.info(f"Raw LLM Output ({len(raw_response)} chars):\n{raw_response}")

        match_title = re.search(r"Paper Title:\s*(.+)", user_prompt)
        t_str = match_title.group(1).strip() if match_title else "the paper"

        # Try parsing JSON first if model generated valid JSON
        cleaned = clean_json_response(raw_response)
        try:
            data = json.loads(cleaned)
            if isinstance(data, dict) and all(k in data for k in ["problem_formulation", "methodological_novelty", "empirical_findings", "paragraph_summary"]):
                return TechnicalSummary(**data)
        except Exception:
            pass

        # Parse labeled section headers from prose
        patterns = {
            "problem_formulation": r"(?:PROBLEM FORMULATION|PROBLEM STATEMENT|PROBLEM)\s*:\s*(.*?)(?=\n\s*(?:\*\*|\#\#|\#)?\s*(?:METHODOLOGICAL NOVELTY|METHODOLOGY|EMPIRICAL FINDINGS|RESULTS|DATA AVAILABILITY|TECHNICAL SYNTHESIS|SYNTHESIS)\s*:|\s*$)",
            "methodological_novelty": r"(?:METHODOLOGICAL NOVELTY|METHODOLOGY|NOVELTY)\s*:\s*(.*?)(?=\n\s*(?:\*\*|\#\#|\#)?\s*(?:EMPIRICAL FINDINGS|RESULTS|DATA AVAILABILITY|TECHNICAL SYNTHESIS|SYNTHESIS)\s*:|\s*$)",
            "empirical_findings": r"(?:EMPIRICAL FINDINGS|RESULTS|EMPIRICAL EVALUATION)\s*:\s*(.*?)(?=\n\s*(?:\*\*|\#\#|\#)?\s*(?:DATA AVAILABILITY|TECHNICAL SYNTHESIS|SYNTHESIS)\s*:|\s*$)",
            "data_availability_raw": r"(?:DATA AVAILABILITY|DATA ACCESSIBILITY|DATA ACCESS)\s*:\s*(.*?)(?=\n\s*(?:\*\*|\#\#|\#)?\s*(?:TECHNICAL SYNTHESIS|SYNTHESIS|SUMMARY)\s*:|\s*$)",
            "paragraph_summary": r"(?:TECHNICAL SYNTHESIS|SYNTHESIS|SUMMARY)\s*:\s*(.*?)(?=\s*$)"
        }

        parsed_fields = {}
        for key, pat in patterns.items():
            m = re.search(pat, raw_response, re.IGNORECASE | re.DOTALL)
            if m:
                val = m.group(1).strip()
                val_clean = " ".join(val.split())
                if len(val_clean) > 5:
                    parsed_fields[key] = val_clean

        missing = [k for k in ["problem_formulation", "methodological_novelty", "empirical_findings", "paragraph_summary"] if k not in parsed_fields]
        if missing:
            raise SummaryValidationError(
                f"LLM output failed to produce required section(s) {missing} for paper '{t_str}'. "
                f"Raw response generated by model:\n{raw_response[:600]}"
            )

        # Parse data availability classification & location
        da_text = parsed_fields.pop("data_availability_raw", "")
        status, location = parse_data_availability_statement(da_text)

        parsed_fields["data_availability"] = status
        parsed_fields["data_location"] = location

        # Truncate paragraph_summary to <= 300 words if necessary
        words = parsed_fields["paragraph_summary"].split()
        if len(words) > 300:
            parsed_fields["paragraph_summary"] = " ".join(words[:290])

        return TechnicalSummary(**parsed_fields)
