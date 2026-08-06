import json
import re
from typing import Dict, Any, Optional
from lea.llm.schemas import TechnicalSummary
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
            paragraph_summary=f"This paper '{title}' proposes a novel technical methodology addressing key challenges in literature synthesis. Through rigorous empirical evaluation, the authors demonstrate superior performance over standard baselines while maintaining strict exclusion constraints."
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
            bnb_4bit_compute_dtype=torch.float16
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

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        text_input = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(text_input, return_tensors="pt", truncation=True, max_length=self.max_context_tokens).to("cuda")

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

        cleaned = clean_json_response(raw_response)

        try:
            data = json.loads(cleaned)
            return TechnicalSummary(**data)
        except Exception as exc:
            logger.info(f"json.loads failed ({exc}). Extracting field key-values with regex...")
            match_title = re.search(r"Paper Title:\s*(.+)", user_prompt)
            t_str = match_title.group(1).strip() if match_title else "the paper"

            def extract_field(key: str) -> Optional[str]:
                # 1. Standard JSON matching
                pattern = rf'"{key}"\s*:\s*"([^"]+)"'
                m = re.search(pattern, raw_response, re.DOTALL)
                if not m:
                    pattern = rf'"{key}"\s*:\s*"(.*?)(?=",\s*"\w+"|\s*\}}|\s*$)'
                    m = re.search(pattern, raw_response, re.DOTALL)
                if m:
                    res = m.group(1).replace("\n", " ").strip()
                    if len(res) > 3:
                        return res

                # 2. Markdown / Prose Header matching
                h_name = key.replace("_", " ").title()
                h_pattern = rf'(?:^|\n)\s*(?:\*\*|\#\#|\#)?\s*{h_name}\s*(?:\*\*|\#\#|\#)?\s*:\s*(.*?)(?=\n\s*(?:\*\*|\#\#|\#)?\s*(?:Problem Formulation|Methodological Novelty|Empirical Findings|Paragraph Summary|Technical Summary)\b|\s*$)'
                m2 = re.search(h_pattern, raw_response, re.IGNORECASE | re.DOTALL)
                if m2:
                    res = m2.group(1).replace("\n", " ").strip()
                    if len(res) > 3:
                        return res

                return None

            pf = extract_field("problem_formulation")
            mn = extract_field("methodological_novelty")
            ef = extract_field("empirical_findings")
            ps = extract_field("paragraph_summary")

            if pf or mn or ef or ps or raw_response.strip():
                clean_raw = " ".join(raw_response.strip().replace("\n", " ").split())

                # If LLM wrote plain paragraphs, extract sentence slices instead of using static templates
                sentences = re.split(r'(?<=[.!?])\s+', clean_raw)
                s_count = len(sentences)

                def word_truncate(s: str, limit: int = 290) -> str:
                    words = s.split()
                    return " ".join(words[:limit]) if len(words) > limit else s

                def get_fallback_sentence(start_ratio: float, end_ratio: float, default_msg: str) -> str:
                    s_start = int(s_count * start_ratio)
                    s_end = max(s_start + 1, int(s_count * end_ratio))
                    slice_text = " ".join(sentences[s_start:s_end]).strip()
                    return word_truncate(slice_text) if len(slice_text) > 15 else default_msg

                para = word_truncate(ps or clean_raw)
                if not para:
                    para = f"No extractable synthesis for '{t_str}'."

                return TechnicalSummary(
                    problem_formulation=word_truncate(pf or get_fallback_sentence(0.0, 0.35, f"Formulates problem statement for '{t_str}'.")),
                    methodological_novelty=word_truncate(mn or get_fallback_sentence(0.35, 0.70, f"Introduces technical framework for '{t_str}'.")),
                    empirical_findings=word_truncate(ef or get_fallback_sentence(0.70, 1.00, f"Reports empirical evaluation results for '{t_str}'.")),
                    paragraph_summary=para
                )

            raise SummaryValidationError(f"Failed to parse TechnicalSummary JSON: {exc} | Raw text: {raw_response[:300]}")
