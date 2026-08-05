import json
import re
from typing import Dict, Any, Optional
from lea.llm.schemas import TechnicalSummary
from lea.exceptions import SummaryValidationError
from lea.logging import logger

def clean_json_response(raw_text: str) -> str:
    # Remove markdown code blocks if present
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw_text.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip(), flags=re.MULTILINE)
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
        cleaned = clean_json_response(raw_response)

        try:
            data = json.loads(cleaned)
            return TechnicalSummary(**data)
        except Exception as exc:
            raise SummaryValidationError(f"Failed to parse TechnicalSummary JSON: {exc} | Raw text: {raw_response[:300]}")
