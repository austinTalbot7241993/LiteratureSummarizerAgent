import os
import re
from pathlib import Path
from typing import List, Optional
import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def _expand_env_vars(content: str) -> str:
    pattern = re.compile(r"\$\{([A-Za-z0-9_]+)(?::-([^}]*))?\}")
    def _replace(match):
        var_name = match.group(1)
        default_val = match.group(2)
        val = os.getenv(var_name)
        if val is not None and val != "":
            return val
        return default_val if default_val is not None else ""
    return pattern.sub(_replace, content)


class ApplicationConfig(BaseModel):
    log_level: str = "INFO"
    cache_dir: str = ".cache/lea"
    artifact_dir: str = "artifacts"
    request_timeout_seconds: int = 30
    max_retries: int = 4

class ServicesConfig(BaseModel):
    database_url: str = Field(default_factory=lambda: os.getenv("LEA_DATABASE_URL", "postgresql://lea_user:lea_pass@localhost:5433/lea_db"))
    grobid_url: str = Field(default_factory=lambda: os.getenv("LEA_GROBID_URL", "http://localhost:8070"))
    openalex_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("OPENALEX_API_KEY"))
    semantic_scholar_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("SEMANTIC_SCHOLAR_API_KEY"))
    unpaywall_email: Optional[str] = Field(default_factory=lambda: os.getenv("UNPAYWALL_EMAIL"))
    semantic_scholar_rate_limit_rps: float = 0.2
    openalex_rate_limit_rps: float = 5.0

class ExtractionConfig(BaseModel):
    require_complete_bibliography: bool = True
    allow_pymupdf_fallback: bool = True
    allow_openalex_reference_fallback: bool = True
    allow_incomplete_citation_exclusion: bool = False

class ScreeningConfig(BaseModel):
    enabled: bool = True
    method: str = "llm"
    pre_screening_limit: int = 50
    min_relevance_score: float = 6.0
    max_screened_candidates: int = 10
    fallback_on_missing_abstract: str = "pass"

class DiscoveryConfig(BaseModel):
    openalex_limit: int = 100
    semantic_scholar_limit: int = 100
    final_candidate_limit: int = 20
    source_rrf_k: int = 60
    title_similarity_threshold: float = 0.96
    year_tolerance: int = 1
    screening: ScreeningConfig = ScreeningConfig()

class AcquisitionConfig(BaseModel):
    download_open_access_pdfs: bool = True
    require_downloaded_pdf: bool = True
    max_pdf_bytes: int = 104857600
    allowed_content_types: List[str] = ["application/pdf"]
    user_agent: str = "LEA/0.1 scholarly-research-agent"

class ChunkingConfig(BaseModel):
    tokenizer_model: str = "Qwen/Qwen2.5-7B-Instruct"
    parent_tokens: int = 768
    parent_overlap_tokens: int = 96
    child_tokens: int = 256
    child_overlap_tokens: int = 48

class EmbeddingConfig(BaseModel):
    model: str = "BAAI/bge-m3"
    dimension: int = 1024
    device: str = "auto"
    batch_size_gpu: int = 4
    batch_size_cpu: int = 1
    normalize_embeddings: bool = True
    max_length: int = 1024

class RetrievalConfig(BaseModel):
    dense_top_k: int = 30
    sparse_top_k: int = 30
    fused_top_k: int = 8
    rrf_k: int = 60
    sparse_backend: str = "bm25s"
    scope_sparse_index_per_run: bool = True

class LLMConfig(BaseModel):
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    adapter_path: Optional[str] = None
    backend: str = "transformers_peft"
    load_in_4bit: bool = True
    quant_type: str = "nf4"
    use_double_quant: bool = True
    compute_dtype: str = "float16"
    device: str = "cuda"
    inference_max_model_length: int = 2048
    max_context_tokens: int = 1800
    max_new_tokens: int = 600
    temperature: float = 0.1
    top_p: float = 0.9
    generation_attempts: int = 2

class ReportConfig(BaseModel):
    title: str = "Literature Exploration Report"
    include_abstract_only_results: bool = False
    include_retrieval_provenance: bool = True
    include_bibtex: bool = True

class LEAConfig(BaseModel):
    application: ApplicationConfig = ApplicationConfig()
    services: ServicesConfig = ServicesConfig()
    extraction: ExtractionConfig = ExtractionConfig()
    discovery: DiscoveryConfig = DiscoveryConfig()
    acquisition: AcquisitionConfig = AcquisitionConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    llm: LLMConfig = LLMConfig()
    report: ReportConfig = ReportConfig()

def load_config(config_path: Optional[str] = None) -> LEAConfig:
    paths_to_try = []
    if config_path:
        paths_to_try.append(Path(config_path))
    paths_to_try.extend([
        Path("configs/default_config.yaml"),
        Path(__file__).parents[2] / "configs" / "default_config.yaml"
    ])

    data = {}
    for p in paths_to_try:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                content = f.read()
                # Env expansion for ${VAR:-default} or ${VAR}
                expanded = _expand_env_vars(content)
                data = yaml.safe_load(expanded) or {}
                break

    return LEAConfig(**data)
