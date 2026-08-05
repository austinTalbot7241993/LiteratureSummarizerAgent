# Antigravity Task Specification: Literature Exploration Agent (`LEA`)

**Target Agent Runner:** Google Antigravity CLI / Agent Workspace  
**Project Identifier:** `literature-exploration-agent`  
**Target Hardware:** 1x NVIDIA RTX 2080 Ti, 11 GB VRAM, Turing SM 7.5  
**Primary Generator:** `Qwen/Qwen2.5-7B-Instruct`  
**Default Inference Mode:** 4-bit NF4 base model with FP16 compute and an optional PEFT LoRA adapter  
**Dense Embedding Model:** `BAAI/bge-m3`, 1024-dimensional embeddings  
**Supported Python Version:** Python 3.11

---

## 1. Objective

Autonomously scaffold, implement, test, and document a Python package and Typer CLI that:

1. Ingests an academic PDF, computes its SHA-256 hash, and deduplicates repeated inputs.
2. Extracts paper metadata, full text, bibliography entries, and in-text citation markers using GROBID as a primary engine, with a resilient multi-stage fallback (PyMuPDF + RegEx DOI/ArXiv extraction + OpenAlex Reference API).
3. Resolves the input paper and its references against OpenAlex and Semantic Scholar.
4. Discovers related literature while strictly excluding:
   - the input paper itself; and
   - every work identified in the input paper's bibliography.
5. Downloads only lawfully accessible open-access PDFs, without bypassing authentication or paywalls.
6. Stores paper metadata, provenance, text chunks, embeddings, discovery runs, and summaries in PostgreSQL 16 with `pgvector`.
7. Performs hybrid retrieval using:
   - dense cosine similarity over BGE-M3 embeddings; and
   - sparse BM25 ranking over the same child chunks (scoped per candidate paper/run);
   - fused with Reciprocal Rank Fusion (RRF).
8. Generates validated technical summaries with Qwen 2.5 7B.
9. Exports a single-file HTML report containing:
    - ranked related papers;
    - single-paragraph technical summaries;
    - evidence scope and retrieval provenance;
    - verified bibliographic metadata and BibTeX;
    - open-access PDF links when available.
10. Provides a separate, optional QLoRA training workflow. Fine-tuning must not be required to run the core literature-exploration pipeline.

The exclusion invariant is:

$$
\mathcal{S}_{\text{discovered}} \cap
\left(\mathcal{C}_{\text{cited}} \cup \{p_{\text{input}}\}\right)
= \emptyset.
$$

The default pipeline must fail closed only when this invariant cannot be established after all fallback extraction stages fail.

---

## 2. Non-Goals and Safety Constraints

The implementation must not:

- bypass publisher paywalls, institutional authentication, robots restrictions, or access controls;
- treat a metadata page as evidence that a PDF is downloadable;
- generate BibTeX from the LLM;
- claim that citation exclusion is complete when bibliography extraction is incomplete;
- require a GPU for metadata ingestion, discovery, database operations, report generation, or unit tests;
- automatically fine-tune the model during `lea run`;
- keep the embedding model and generator resident on the 11 GB GPU at the same time in the same process context.

---

## 3. Required Repository Structure

```text
literature-exploration-agent/
├── .env.example
├── .gitignore
├── README.md
├── pyproject.toml
├── uv.lock
├── docker-compose.yml
├── alembic.ini
├── configs/
│   ├── default_config.yaml
│   └── qlora_2080ti.yaml
├── migrations/
│   ├── env.py
│   └── versions/
│       └── 0001_initial_schema.py
├── schema.sql
├── scripts/
│   └── verify.sh
├── src/
│   └── lea/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py
│       ├── config.py
│       ├── logging.py
│       ├── exceptions.py
│       ├── db/
│       │   ├── models.py
│       │   ├── repository.py
│       │   └── session.py
│       ├── ingester/
│       │   ├── grobid_client.py
│       │   ├── pdf_parser.py
│       │   ├── tei_parser.py
│       │   └── reference_parser.py
│       ├── resolution/
│       │   ├── identifiers.py
│       │   ├── matcher.py
│       │   └── metadata_merge.py
│       ├── discovery/
│       │   ├── openalex.py
│       │   ├── semantic_scholar.py
│       │   ├── candidate_builder.py
│       │   └── exclusion.py
│       ├── acquisition/
│       │   ├── downloader.py
│       │   ├── open_access.py
│       │   └── validation.py
│       ├── rag/
│       │   ├── chunker.py
│       │   ├── embedder.py
│       │   ├── bm25_index.py
│       │   ├── dense_search.py
│       │   ├── hybrid_search.py
│       │   └── prompts.py
│       ├── llm/
│       │   ├── schemas.py
│       │   ├── inference.py
│       │   ├── backends.py
│       │   └── qlora_trainer.py
│       ├── bibliography/
│       │   ├── verifier.py
│       │   └── bibtex.py
│       └── exporter/
│           ├── html_builder.py
│           └── templates/
│               └── report.html.j2
└── tests/
    ├── fixtures/
    │   ├── sample_paper.pdf
    │   ├── sample_grobid.tei.xml
    │   └── api_responses/
    ├── unit/
    │   ├── test_identifier_normalization.py
    │   ├── test_reference_matching.py
    │   ├── test_citation_exclusion.py
    │   ├── test_chunker.py
    │   ├── test_rrf.py
    │   ├── test_summary_validation.py
    │   ├── test_bibtex_generation.py
    │   └── test_html_exporter.py
    ├── integration/
    │   ├── test_database_repository.py
    │   ├── test_pgvector_search.py
    │   ├── test_bm25_index.py
    │   ├── test_grobid_client.py
    │   └── test_cli_pipeline.py
    └── gpu/
        ├── test_embedding_smoke.py
        ├── test_inference_smoke.py
        └── test_qlora_smoke.py
```

---

## 4. Required CLI Surface

Use Typer. The package must install a `lea` executable.

```text
lea doctor
lea db init
lea ingest PAPER.pdf
lea discover PAPER_ID
lea acquire RUN_ID
lea index RUN_ID
lea summarize RUN_ID
lea report RUN_ID --output report.html
lea run PAPER.pdf --output report.html
lea train DATASET.jsonl --output-dir adapters/lea-qwen
```

---

## 5. Configuration Contract

The default configuration must be stored in `configs/default_config.yaml`.

```yaml
application:
  log_level: INFO
  cache_dir: .cache/lea
  artifact_dir: artifacts
  request_timeout_seconds: 30
  max_retries: 4

services:
  database_url: ${LEA_DATABASE_URL}
  grobid_url: ${LEA_GROBID_URL:-http://localhost:8070}
  openalex_api_key: ${OPENALEX_API_KEY}
  semantic_scholar_api_key: ${SEMANTIC_SCHOLAR_API_KEY:-}
  unpaywall_email: ${UNPAYWALL_EMAIL:-}
  # Rate limiting parameters to prevent HTTP 429 bans
  semantic_scholar_rate_limit_rps: 0.2
  openalex_rate_limit_rps: 5.0

extraction:
  require_complete_bibliography: true
  allow_pymupdf_fallback: true
  allow_openalex_reference_fallback: true
  allow_incomplete_citation_exclusion: false

discovery:
  openalex_limit: 100
  semantic_scholar_limit: 100
  final_candidate_limit: 20
  source_rrf_k: 60
  title_similarity_threshold: 0.96
  year_tolerance: 1

acquisition:
  download_open_access_pdfs: true
  max_pdf_bytes: 104857600
  allowed_content_types:
    - application/pdf
  user_agent: "LEA/0.1 scholarly-research-agent"

chunking:
  tokenizer_model: Qwen/Qwen2.5-7B-Instruct
  parent_tokens: 768
  parent_overlap_tokens: 96
  child_tokens: 256
  child_overlap_tokens: 48

embedding:
  model: BAAI/bge-m3
  dimension: 1024
  device: auto
  batch_size_gpu: 4
  batch_size_cpu: 1
  normalize_embeddings: true
  max_length: 1024

retrieval:
  dense_top_k: 30
  sparse_top_k: 30
  fused_top_k: 8
  rrf_k: 60
  sparse_backend: bm25s
  scope_sparse_index_per_run: true

llm:
  model: Qwen/Qwen2.5-7B-Instruct
  adapter_path: null
  backend: transformers_peft
  load_in_4bit: true
  quant_type: nf4
  use_double_quant: true
  compute_dtype: float16
  device: cuda
  inference_max_model_length: 2048 # Adjusted for 11GB VRAM headroom
  max_context_tokens: 1800         # Caps context window to prevent CUDA OOM
  max_new_tokens: 600
  temperature: 0.1
  top_p: 0.9
  generation_attempts: 2

report:
  title: Literature Exploration Report
  include_abstract_only_results: true
  include_retrieval_provenance: true
  include_bibtex: true
```

---

## 6. Hardware and Runtime Constraints (RTX 2080 Ti - 11 GB VRAM)

### 6.1 GPU Compute & Precision
The RTX 2080 Ti is a Turing GPU and must use FP16 compute (`torch.float16`). The implementation must **never** request `bfloat16`.

### 6.2 Process-Isolated Sequential GPU Pipeline
To prevent PyTorch CUDA memory fragmentation and VRAM allocation overlap:
1. **Embedding Phase:** Execute `bge-m3` embedding in a dedicated Python worker sub-process (using `multiprocessing` with a `spawn` context). Terminate the sub-process upon completion to ensure 100% CUDA memory release back to the OS driver.
2. **Inference Phase:** Spawn a fresh Python process to load Qwen 2.5 7B in 4-bit NF4 (`BitsAndBytes`) for batch summarization.
3. Max model context is capped at `2048` tokens with `max_context_tokens: 1800` to prevent KV-cache OOM spikes during inference.

---

## 7. Citation Resolution and Resilient Exclusion

### 7.1 Multi-Stage Reference Extraction Fallback
When parsing an input PDF's works cited ($\mathcal{C}_{	ext{cited}}$):
1. **Primary:** Submit PDF to local GROBID microservice for TEI-XML reference parsing.
2. **Secondary (Fallback):** If GROBID times out or fails, parse text via PyMuPDF and extract inline DOIs/arXiv IDs via RegEx patterns.
3. **Tertiary (API Fallback):** If input paper DOI/arXiv ID is known, fetch verified references directly from the OpenAlex `/works/{id}` reference endpoint.

Only if all three stages fail to produce a verifiable bibliography does the system flag `citation_exclusion_status='incomplete'` and fail closed.

---

## 8. Structured Summarization Contract

### 8.1 Schema & Word Budget
```python
from pydantic import BaseModel, Field, field_validator


class TechnicalSummary(BaseModel):
    problem_formulation: str = Field(
        min_length=1,
        description="Mathematical, statistical, or scientific problem statement",
    )
    methodological_novelty: str = Field(
        min_length=1,
        description="Core algorithmic, empirical, or theoretical novelty",
    )
    empirical_findings: str = Field(
        min_length=1,
        description="Quantifiable validation, or an explicit statement that it was not reported",
    )
    paragraph_summary: str = Field(
        min_length=1,
        description="Single-paragraph technical synthesis of at most 300 words",
    )

    @field_validator("paragraph_summary")
    @classmethod
    def validate_paragraph_summary(cls, value: str) -> str:
        words = value.strip().split()
        if len(words) > 300:
            raise ValueError(f"paragraph_summary must contain at most 300 words (got {len(words)})")
        if "

" in value:
            raise ValueError("paragraph_summary must be a single paragraph")
        return value.strip()
```

---

## 9. Antigravity Execution Command

```bash
antigravity run   --spec TASK_SPEC_FINAL.md   --autonomy-level full   --verify-commands "./scripts/verify.sh"
```
