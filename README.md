# Literature Exploration Agent (`LEA`)

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16%20%2B%20pgvector-blue)](https://github.com/pgvector/pgvector)
[![LLM](https://img.shields.io/badge/LLM-Qwen%202.5%207B%20%284--bit%20NF4%29-purple)](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)

**Literature Exploration Agent (LEA)** is an autonomous academic paper analysis system designed to ingest scientific PDFs, harvest related work across open academic graphs, enforce strict **citation exclusion**, perform hybrid RAG retrieval, and generate validated technical summaries using **Qwen 2.5 7B**.

---

## 📌 Overview

When conducting literature reviews, existing citation graph traversal tools often surface papers already listed in a target paper's bibliography or return the target paper itself. **LEA** solves this by enforcing a mathematical **Citation Exclusion Invariant**:

$$\mathcal{S}_{\text{discovered}} \cap \left(\mathcal{C}_{\text{cited}} \cup \{p_{\text{input}}\}\right) = \emptyset$$

Where $\mathcal{S}_{\text{discovered}}$ is the set of discovered candidate papers, $\mathcal{C}_{\text{cited}}$ is the set of works cited in the input paper's bibliography, and $p_{\text{input}}$ is the input paper itself.

### Key Capabilities

- **Resilient Multi-Stage Reference Extraction**: Employs a 3-stage fallback pipeline (Local GROBID microservice $\rightarrow$ PyMuPDF + RegEx pattern extraction $\rightarrow$ OpenAlex `/works` Reference API).
- **Strict Citation Exclusion**: Filters out all cited papers and identical preprints/re-publications before candidate ranking.
- **Dual API Harvesting**: Pulls candidate papers from OpenAlex and Semantic Scholar with configurable rate limiting.
- **Lawful Open-Access Acquisition**: Downloads open-access PDFs via Unpaywall and direct open-access endpoints without bypassing paywalls or access controls.
- **Hybrid RAG Retrieval Engine**: Combines **BGE-M3** dense vector embeddings (stored in PostgreSQL via `pgvector`) with **BM25s** sparse keyword search, fused via Reciprocal Rank Fusion (RRF).
- **Quantized Technical Summarization**: Generates structured, single-paragraph technical summaries using **Qwen 2.5 7B** (4-bit NF4 with FP16 compute), validated via Pydantic word-count and formatting contracts.
- **Single-File HTML Exporter**: Produces responsive single-file HTML reports complete with metadata, ranked candidates, retrieval provenance, and verified BibTeX entries.
- **Hardware-Optimized VRAM Management**: Designed for 11 GB VRAM GPUs (such as NVIDIA RTX 2080 Ti) using process-isolated sub-processes for embedding and inference phases.

---

## 🏗️ Architecture Pipeline

```text
               ┌────────────────────────┐
               │    Input Paper PDF     │
               └───────────┬────────────┘
                           │
             ┌─────────────▼────────────┐
             │   1. PDF Ingestion &     │
             │   Reference Parsing      │ (GROBID -> PyMuPDF -> OpenAlex)
             └─────────────┬────────────┘
                           │
             ┌─────────────▼────────────┐
             │  2. Candidate Discovery  │
             │  & Citation Exclusion    │ (OpenAlex + Semantic Scholar)
             └─────────────┬────────────┘
                           │
             ┌─────────────▼────────────┐
             │  3. Open-Access PDF      │
             │     Acquisition          │ (Unpaywall / OA Direct)
             └─────────────┬────────────┘
                           │
             ┌─────────────▼────────────┐
             │  4. Hierarchical Chunking│
             │     & Indexing           │ (BGE-M3 Dense + BM25s Sparse)
             └─────────────┬────────────┘
                           │
             ┌─────────────▼────────────┐
             │  5. Hybrid RAG &         │
             │  LLM Summarization       │ (Qwen 2.5 7B 4-bit NF4)
             └─────────────┬────────────┘
                           │
             ┌─────────────▼────────────┐
             │  6. Single-File HTML     │
             │     Report Generation    │
             └──────────────────────────┘
```

---

## 🛠️ Prerequisites & Installation

### Requirements

- **OS**: Linux / macOS
- **Python**: Python 3.11+
- **Docker & Docker Compose**: For running PostgreSQL (`pgvector`) and GROBID microservices.
- **Hardware (Optional for GPU Acceleration)**: NVIDIA GPU with $\ge 11\text{ GB}$ VRAM (e.g., RTX 2080 Ti). Running on CPU mode is fully supported.

### Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/austinTalbot7241993/LiteratureSummarizerAgent.git
   cd LiteratureSummarizerAgent
   ```

2. **Create Virtual Environment & Install Package**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e .
   # Or install with dev dependencies:
   pip install -e ".[dev]"
   ```

3. **Launch Docker Services**:
   LEA relies on PostgreSQL with `pgvector` and the GROBID TEI-XML parsing service.
   ```bash
   docker compose up -d
   ```
   *Verify container status with `docker compose ps`.*

4. **Environment Variables**:
   Copy `.env.example` to `.env` and fill in optional API keys:
   ```bash
   cp .env.example .env
   ```
   Key environment variables:
   ```ini
   LEA_DATABASE_URL=postgresql://lea_user:lea_pass@localhost:5432/lea_db
   LEA_GROBID_URL=http://localhost:8070
   OPENALEX_API_KEY=
   SEMANTIC_SCHOLAR_API_KEY=
   UNPAYWALL_EMAIL=researcher@example.com
   ```

5. **Initialize Database Schema**:
   ```bash
   lea db init
   ```

6. **Run Environment Diagnostics**:
   ```bash
   lea doctor
   ```

---

## 💻 CLI Reference

The package exposes a Typer CLI entrypoint `lea`.

```text
Usage: lea [OPTIONS] COMMAND [ARGS]...

  Literature Exploration Agent (LEA) CLI

Commands:
  doctor     Checks database connectivity, GROBID microservice, and GPU environment.
  db         Database administration subcommands (init).
  run        Executes full end-to-end literature exploration pipeline.
  ingest     Ingests PDF paper, extracts metadata/references, and stores entry in DB.
  discover   Discovers related literature excluding input paper and works in bibliography.
  acquire    Acquires open-access PDFs for candidate papers in a discovery run.
  index      Indexes candidate text chunks and computes dense embeddings.
  summarize  Generates structured technical summaries using hybrid retrieval and Qwen 2.5 7B.
  report     Exports single-file HTML report for a completed discovery run.
  train      Executes optional QLoRA training workflow for Qwen 2.5 7B.
```

### 🚀 One-Command Execution

To run the complete end-to-end pipeline (Ingest $\rightarrow$ Discover $\rightarrow$ Acquire $\rightarrow$ Index $\rightarrow$ Summarize $\rightarrow$ Report):

```bash
lea run path/to/paper.pdf --output report.html
```

### ⚙️ Step-by-Step Modular Workflow

You can also run pipeline stages individually:

```bash
# 1. Ingest input paper PDF
lea ingest path/to/paper.pdf

# 2. Discover related candidate papers (takes Paper UUID from ingest step)
lea discover <PAPER_UUID>

# 3. Acquire candidate open-access PDFs (takes Discovery Run UUID)
lea acquire <RUN_UUID>

# 4. Chunk & compute embeddings for candidate papers
lea index <RUN_UUID>

# 5. Run RAG & Qwen 2.5 7B summarization
lea summarize <RUN_UUID>

# 6. Build single-file HTML report
lea report <RUN_UUID> --output output_report.html
```

### 🏋️ Optional Fine-Tuning

Fine-tune Qwen 2.5 7B with QLoRA on a domain-specific `.jsonl` dataset:

```bash
lea train dataset.jsonl --output-dir adapters/lea-qwen
```

---

## ⚙️ Configuration Contract

The system configuration is loaded from [`configs/default_config.yaml`](file:///home/austin/Forks/LiteratureSummarizerAgent/configs/default_config.yaml) and can be overridden via environment variables or custom config files.

Key configuration parameters include:

| Section | Parameter | Default | Description |
| :--- | :--- | :--- | :--- |
| `services` | `database_url` | `${LEA_DATABASE_URL}` | PostgreSQL database URL with `pgvector` |
| `services` | `grobid_url` | `http://localhost:8070` | Local GROBID microservice endpoint |
| `extraction` | `require_complete_bibliography` | `true` | Fails closed if bibliography extraction fails |
| `discovery` | `final_candidate_limit` | `20` | Maximum candidate papers returned |
| `embedding` | `model` | `BAAI/bge-m3` | 1024-dim dense embedding model |
| `retrieval` | `sparse_backend` | `bm25s` | Sparse keyword search engine |
| `llm` | `model` | `Qwen/Qwen2.5-7B-Instruct` | Primary LLM generator |
| `llm` | `load_in_4bit` | `true` | 4-bit NF4 quantization flag |
| `llm` | `compute_dtype` | `float16` | Compute precision (Turing SM 7.5 optimized) |

---

## ⚡ Hardware & Memory Optimization

To operate seamlessly within an **11 GB VRAM budget** (e.g., NVIDIA RTX 2080 Ti):

1. **Process-Isolated Pipeline**:
   - Dense embedding (`BAAI/bge-m3`) runs inside a dedicated Python sub-process that terminates completely before inference begins.
   - LLM Inference (`Qwen/Qwen2.5-7B-Instruct`) runs in a fresh process context, ensuring zero CUDA VRAM overlap or fragmentation.
2. **Precision & Quantization**:
   - Models use 4-bit NF4 quantization (`bitsandbytes`) with FP16 (`torch.float16`) compute precision (`bfloat16` is intentionally disabled for Turing compatibility).
   - Maximum context length is capped at 1800 tokens (`max_context_tokens: 1800`) to prevent CUDA OOM spikes during KV-cache allocation.

---

## 🧪 Testing & Verification

Run the verification suite to execute unit tests, integration tests, and environment checks:

```bash
./scripts/verify.sh
```

Or run pytest directly:

```bash
# Unit tests
pytest tests/unit -v

# Integration tests (requires Docker services up)
pytest tests/integration -v
```

---

## 📄 License

Distributed under the [MIT License](file:///home/austin/Forks/LiteratureSummarizerAgent/LICENSE).