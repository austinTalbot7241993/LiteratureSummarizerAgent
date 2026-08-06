import asyncio
import os
import sys
import uuid
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table

from lea.config import load_config
from lea.logging import setup_logging, logger
from lea.db.session import init_db, create_tables, get_db_session
from lea.db.repository import LEARepository

app = typer.Typer(
    name="lea",
    help="Literature Exploration Agent (LEA) CLI",
    add_completion=False
)
console = Console()
db_app = typer.Typer(help="Database administration subcommands")
app.add_typer(db_app, name="db")


@app.command()
def doctor():
    """Checks database connectivity, GROBID microservice, and GPU environment."""
    console.print("[bold blue]Running LEA Environment Diagnostic Check...[/bold blue]\n")
    config = load_config()

    table = Table(title="System Environment Check")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Details")

    # 1. Database Check
    try:
        engine = init_db(config.services.database_url)
        with engine.connect() as conn:
            table.add_row("Database Connection", "[green]OK[/green]", f"Connected to {config.services.database_url.split('@')[-1]}")
    except Exception as exc:
        table.add_row("Database Connection", "[red]FAIL[/red]", str(exc))

    # 2. GROBID Check
    try:
        from lea.ingester.grobid_client import GrobidClient
        client = GrobidClient(base_url=config.services.grobid_url)
        alive = asyncio.run(client.is_alive())
        if alive:
            table.add_row("GROBID Service", "[green]OK[/green]", f"Reachable at {config.services.grobid_url}")
        else:
            table.add_row("GROBID Service", "[yellow]UNAVAILABLE[/yellow]", f"Not responding at {config.services.grobid_url}")
    except Exception as exc:
        table.add_row("GROBID Service", "[yellow]UNAVAILABLE[/yellow]", str(exc))

    # 3. PyTorch / GPU Check
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            table.add_row("GPU Device", "[green]OK[/green]", f"Detected {device_name} (CUDA available)")
        else:
            table.add_row("GPU Device", "[yellow]CPU MODE[/yellow]", "No CUDA GPU detected; falling back to CPU")
    except ImportError:
        table.add_row("PyTorch", "[yellow]NOT INSTALLED[/yellow]", "PyTorch not found; running in lightweight CPU mode")

    console.print(table)


@db_app.command("init")
def db_init():
    """Initializes database schema and extensions."""
    config = load_config()
    console.print(f"Initializing database at [cyan]{config.services.database_url}[/cyan]...")
    try:
        engine = init_db(config.services.database_url)
        create_tables(engine)
        console.print("[bold green]Database schema successfully initialized.[/bold green]")
    except Exception as exc:
        console.print(f"[bold red]Database initialization failed:[/bold red] {exc}")
        raise typer.Exit(1)


@app.command()
def ingest(pdf_path: str = typer.Argument(..., help="Path to input paper PDF")):
    """Ingests PDF paper, extracts metadata/references, and stores entry in DB."""
    config = load_config()
    setup_logging(config.application.log_level)
    init_db(config.services.database_url)

    from lea.ingester.pdf_parser import PDFParser
    from lea.ingester.grobid_client import GrobidClient
    from lea.ingester.tei_parser import TEIParser
    from lea.ingester.reference_parser import ReferenceParser
    from lea.resolution.identifiers import normalize_doi, normalize_arxiv
    from lea.discovery.openalex import OpenAlexClient

    pdf = PDFParser(pdf_path)
    sha256 = pdf.compute_sha256()

    with get_db_session() as session:
        repo = LEARepository(session)
        existing = repo.get_paper_by_hash(sha256)
        if existing:
            console.print(f"[bold yellow]Paper already ingested (ID: {existing.id})[/bold yellow]")
            return

        # Fallback metadata initially
        fallback_meta = pdf.parse_fallback_metadata()
        title = fallback_meta["title"]
        authors = fallback_meta["authors"]
        doi = fallback_meta["doi"]
        arxiv_id = fallback_meta["arxiv_id"]

        # Attempt GROBID fulltext parse
        grobid_client = GrobidClient(base_url=config.services.grobid_url)
        if asyncio.run(grobid_client.is_alive()):
            try:
                tei_xml = asyncio.run(grobid_client.process_fulltext(pdf_path))
                tei_header = TEIParser(tei_xml).parse_header()
                if tei_header.get("title"):
                    title = tei_header["title"]
                if tei_header.get("authors"):
                    authors = tei_header["authors"]
                if tei_header.get("doi"):
                    doi = tei_header["doi"]
                if tei_header.get("arxiv_id"):
                    arxiv_id = tei_header["arxiv_id"]
            except Exception as exc:
                logger.warning(f"GROBID header extraction fallback: {exc}")

        paper = repo.create_paper(
            sha256_hash=sha256,
            title=title,
            authors=authors,
            doi=normalize_doi(doi),
            arxiv_id=normalize_arxiv(arxiv_id),
            publication_year=fallback_meta.get("publication_year"),
            pdf_path=pdf_path
        )

        # Extract bibliography references
        oa_client = OpenAlexClient()
        ref_parser = ReferenceParser(grobid_url=config.services.grobid_url)
        references, status = asyncio.run(ref_parser.extract_references(
            pdf_path=pdf_path,
            doi=doi,
            arxiv_id=arxiv_id,
            openalex_client=oa_client
        ))

        for ref in references:
            repo.add_reference(source_paper_id=paper.id, **ref)

        console.print(f"[bold green]Successfully ingested paper '{title}'[/bold green]")
        console.print(f"Paper ID: [cyan]{paper.id}[/cyan] (References extracted: {len(references)}, status: {status})")


@app.command()
def discover(paper_id: str = typer.Argument(..., help="Paper UUID to discover related literature for")):
    """Discovers related literature excluding input paper and works in bibliography."""
    config = load_config()
    setup_logging(config.application.log_level)
    init_db(config.services.database_url)

    p_uuid = uuid.UUID(paper_id)
    with get_db_session() as session:
        repo = LEARepository(session)
        paper = repo.get_paper_by_id(p_uuid)
        if not paper:
            console.print(f"[bold red]Paper {paper_id} not found in database.[/bold red]")
            raise typer.Exit(1)

        references = repo.get_references_for_paper(p_uuid)
        ref_dicts = [
            {
                "title": r.title,
                "authors": r.authors or [],
                "doi": r.doi,
                "arxiv_id": r.arxiv_id,
                "openalex_id": r.openalex_id,
                "s2_id": r.s2_id,
                "year": r.year
            }
            for r in references
        ]

        paper_meta = {
            "title": paper.title,
            "authors": paper.authors or [],
            "doi": paper.doi,
            "arxiv_id": paper.arxiv_id,
            "openalex_id": paper.openalex_id,
            "s2_id": paper.s2_id,
            "publication_year": paper.publication_year
        }

        # Create discovery run
        exclusion_status = "complete" if references or not config.extraction.require_complete_bibliography else "incomplete"
        run = repo.create_discovery_run(input_paper_id=paper.id, exclusion_status=exclusion_status)

        from lea.discovery.candidate_builder import CandidateBuilder
        builder = CandidateBuilder(config=config)

        candidates = asyncio.run(builder.build_candidates(
            input_paper_meta=paper_meta,
            cited_references=ref_dicts,
            exclusion_status=exclusion_status,
            final_candidate_limit=config.discovery.final_candidate_limit,
            source_rrf_k=config.discovery.source_rrf_k
        ))

        for cand_dict in candidates:
            # Upsert candidate paper into DB
            c_sha = str(uuid.uuid4())[:32]
            c_paper = repo.create_paper(
                sha256_hash=c_sha,
                title=cand_dict.get("title", "Untitled"),
                authors=cand_dict.get("authors", []),
                doi=cand_dict.get("doi"),
                arxiv_id=cand_dict.get("arxiv_id"),
                openalex_id=cand_dict.get("openalex_id"),
                s2_id=cand_dict.get("s2_id"),
                publication_year=cand_dict.get("publication_year"),
                venue=cand_dict.get("venue"),
                abstract=cand_dict.get("abstract"),
                is_open_access=cand_dict.get("is_open_access", False),
                oa_pdf_url=cand_dict.get("oa_pdf_url")
            )

            repo.add_candidate_paper(
                run_id=run.id,
                paper_id=c_paper.id,
                score=cand_dict.get("rrf_score", 0.0),
                rrf_rank=cand_dict.get("rrf_rank"),
                source_apis=cand_dict.get("source_apis", []),
                open_access_url=cand_dict.get("oa_pdf_url")
            )

        repo.update_discovery_run(run.id, run_status="discovered")
        console.print(f"[bold green]Discovery run completed (RUN ID: {run.id})[/bold green]")
        console.print(f"Discovered and stored [cyan]{len(candidates)}[/cyan] candidate papers after strict citation exclusion.")


@app.command()
def acquire(run_id: str = typer.Argument(..., help="Discovery run UUID")):
    """Acquires open-access PDFs for candidate papers in a discovery run."""
    config = load_config()
    setup_logging(config.application.log_level)
    init_db(config.services.database_url)

    r_uuid = uuid.UUID(run_id)
    with get_db_session() as session:
        repo = LEARepository(session)
        run = repo.get_discovery_run(r_uuid)
        if not run:
            console.print(f"[bold red]Discovery run {run_id} not found.[/bold red]")
            raise typer.Exit(1)

        candidates = repo.get_candidates_for_run(r_uuid)
        from lea.acquisition.downloader import PDFDownloader
        downloader = PDFDownloader(
            cache_dir=config.application.cache_dir + "/pdfs",
            max_pdf_bytes=config.acquisition.max_pdf_bytes,
            user_agent=config.acquisition.user_agent,
            unpaywall_email=config.services.unpaywall_email or None
        )

        downloaded_count = 0
        for cand in candidates:
            p = cand.paper
            cand_meta = {
                "title": p.title,
                "doi": p.doi,
                "arxiv_id": p.arxiv_id,
                "s2_id": p.s2_id,
                "oa_pdf_url": cand.open_access_url or p.oa_pdf_url
            }
            pdf_path = asyncio.run(downloader.download_candidate_pdf(cand_meta))
            if pdf_path:
                cand.pdf_path = pdf_path
                cand.is_downloaded = True
                downloaded_count += 1

        repo.update_discovery_run(r_uuid, run_status="acquired")
        console.print(f"[bold green]Acquisition completed for RUN ID: {run_id}[/bold green]")
        console.print(f"Downloaded [cyan]{downloaded_count}[/cyan] / {len(candidates)} candidate PDFs.")


@app.command()
def index(run_id: str = typer.Argument(..., help="Discovery run UUID")):
    """Indexes candidate text chunks and computes dense embeddings."""
    config = load_config()
    setup_logging(config.application.log_level)
    init_db(config.services.database_url)

    r_uuid = uuid.UUID(run_id)
    with get_db_session() as session:
        repo = LEARepository(session)
        run = repo.get_discovery_run(r_uuid)
        if not run:
            console.print(f"[bold red]Discovery run {run_id} not found.[/bold red]")
            raise typer.Exit(1)

        candidates = repo.get_candidates_for_run(r_uuid)
        from lea.rag.chunker import HierarchicalChunker
        from lea.rag.embedder import BGEEmbedder
        from lea.ingester.pdf_parser import PDFParser

        chunker = HierarchicalChunker(
            tokenizer_model=config.chunking.tokenizer_model,
            parent_tokens=config.chunking.parent_tokens,
            parent_overlap_tokens=config.chunking.parent_overlap_tokens,
            child_tokens=config.chunking.child_tokens,
            child_overlap_tokens=config.chunking.child_overlap_tokens
        )

        embedder = BGEEmbedder(
            model_name=config.embedding.model,
            max_length=config.embedding.max_length,
            use_subprocess=True if config.embedding.device == "cuda" else False
        )

        total_chunks = 0
        for cand in candidates:
            p = cand.paper
            if not (cand.pdf_path and os.path.exists(cand.pdf_path)):
                if config.acquisition.require_downloaded_pdf:
                    logger.info(f"Skipping indexing for '{p.title}' - no downloaded PDF available (require_downloaded_pdf=True)")
                    continue
                text_content = p.abstract or p.title
            else:
                try:
                    text_content = PDFParser(cand.pdf_path).extract_body_text()
                except Exception as exc:
                    logger.warning(f"Could not extract PDF text for {p.title}: {exc}")
                    if config.acquisition.require_downloaded_pdf:
                        continue
                    text_content = p.abstract or p.title

            chunk_objs = chunker.chunk_text(text_content)
            child_contents = [c["content"] for c in chunk_objs if c["chunk_type"] == "child"]
            embeddings = embedder.embed_texts(child_contents)

            # Map parent indices
            parent_id_map = {}
            child_emb_idx = 0

            for c in chunk_objs:
                p_id = parent_id_map.get(c.get("parent_index")) if c["chunk_type"] == "child" else None
                emb = None
                if c["chunk_type"] == "child" and child_emb_idx < len(embeddings):
                    emb = embeddings[child_emb_idx]
                    child_emb_idx += 1

                chunk_rec = repo.add_chunk(
                    paper_id=p.id,
                    run_id=r_uuid,
                    chunk_type=c["chunk_type"],
                    content=c["content"],
                    chunk_index=c["chunk_index"],
                    token_count=c["token_count"],
                    parent_id=p_id,
                    embedding=emb
                )

                if c["chunk_type"] == "parent":
                    parent_id_map[c["chunk_index"]] = chunk_rec.id
                total_chunks += 1

        repo.update_discovery_run(r_uuid, run_status="indexed")
        console.print(f"[bold green]Indexing completed for RUN ID: {run_id}[/bold green]")
        console.print(f"Created and stored [cyan]{total_chunks}[/cyan] parent/child text chunks with embeddings.")


@app.command()
def summarize(
    run_id: str = typer.Argument(..., help="Discovery run UUID"),
    allow_mock: bool = typer.Option(False, "--mock", help="Allow mock LLM backend fallback for testing without GPU")
):
    """Generates structured technical summaries using hybrid retrieval and Qwen 2.5 7B."""
    config = load_config()
    setup_logging(config.application.log_level)
    init_db(config.services.database_url)

    r_uuid = uuid.UUID(run_id)
    with get_db_session() as session:
        repo = LEARepository(session)
        run = repo.get_discovery_run(r_uuid)
        if not run:
            console.print(f"[bold red]Discovery run {run_id} not found.[/bold red]")
            raise typer.Exit(1)

        candidates = repo.get_candidates_for_run(r_uuid)
        from lea.rag.embedder import BGEEmbedder
        from lea.rag.dense_search import DenseSearchEngine
        from lea.rag.hybrid_search import HybridSearchEngine
        from lea.llm.inference import TechnicalSummarizer
        from lea.llm.backends import MockLLMBackend, TransformersPeftBackend

        embedder = BGEEmbedder(model_name=config.embedding.model)
        dense_engine = DenseSearchEngine(embedder)
        hybrid_engine = HybridSearchEngine(dense_engine, rrf_k=config.retrieval.rrf_k)

        # Select backend
        import torch
        if torch.cuda.is_available() and config.llm.device == "cuda":
            backend = TransformersPeftBackend(
                model_name=config.llm.model,
                adapter_path=config.llm.adapter_path,
                max_context_tokens=config.llm.max_context_tokens
            )
        elif allow_mock:
            console.print("[bold yellow]WARNING: Using MockLLMBackend because --mock was explicitly specified.[/bold yellow]")
            backend = MockLLMBackend()
        else:
            console.print("[bold red]ERROR: CUDA GPU acceleration is required for LLM summarization, but PyTorch CUDA is not available![/bold red]")
            console.print("[yellow]To run real LLM inference with Qwen 2.5 7B on your RTX 2080 Ti, install PyTorch compiled for CUDA 12.1 (+cu121).[/yellow]")
            console.print("[yellow]If you explicitly want mock summaries for testing without GPU, re-run with --mock.[/yellow]")
            raise typer.Exit(code=1)

        summarizer = TechnicalSummarizer(backend=backend, max_attempts=config.llm.generation_attempts)

        for cand in candidates:
            p = cand.paper
            if not (cand.pdf_path and os.path.exists(cand.pdf_path)):
                if config.acquisition.require_downloaded_pdf:
                    logger.info(f"Skipping summarization for '{p.title}' - no downloaded PDF available")
                    continue
            # Fetch targeted chunks for both methodology and empirical findings
            retrieved_methodology = hybrid_engine.hybrid_search(
                repo=repo,
                run_id=r_uuid,
                paper_id=p.id,
                query_text=f"Algorithm framework model formulation methodology approach of {p.title}",
                dense_top_k=config.retrieval.dense_top_k,
                sparse_top_k=config.retrieval.sparse_top_k,
                fused_top_k=config.retrieval.fused_top_k // 2
            )
            retrieved_empirical = hybrid_engine.hybrid_search(
                repo=repo,
                run_id=r_uuid,
                paper_id=p.id,
                query_text=f"Empirical results benchmark evaluation findings accuracy performance of {p.title}",
                dense_top_k=config.retrieval.dense_top_k,
                sparse_top_k=config.retrieval.sparse_top_k,
                fused_top_k=config.retrieval.fused_top_k // 2
            )

            seen_cids = set()
            retrieved_chunks = []
            for item in retrieved_methodology + retrieved_empirical:
                chunk_obj = item[0]
                cid = chunk_obj.get("id") or chunk_obj.get("chunk_index")
                if cid not in seen_cids:
                    seen_cids.add(cid)
                    retrieved_chunks.append(chunk_obj)

            cand_meta = {"title": p.title, "authors": p.authors, "publication_year": p.publication_year}

            tech_summary = summarizer.summarize_candidate(cand_meta, retrieved_chunks)
            repo.add_summary(
                run_id=r_uuid,
                candidate_paper_id=cand.id,
                problem_formulation=tech_summary.problem_formulation,
                methodological_novelty=tech_summary.methodological_novelty,
                empirical_findings=tech_summary.empirical_findings,
                paragraph_summary=tech_summary.paragraph_summary,
                model_name=config.llm.model
            )

        repo.update_discovery_run(r_uuid, run_status="summarized")
        console.print(f"[bold green]Summarization completed for RUN ID: {run_id}[/bold green]")


@app.command()
def report(
    run_id: str = typer.Argument(..., help="Discovery run UUID"),
    output: str = typer.Option("report.html", "--output", "-o", help="Path to output HTML report")
):
    """Exports single-file HTML report for a completed discovery run."""
    config = load_config()
    setup_logging(config.application.log_level)
    init_db(config.services.database_url)

    r_uuid = uuid.UUID(run_id)
    with get_db_session() as session:
        repo = LEARepository(session)
        from lea.exporter.html_builder import HTMLReportExporter
        exporter = HTMLReportExporter()
        out_path = exporter.export_report(
            repo,
            r_uuid,
            output,
            title=config.report.title,
            include_abstract_only=config.report.include_abstract_only_results
        )
        console.print(f"[bold green]Successfully generated report at [cyan]{out_path}[/cyan][/bold green]")


@app.command()
def run(
    pdf_path: str = typer.Argument(..., help="Path to input paper PDF"),
    output: str = typer.Option("report.html", "--output", "-o", help="Path to output HTML report"),
    allow_mock: bool = typer.Option(False, "--mock", help="Allow mock LLM backend fallback for testing without GPU")
):
    """Executes full end-to-end literature exploration pipeline."""
    console.print(f"[bold blue]Starting end-to-end LEA pipeline for paper {pdf_path}...[/bold blue]")
    config = load_config()
    setup_logging(config.application.log_level)
    init_db(config.services.database_url)

    from lea.ingester.pdf_parser import PDFParser
    from lea.ingester.grobid_client import GrobidClient
    from lea.ingester.tei_parser import TEIParser
    from lea.ingester.reference_parser import ReferenceParser
    from lea.resolution.identifiers import normalize_doi, normalize_arxiv
    from lea.discovery.openalex import OpenAlexClient

    pdf = PDFParser(pdf_path)
    sha256 = pdf.compute_sha256()

    with get_db_session() as session:
        repo = LEARepository(session)
        paper = repo.get_paper_by_hash(sha256)
        if not paper:
            fallback_meta = pdf.parse_fallback_metadata()
            title = fallback_meta["title"]
            authors = fallback_meta["authors"]
            doi = fallback_meta["doi"]
            arxiv_id = fallback_meta["arxiv_id"]

            grobid_client = GrobidClient(base_url=config.services.grobid_url)
            if asyncio.run(grobid_client.is_alive()):
                try:
                    tei_xml = asyncio.run(grobid_client.process_fulltext(pdf_path))
                    tei_header = TEIParser(tei_xml).parse_header()
                    if tei_header.get("title"):
                        title = tei_header["title"]
                    if tei_header.get("authors"):
                        authors = tei_header["authors"]
                    if tei_header.get("doi"):
                        doi = tei_header["doi"]
                    if tei_header.get("arxiv_id"):
                        arxiv_id = tei_header["arxiv_id"]
                except Exception as exc:
                    logger.warning(f"GROBID error: {exc}")

            paper = repo.create_paper(
                sha256_hash=sha256,
                title=title,
                authors=authors,
                doi=normalize_doi(doi),
                arxiv_id=normalize_arxiv(arxiv_id),
                publication_year=fallback_meta.get("publication_year"),
                pdf_path=pdf_path
            )

            oa_client = OpenAlexClient()
            ref_parser = ReferenceParser(grobid_url=config.services.grobid_url)
            references, status = asyncio.run(ref_parser.extract_references(
                pdf_path=pdf_path,
                doi=doi,
                arxiv_id=arxiv_id,
                openalex_client=oa_client
            ))
            for ref in references:
                repo.add_reference(source_paper_id=paper.id, **ref)

        paper_id = paper.id

    # Pipeline stages
    discover(str(paper_id))

    # Retrieve created run_id
    with get_db_session() as session:
        repo = LEARepository(session)
        runs = repo.get_paper_by_id(paper_id).discovery_runs
        run_id = str(runs[-1].id)

    acquire(run_id)
    index(run_id)
    summarize(run_id, allow_mock=allow_mock)
    report(run_id, output=output)

    console.print(f"[bold green]End-to-end pipeline completed successfully! Output: {output}[/bold green]")


@app.command()
def train(
    dataset_path: str = typer.Argument(..., help="Path to jsonl training dataset"),
    output_dir: str = typer.Option("adapters/lea-qwen", "--output-dir", "-o", help="Adapter output directory")
):
    """Executes optional QLoRA training workflow for Qwen 2.5 7B."""
    from lea.llm.qlora_trainer import train_qlora
    train_qlora(dataset_path=dataset_path, output_dir=output_dir)


if __name__ == "__main__":
    app()
