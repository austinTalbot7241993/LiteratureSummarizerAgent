import re
from typing import Dict, Any

def generate_bibtex_key(meta: Dict[str, Any]) -> str:
    authors = meta.get("authors") or []
    first_author = "unknown"
    if authors:
        # Get surname of first author
        parts = authors[0].strip().split()
        first_author = parts[-1].lower() if parts else "unknown"
    first_author = re.sub(r"[^\w]", "", first_author)

    year = str(meta.get("publication_year") or meta.get("year") or "nodate")

    title = meta.get("title") or "paper"
    first_title_word = "paper"
    words = [w.lower() for w in re.sub(r"[^\w\s]", "", title).split() if len(w) > 3]
    if words:
        first_title_word = words[0]

    return f"{first_author}{year}{first_title_word}"

def generate_bibtex(meta: Dict[str, Any]) -> str:
    """
    Generates a deterministic BibTeX string from paper metadata.
    Does NOT use LLM generation.
    """
    if meta.get("raw_bibtex") and "@" in meta["raw_bibtex"]:
        return meta["raw_bibtex"].strip()

    key = generate_bibtex_key(meta)
    title = meta.get("title", "Untitled Paper")
    authors = " and ".join(meta.get("authors", [])) or "Unknown Author"
    year = str(meta.get("publication_year") or meta.get("year") or "")
    venue = meta.get("venue") or "arXiv preprint" if meta.get("arxiv_id") else "Publication"
    doi = meta.get("doi")
    arxiv_id = meta.get("arxiv_id")

    lines = [
        f"@article{{{key},",
        f"  title = {{{title}}},",
        f"  author = {{{authors}}},"
    ]

    if year:
        lines.append(f"  year = {{{year}}},")
    if venue:
        lines.append(f"  journal = {{{venue}}},")
    if doi:
        lines.append(f"  doi = {{{doi}}},")
    if arxiv_id:
        lines.append(f"  eprint = {{{arxiv_id}}},")
        lines.append("  archiveprefix = {arXiv},")

    # Remove trailing comma from last field if present
    if lines[-1].endswith(","):
        lines[-1] = lines[-1][:-1]

    lines.append("}")
    return "\n".join(lines)
