import re
import uuid
from typing import List, Dict, Any, Optional
from lea.db.repository import LEARepository

HEADING_PATTERNS = [
    r"\bdata\s+availability\b",
    r"\bavailability\s+of\s+data(?:\s+and\s+materials)?\b",
    r"\bdata\s+sharing\b",
    r"\bdata\s+access\b",
    r"\bcode\s+and\s+data\s+availability\b",
    r"\bsupplementary\s+data\b",
]

PHRASE_PATTERNS = [
    r"\bavailable\s+upon\s+reasonable\s+request\b",
    r"\breasonable\s+request\b",
    r"\bcontrolled\s+access\b",
    r"\bmanaged\s+access\b",
    r"\bdata[- ]use\s+agreement\b",
    r"\bdata\s+access\s+committee\b",
    r"\bapplication\s+required\b",
    r"\bauthorization\b",
    r"\bcredentialed\s+access\b",
    r"\bcannot\s+be\s+shared\b",
    r"\bnot\s+publicly\s+available\b",
    r"\bprivacy\s+restrictions?\b",
    r"\bethical\s+restrictions?\b",
    r"\bdeposited\s+in\b",
    r"\baccession\b",
]

ACCESSION_PATTERNS = [
    r"\bGSE\d+\b",
    r"\b(?:SRP|SRR|SRS|SRX)\d+\b",
    r"\b(?:PRJNA|PRJEB|PRJDB)\d+\b",
    r"\bphs\d+(?:\.v\d+\.p\d+)?\b",
    r"\b(?:EGAS|EGAD|EGAN)\d+\b",
    r"\bPXD\d+\b",
    r"\bE-[A-Z]{4}-\d+\b",
    r"\bds\d+\b",
    r"\bsyn\d+\b",
    r"\bMTBLS\d+\b",
]

ALL_PATTERNS = HEADING_PATTERNS + PHRASE_PATTERNS + ACCESSION_PATTERNS


def score_chunk_for_availability(content: str, section_title: Optional[str] = None) -> float:
    text_to_check = f"{section_title or ''}\n{content}"
    score = 0.0

    for pat in HEADING_PATTERNS:
        if re.search(pat, text_to_check, re.IGNORECASE):
            score += 10.0

    for pat in ACCESSION_PATTERNS:
        if re.search(pat, text_to_check, re.IGNORECASE):
            score += 8.0

    for pat in PHRASE_PATTERNS:
        if re.search(pat, text_to_check, re.IGNORECASE):
            score += 5.0

    return score


def retrieve_data_availability_context(
    repo: LEARepository,
    run_id: uuid.UUID,
    paper_id: uuid.UUID,
    query_embedding: Optional[List[float]] = None,
    max_tokens: int = 1500
) -> List[Dict[str, Any]]:
    # Retrieve all child chunks for this paper
    chunks = repo.get_chunks_for_paper(paper_id, run_id, chunk_type="child")
    if not chunks:
        return []

    chunk_map = {c.chunk_index: c for c in chunks}
    scores: Dict[int, float] = {}

    # Rule-based scoring (headings, accessions, key phrases)
    for index, c in chunk_map.items():
        s = score_chunk_for_availability(c.content, getattr(c, "section_title", None))
        if s > 0:
            scores[index] = scores.get(index, 0.0) + s

    # Dense search scoring if embedding provided
    if query_embedding:
        try:
            dense_chunks = repo.search_dense_vector(run_id, paper_id, query_embedding, top_k=10)
            for rank, c in enumerate(dense_chunks, start=1):
                scores[c.chunk_index] = scores.get(c.chunk_index, 0.0) + (10.0 / rank)
        except Exception:
            pass

    if not scores:
        # Fallback: inspect all chunks for any subtle mentions or return initial/final chunks
        for index, c in chunk_map.items():
            if "data" in c.content.lower():
                scores[index] = 1.0

    # Expand selected chunks with immediate neighbors (index-1, index+1) to catch split headers
    expanded_indices = set()
    for index in sorted(scores.keys(), key=lambda i: scores[i], reverse=True):
        expanded_indices.add(index)
        if index - 1 in chunk_map:
            expanded_indices.add(index - 1)
        if index + 1 in chunk_map:
            expanded_indices.add(index + 1)

    # Sort chunks by original index order to preserve logical reading order
    selected_indices = sorted(list(expanded_indices), key=lambda i: (scores.get(i, 0.0) == 0, i))
    
    # Filter by budget at complete chunk boundaries
    selected_chunks = []
    total_tokens = 0

    for idx in selected_indices:
        c = chunk_map[idx]
        token_count = c.token_count or len(c.content.split())
        if total_tokens + token_count > max_tokens and selected_chunks:
            break
        selected_chunks.append({
            "id": str(c.id),
            "paper_id": str(c.paper_id),
            "run_id": str(c.run_id),
            "chunk_type": c.chunk_type,
            "parent_id": str(c.parent_id) if c.parent_id else None,
            "content": c.content,
            "chunk_index": c.chunk_index,
            "token_count": token_count,
            "section_title": getattr(c, "section_title", None),
            "page_number": getattr(c, "page_number", None)
        })
        total_tokens += token_count

    # Re-sort selected chunks strictly by chunk_index
    selected_chunks.sort(key=lambda x: x["chunk_index"])
    return selected_chunks


def format_availability_context(chunks: List[Dict[str, Any]]) -> str:
    if not chunks:
        return "No text chunks available for data availability extraction."

    formatted = []
    for c in chunks:
        cid = c.get("id")
        cidx = c.get("chunk_index")
        sec = c.get("section_title")
        page = c.get("page_number")
        meta_parts = [f"id={cid}", f"index={cidx}"]
        if sec:
            meta_parts.append(f'section="{sec}"')
        if page is not None:
            meta_parts.append(f"page={page}")
        header = f"[{' '.join(meta_parts)}]"
        formatted.append(f"{header}\n{c.get('content', '').strip()}")

    return "\n\n".join(formatted)
