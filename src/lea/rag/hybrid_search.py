from typing import List, Dict, Any, Tuple
from lea.rag.bm25_index import BM25Index
from lea.rag.dense_search import DenseSearchEngine
from lea.rag.chunker import is_reference_chunk, is_low_quality_chunk
from lea.db.repository import LEARepository

def compute_rrf_fusion(
    dense_results: List[Tuple[Dict[str, Any], float]],
    sparse_results: List[Tuple[Dict[str, Any], float]],
    rrf_k: int = 60,
    fused_top_k: int = 8
) -> List[Tuple[Dict[str, Any], float]]:
    scores: Dict[str, float] = {}
    chunk_map: Dict[str, Dict[str, Any]] = {}

    # Process dense ranks
    for rank, (chunk, _) in enumerate(dense_results, start=1):
        cid = str(chunk.get("id") or chunk.get("chunk_index"))
        chunk_map[cid] = chunk
        scores[cid] = scores.get(cid, 0.0) + (1.0 / (rrf_k + rank))

    # Process sparse ranks
    for rank, (chunk, _) in enumerate(sparse_results, start=1):
        cid = str(chunk.get("id") or chunk.get("chunk_index"))
        chunk_map[cid] = chunk
        scores[cid] = scores.get(cid, 0.0) + (1.0 / (rrf_k + rank))

    sorted_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)
    fused = [(chunk_map[cid], scores[cid]) for cid in sorted_ids[:fused_top_k]]
    return fused


def _is_clean_chunk(chunk: Dict[str, Any]) -> bool:
    """Return True if chunk content is suitable prose to pass to the LLM."""
    content = chunk.get("content", "") or ""
    if is_reference_chunk(content) or is_low_quality_chunk(content):
        return False
    return True


class HybridSearchEngine:
    def __init__(self, dense_engine: DenseSearchEngine, rrf_k: int = 60):
        self.dense_engine = dense_engine
        self.rrf_k = rrf_k

    def hybrid_search(
        self,
        repo: LEARepository,
        run_id,
        paper_id,
        query_text: str,
        dense_top_k: int = 30,
        sparse_top_k: int = 30,
        fused_top_k: int = 8
    ) -> List[Tuple[Dict[str, Any], float]]:
        # 1. Fetch dense vector results for candidate paper
        dense_results = self.dense_engine.search(repo, run_id, paper_id, query_text, top_k=dense_top_k)
        paper_dense = [item for item in dense_results if str(item[0].get("paper_id")) == str(paper_id)]

        # 2. Build BM25 index over candidate paper child chunks
        chunks = repo.get_chunks_for_paper(paper_id, run_id, chunk_type="child")
        chunk_dicts = [
            {
                "id": str(c.id),
                "paper_id": str(c.paper_id),
                "run_id": str(c.run_id),
                "chunk_type": c.chunk_type,
                "parent_id": str(c.parent_id) if c.parent_id else None,
                "content": c.content,
                "chunk_index": c.chunk_index,
                "token_count": c.token_count,
                "section_title": getattr(c, "section_title", None),
                "page_number": getattr(c, "page_number", None)
            }
            for c in chunks
        ]

        bm25_index = BM25Index()
        bm25_index.index_chunks(chunk_dicts)
        sparse_results = bm25_index.search(query_text, top_k=sparse_top_k)

        # 3. Reciprocal Rank Fusion
        fused = compute_rrf_fusion(paper_dense, sparse_results, rrf_k=self.rrf_k, fused_top_k=fused_top_k * 3)

        # 4. Post-retrieval filter: drop reference lists and low-quality noise chunks
        clean = [(chunk, score) for chunk, score in fused if _is_clean_chunk(chunk)]
        return clean[:fused_top_k]

