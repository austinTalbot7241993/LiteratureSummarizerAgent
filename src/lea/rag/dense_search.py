from typing import List, Tuple, Dict, Any
from lea.db.repository import LEARepository
from lea.rag.embedder import BGEEmbedder

class DenseSearchEngine:
    def __init__(self, embedder: BGEEmbedder):
        self.embedder = embedder

    def search(self, repo: LEARepository, run_id, query_text: str, top_k: int = 30) -> List[Tuple[Dict[str, Any], float]]:
        query_embeddings = self.embedder.embed_texts([query_text])
        if not query_embeddings:
            return []
        q_vec = query_embeddings[0]

        chunk_models = repo.search_dense_vector(run_id, q_vec, top_k=top_k)
        results = []
        for c in chunk_models:
            chunk_dict = {
                "id": str(c.id),
                "paper_id": str(c.paper_id),
                "run_id": str(c.run_id),
                "chunk_type": c.chunk_type,
                "parent_id": str(c.parent_id) if c.parent_id else None,
                "content": c.content,
                "chunk_index": c.chunk_index,
                "token_count": c.token_count
            }
            results.append((chunk_dict, 1.0)) # Rank-ordered
        return results
