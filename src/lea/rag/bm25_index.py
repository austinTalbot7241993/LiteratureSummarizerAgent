from typing import List, Dict, Any, Tuple, Optional
import math
from lea.logging import logger

class BM25Index:
    def __init__(self):
        self.retriever = None
        self.corpus_chunks = []
        self._bm25s_available = False

        try:
            import bm25s
            self._bm25s_available = True
        except ImportError:
            logger.warning("bm25s module not installed; using pure Python BM25 fallback.")

    def index_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        self.corpus_chunks = chunks
        if not chunks:
            return

        corpus_texts = [c["content"] for c in chunks]

        if self._bm25s_available:
            try:
                import bm25s
                corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="english")
                self.retriever = bm25s.BM25()
                self.retriever.index(corpus_tokens)
                return
            except Exception as exc:
                logger.warning(f"bm25s index creation failed: {exc}")

        # Pure Python BM25 initialization fallback
        self._init_python_bm25(corpus_texts)

    def _init_python_bm25(self, corpus_texts: List[str]):
        self.doc_tokens = [t.lower().split() for t in corpus_texts]
        self.N = len(self.doc_tokens)
        self.avgdl = sum(len(d) for d in self.doc_tokens) / self.N if self.N > 0 else 1.0
        self.df = {}
        for d in self.doc_tokens:
            for term in set(d):
                self.df[term] = self.df.get(term, 0) + 1

    def search(self, query: str, top_k: int = 30) -> List[Tuple[Dict[str, Any], float]]:
        if not self.corpus_chunks:
            return []

        if self._bm25s_available and self.retriever:
            try:
                import bm25s
                query_tokens = bm25s.tokenize([query], stopwords="english")
                results, scores = self.retriever.retrieve(query_tokens, k=min(top_k, len(self.corpus_chunks)))
                # Extract results
                matched = []
                for idx, score in zip(results[0], scores[0]):
                    matched.append((self.corpus_chunks[idx], float(score)))
                return matched
            except Exception as exc:
                logger.warning(f"bm25s search error: {exc}")

        # Python BM25 fallback search calculation
        q_terms = query.lower().split()
        scores = []
        k1 = 1.5
        b = 0.75

        for idx, doc in enumerate(self.doc_tokens):
            doc_len = len(doc)
            score = 0.0
            for term in q_terms:
                if term in self.df:
                    f = doc.count(term)
                    idf = math.log((self.N - self.df[term] + 0.5) / (self.df[term] + 0.5) + 1.0)
                    denom = f + k1 * (1 - b + b * (doc_len / self.avgdl))
                    score += idf * (f * (k1 + 1)) / denom
            scores.append((self.corpus_chunks[idx], score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
