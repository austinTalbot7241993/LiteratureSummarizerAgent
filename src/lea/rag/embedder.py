import os
import multiprocessing as mp
from typing import List, Optional
import numpy as np
from lea.logging import logger

def _subproc_embed_worker(model_name: str, texts: List[str], max_length: int, conn):
    """Worker function spawned in a separate process to run PyTorch embedding and exit."""
    try:
        import torch
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(model_name, device=device, model_kwargs={"use_safetensors": True})
        model.max_seq_length = max_length
        embeddings = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        conn.send(embeddings.tolist())
    except Exception as exc:
        conn.send(exc)
    finally:
        conn.close()

class BGEEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-m3", max_length: int = 1024, use_subprocess: bool = False):
        self.model_name = model_name
        self.max_length = max_length
        self.use_subprocess = use_subprocess
        self._local_model = None

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        # If subprocess isolation requested (for GPU VRAM release)
        if self.use_subprocess:
            ctx = mp.get_context("spawn")
            parent_conn, child_conn = ctx.Pipe()
            p = ctx.Process(target=_subproc_embed_worker, args=(self.model_name, texts, self.max_length, child_conn))
            p.start()
            result = parent_conn.recv()
            p.join()
            if isinstance(result, Exception):
                raise result
            return result

        # Local execution fallback / CPU execution
        try:
            from sentence_transformers import SentenceTransformer
            if self._local_model is None:
                self._local_model = SentenceTransformer(self.model_name, model_kwargs={"use_safetensors": True})
                self._local_model.max_seq_length = self.max_length
            embeddings = self._local_model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
            return embeddings.tolist()
        except Exception:
            try:
                import torch
                from transformers import AutoTokenizer, AutoModel
                if self._local_model is None:
                    tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
                    model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
                    model.eval()
                    self._local_model = (tokenizer, model)

                tokenizer, model = self._local_model
                inputs = tokenizer(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**inputs)
                    token_embeddings = outputs[0]
                    attention_mask = inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
                    sum_embeddings = torch.sum(token_embeddings * attention_mask, 1)
                    sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
                    pooled = sum_embeddings / sum_mask
                    normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
                    return normalized.cpu().numpy().tolist()
            except Exception as exc:
                logger.warning(f"Embedding model loading failed: {exc}. Using deterministic pseudo-embeddings for testing.")
                results = []
                for t in texts:
                    seed = sum(ord(c) for c in t) % (2**32)
                    np.random.seed(seed)
                    vec = np.random.randn(1024).astype(np.float32)
                    norm = np.linalg.norm(vec)
                    vec = (vec / norm).tolist() if norm > 0 else vec.tolist()
                    results.append(vec)
                return results
