from typing import List, Dict, Any, Tuple, Optional
from lea.logging import logger

class HierarchicalChunker:
    def __init__(
        self,
        tokenizer_model: str = "Qwen/Qwen2.5-7B-Instruct",
        parent_tokens: int = 768,
        parent_overlap_tokens: int = 96,
        child_tokens: int = 256,
        child_overlap_tokens: int = 48
    ):
        self.parent_tokens = parent_tokens
        self.parent_overlap = parent_overlap_tokens
        self.child_tokens = child_tokens
        self.child_overlap = child_overlap_tokens
        self.tokenizer = None

        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, trust_remote_code=True)
        except Exception:
            logger.warning("Could not load HuggingFace AutoTokenizer; falling back to whitespace tokenization.")

    def tokenize(self, text: str) -> List[str]:
        if self.tokenizer:
            try:
                return self.tokenizer.tokenize(text)
            except Exception:
                pass
        return text.split()

    def decode(self, tokens: List[str]) -> str:
        if self.tokenizer:
            try:
                # If tokens are ids or strings
                return self.tokenizer.convert_tokens_to_string(tokens)
            except Exception:
                pass
        return " ".join(tokens)

    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Splits text into parent chunks and nested child chunks.
        Returns a list of dicts containing chunk metadata.
        """
        tokens = self.tokenize(text)
        if not tokens:
            return []

        chunks = []
        global_chunk_idx = 0

        # Step 1: Create Parent Chunks
        parent_step = self.parent_tokens - self.parent_overlap
        if parent_step <= 0:
            parent_step = self.parent_tokens

        for p_start in range(0, len(tokens), parent_step):
            p_end = min(p_start + self.parent_tokens, len(tokens))
            p_tokens = tokens[p_start:p_end]
            p_content = self.decode(p_tokens)

            parent_idx = global_chunk_idx
            global_chunk_idx += 1

            chunks.append({
                "chunk_index": parent_idx,
                "chunk_type": "parent",
                "parent_id": None,
                "content": p_content,
                "token_count": len(p_tokens)
            })

            # Step 2: Create Child Chunks inside this parent window
            child_step = self.child_tokens - self.child_overlap
            if child_step <= 0:
                child_step = self.child_tokens

            for c_start in range(p_start, p_end, child_step):
                c_end = min(c_start + self.child_tokens, p_end)
                c_tokens = tokens[c_start:c_end]
                c_content = self.decode(c_tokens)

                child_idx = global_chunk_idx
                global_chunk_idx += 1

                chunks.append({
                    "chunk_index": child_idx,
                    "chunk_type": "child",
                    "parent_index": parent_idx,
                    "content": c_content,
                    "token_count": len(c_tokens)
                })

                if c_end >= p_end:
                    break

        return chunks
