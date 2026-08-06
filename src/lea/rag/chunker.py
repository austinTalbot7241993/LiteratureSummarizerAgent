import re
from typing import List, Dict, Any, Tuple, Optional
from lea.logging import logger

def is_reference_chunk(text: str) -> bool:
    """Return True if this text chunk looks like a bibliography/reference section rather than body text."""
    stripped = text.strip()

    # Catch comma-separated citation index lists: ",14,15,16,17,18,..."
    comma_numbers = re.findall(r'(?:^|,)\s*(\d+)\s*(?=,|$)', stripped)
    if len(comma_numbers) >= 5:
        non_number_tokens = [t for t in re.split(r'[,\s]+', stripped) if t and not t.isdigit()]
        if len(non_number_tokens) < 3:
            return True

    # Catch repetitive author name loops: same surname appearing 4+ times in a row
    surnames = re.findall(r'(?:^|,)\s*([A-Z][a-z]+)\s*,', stripped)
    if len(surnames) >= 6:
        from collections import Counter
        top_surname, top_count = Counter(surnames).most_common(1)[0]
        if top_count >= 4:
            return True

    # Catch URL data dumps, S3 bucket links, file downloads (e.g. s3-us-west-2.amazonaws.com, .tar.gz)
    url_count = len(re.findall(r"https?://|\.tar\.gz|\.bcf|\.vcf|s3[.-]|\.fasta\b", text, re.IGNORECASE))
    if url_count >= 2:
        return True

    # Catch Supplementary Information / Figure / Table lists
    supp_count = len(re.findall(r"\bSupplementary (?:Figure|Table|Note|Information|File|Movie)\b", text, re.IGNORECASE))
    if supp_count >= 2:
        return True

    # Standard bibliography heuristics
    et_al_count = len(re.findall(r'\bet al\b', text, re.IGNORECASE))
    doi_count = len(re.findall(r'10\.\d{4,9}/', text))
    citation_year_pattern = len(re.findall(r'\b(?:19|20)\d{2}\s*;\s*\d+', text))
    numbered_citations = len(re.findall(r'^\s*\d{1,3}\.\s+[A-Z]', text, re.MULTILINE))

    if et_al_count >= 2 or doi_count >= 2 or citation_year_pattern >= 2 or numbered_citations >= 3:
        return True
    return False


def is_low_quality_chunk(text: str, min_words: int = 20) -> bool:
    """Return True if this chunk has too little actual prose to be useful for summarization."""
    words = re.findall(r'[a-zA-Z]{3,}', text)
    if len(words) < min_words:
        return True

    # Reject chunks with high URL density
    if "https://" in text or "http://" in text or "s3-" in text:
        if len(re.findall(r"https?://\S+|s3-\S+", text)) >= 2:
            return True

    # Reject chunks that are mostly punctuation/numbers (e.g. only page numbers, figure labels)
    all_tokens = re.split(r'\s+', text.strip())
    if not all_tokens:
        return True
    noise_tokens = [t for t in all_tokens if re.fullmatch(r'[^a-zA-Z]+', t)]
    if len(noise_tokens) / len(all_tokens) > 0.5:
        return True
    return False

class HierarchicalChunker:
    def __init__(
        self,
        tokenizer_model: str = "BAAI/bge-m3",
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

        chunks: List[Dict[str, Any]] = []
        parent_step = self.parent_tokens - self.parent_overlap
        if parent_step <= 0:
            parent_step = self.parent_tokens
        child_step = self.child_tokens - self.child_overlap
        if child_step <= 0:
            child_step = self.child_tokens

        global_chunk_idx = 0

        # Step 1: Create Parent Chunks
        for p_start in range(0, len(tokens), parent_step):
            p_end = min(p_start + self.parent_tokens, len(tokens))
            p_tokens = tokens[p_start:p_end]
            p_content = self.decode(p_tokens)

            if is_reference_chunk(p_content) or is_low_quality_chunk(p_content):
                continue

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
            for c_start in range(p_start, p_end, child_step):
                c_end = min(c_start + self.child_tokens, p_end)
                c_tokens = tokens[c_start:c_end]
                c_content = self.decode(c_tokens)

                if is_reference_chunk(c_content) or is_low_quality_chunk(c_content):
                    continue

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
