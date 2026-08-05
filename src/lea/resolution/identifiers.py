import re
from typing import Optional

def normalize_doi(doi: Optional[str]) -> Optional[str]:
    if not doi:
        return None
    d = doi.strip().lower()
    for prefix in ["https://doi.org/", "http://doi.org/", "doi:"]:
        if d.startswith(prefix):
            d = d[len(prefix):]
    d = d.strip("/ ")
    return d if d else None

def normalize_arxiv(arxiv_id: Optional[str]) -> Optional[str]:
    if not arxiv_id:
        return None
    a = arxiv_id.strip().lower()
    for prefix in ["https://arxiv.org/abs/", "http://arxiv.org/abs/", "arxiv:"]:
        if a.startswith(prefix):
            a = a[len(prefix):]
    # Strip version suffix if present e.g. 2301.12345v2 -> 2301.12345
    a = re.sub(r"v\d+$", "", a)
    return a if a else None

def normalize_openalex_id(openalex_id: Optional[str]) -> Optional[str]:
    if not openalex_id:
        return None
    o = openalex_id.strip()
    for prefix in ["https://openalex.org/", "openalex:"]:
        if o.startswith(prefix):
            o = o[len(prefix):]
    return o.upper() if o else None

def normalize_s2_id(s2_id: Optional[str]) -> Optional[str]:
    if not s2_id:
        return None
    s = s2_id.strip()
    for prefix in ["https://api.semanticscholar.org/", "CorpusId:"]:
        if s.startswith(prefix):
            s = s[len(prefix):]
    return s if s else None

def normalize_title(title: Optional[str]) -> str:
    if not title:
        return ""
    t = title.lower().strip()
    # Remove non-alphanumeric characters except spaces
    t = re.sub(r"[^\w\s]", " ", t)
    # Collapse multiple spaces
    t = re.sub(r"\s+", " ", t).strip()
    return t
