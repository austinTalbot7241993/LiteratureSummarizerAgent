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

ARXIV_ID_PATTERN = re.compile(r"\d{4}\.\d{4,5}(?:v\d+)?")

def normalize_arxiv(arxiv_id: Optional[str]) -> Optional[str]:
    if not arxiv_id:
        return None
    a = arxiv_id.strip().lower()
    for prefix in ["https://arxiv.org/abs/", "http://arxiv.org/abs/", "arxiv:"]:
        if a.startswith(prefix):
            a = a[len(prefix):]
    # Extract just the valid arXiv-ID-shaped substring (YYMM.NNNNN[vN]) via
    # search rather than trusting the whole remaining string is clean and
    # merely stripping a trailing "vN". A raw GROBID/PDF-text extraction can
    # glue adjacent watermark text onto the ID with no separator (observed:
    # "2608.12838v1[math.st]" from an arXiv sidebar watermark rendered
    # across a line break), which the old trailing-anchor regex (`v\d+$`)
    # would silently fail to strip since the string no longer ends in just
    # "vN". Searching for the ID pattern and discarding everything else
    # (including any version suffix, matching prior behavior) is robust to
    # whatever garbage is attached before or after it.
    match = ARXIV_ID_PATTERN.search(a)
    if not match:
        return None
    return re.sub(r"v\d+$", "", match.group(0))

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
