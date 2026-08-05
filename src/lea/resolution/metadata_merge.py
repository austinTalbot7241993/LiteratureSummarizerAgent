from typing import Dict, Any, Optional
from lea.resolution.identifiers import (
    normalize_doi, normalize_arxiv, normalize_openalex_id, normalize_s2_id
)

def merge_paper_metadata(primary: Dict[str, Any], secondary: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(primary)

    for key, val in secondary.items():
        if val is None or val == "" or val == []:
            continue
        if key not in merged or merged[key] is None or merged[key] == "" or merged[key] == []:
            merged[key] = val

    # Normalize identifiers in merged dictionary
    if merged.get("doi"):
        merged["doi"] = normalize_doi(merged["doi"])
    if merged.get("arxiv_id"):
        merged["arxiv_id"] = normalize_arxiv(merged["arxiv_id"])
    if merged.get("openalex_id"):
        merged["openalex_id"] = normalize_openalex_id(merged["openalex_id"])
    if merged.get("s2_id"):
        merged["s2_id"] = normalize_s2_id(merged["s2_id"])

    return merged
