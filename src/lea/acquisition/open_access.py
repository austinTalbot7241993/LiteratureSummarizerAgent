import httpx
from typing import Optional, Dict, Any
from lea.logging import logger
from lea.resolution.identifiers import normalize_doi

class OpenAccessResolver:
    def __init__(self, unpaywall_email: Optional[str] = None, timeout: float = 15.0):
        self.unpaywall_email = unpaywall_email
        self.timeout = timeout

    async def resolve_oa_url(self, paper_meta: Dict[str, Any]) -> Optional[str]:
        # 1. Existing OA URL on metadata
        if paper_meta.get("oa_pdf_url"):
            return paper_meta["oa_pdf_url"]

        # 2. Unpaywall API query if DOI present
        doi = normalize_doi(paper_meta.get("doi"))
        if doi and self.unpaywall_email:
            try:
                url = f"https://api.unpaywall.org/v2/{doi}?email={self.unpaywall_email}"
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    res = await client.get(url)
                    if res.status_code == 200:
                        data = res.json()
                        best_location = data.get("best_oa_location") or {}
                        pdf_url = best_location.get("url_for_pdf") or best_location.get("url")
                        if pdf_url:
                            return pdf_url
            except Exception as exc:
                logger.warning(f"Unpaywall resolution failed for DOI {doi}: {exc}")

        # 3. ArXiv PDF direct link if arXiv ID present
        arxiv_id = paper_meta.get("arxiv_id")
        if arxiv_id:
            return f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        return None
