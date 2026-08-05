import asyncio
import time
import httpx
from typing import Dict, Any, List, Optional
from lea.logging import logger
from lea.resolution.identifiers import normalize_doi, normalize_arxiv

class SemanticScholarClient:
    def __init__(self, api_key: Optional[str] = None, rate_limit_rps: float = 0.2, timeout: float = 30.0):
        self.api_key = api_key
        self.rate_limit_delay = 1.0 / rate_limit_rps if rate_limit_rps > 0 else 5.0
        self.timeout = timeout
        self.last_request_time = 0.0

    async def _rate_limit(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()

    def _headers(self) -> Dict[str, str]:
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    async def get_paper_recommendations(self, paper_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        await self._rate_limit()
        url = f"https://api.semanticscholar.org/recommendations/v1/papers/forpaper/{paper_id}"
        params = {
            "fields": "paperId,externalIds,title,authors,year,venue,abstract,isOpenAccess,openAccessPdf",
            "limit": min(limit, 100)
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await client.get(url, params=params, headers=self._headers())
                if res.status_code == 200:
                    data = res.json()
                    recommendations = data.get("recommendedPapers", [])
                    return [self._parse_paper(p) for p in recommendations]
                return []
        except Exception as exc:
            logger.warning(f"Semantic Scholar recommendations error: {exc}")
            return []

    def _parse_paper(self, item: Dict[str, Any]) -> Dict[str, Any]:
        ext_ids = item.get("externalIds", {}) or {}
        authors = [a.get("name") for a in item.get("authors", []) if a.get("name")]
        oa_pdf = item.get("openAccessPdf", {}) or {}

        return {
            "title": item.get("title", ""),
            "authors": authors,
            "doi": normalize_doi(ext_ids.get("DOI")),
            "arxiv_id": normalize_arxiv(ext_ids.get("ArXiv")),
            "s2_id": item.get("paperId"),
            "publication_year": item.get("year"),
            "venue": item.get("venue"),
            "abstract": item.get("abstract"),
            "is_open_access": item.get("isOpenAccess", False),
            "oa_pdf_url": oa_pdf.get("url")
        }
