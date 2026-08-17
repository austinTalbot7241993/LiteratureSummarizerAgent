import asyncio
import time
import httpx
from typing import Dict, Any, List, Optional
from lea.logging import logger
from lea.resolution.identifiers import normalize_doi, normalize_arxiv

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_RETRIES = 3
BASE_BACKOFF_SECONDS = 1.5

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

    async def _get_with_retry(self, client: httpx.AsyncClient, url: str, **kwargs) -> httpx.Response:
        """GET with bounded exponential-backoff retry on 429/5xx. See the
        identical rationale in OpenAlexClient._get_with_retry -- Semantic
        Scholar's unauthenticated tier (no SEMANTIC_SCHOLAR_API_KEY
        configured) is especially prone to 429, and a transient throttle was
        previously indistinguishable from "genuinely no related papers."
        """
        res = None
        for attempt in range(MAX_RETRIES + 1):
            res = await client.get(url, **kwargs)
            if res.status_code not in RETRYABLE_STATUS_CODES:
                return res
            if attempt == MAX_RETRIES:
                break
            retry_after = res.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after else BASE_BACKOFF_SECONDS * (2 ** attempt)
            except ValueError:
                delay = BASE_BACKOFF_SECONDS * (2 ** attempt)
            logger.warning(
                f"Semantic Scholar request to {url} returned {res.status_code}; "
                f"retrying in {delay:.1f}s (attempt {attempt + 1}/{MAX_RETRIES})"
            )
            await asyncio.sleep(delay)
        return res

    def _headers(self) -> Dict[str, str]:
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    async def search_papers(self, query: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Plain-text keyword search, independent of any citation-graph
        identity. Used as a discovery fallback when the input paper has no
        DOI/arXiv/S2 ID to anchor a recommendations-based lookup (e.g. a
        brand-new preprint with no citations yet) -- `get_paper_recommendations`
        cannot run at all in that case, so without this, S2 contributes
        nothing to discovery for such papers.
        """
        await self._rate_limit()
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "fields": "paperId,externalIds,title,authors,year,venue,abstract,isOpenAccess,openAccessPdf",
            "limit": min(limit, 100)
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await self._get_with_retry(client, url, params=params, headers=self._headers())
                if res.status_code == 200:
                    data = res.json()
                    results = data.get("data", [])
                    return [self._parse_paper(p) for p in results]
                return []
        except Exception as exc:
            logger.warning(f"Semantic Scholar search error: {exc}")
            return []

    async def get_paper_recommendations(self, paper_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        await self._rate_limit()
        url = f"https://api.semanticscholar.org/recommendations/v1/papers/forpaper/{paper_id}"
        params = {
            "fields": "paperId,externalIds,title,authors,year,venue,abstract,isOpenAccess,openAccessPdf",
            "limit": min(limit, 100)
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await self._get_with_retry(client, url, params=params, headers=self._headers())
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
