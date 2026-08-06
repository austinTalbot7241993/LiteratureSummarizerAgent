import asyncio
import time
import httpx
from typing import Dict, Any, List, Optional
from lea.logging import logger
from lea.resolution.identifiers import normalize_doi, normalize_arxiv, normalize_openalex_id

class OpenAlexClient:
    def __init__(self, api_key: Optional[str] = None, rate_limit_rps: float = 5.0, timeout: float = 30.0):
        self.api_key = api_key
        self.rate_limit_delay = 1.0 / rate_limit_rps if rate_limit_rps > 0 else 0.2
        self.timeout = timeout
        self.last_request_time = 0.0

    async def _rate_limit(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()

    async def get_work(self, doi: Optional[str] = None, arxiv_id: Optional[str] = None, openalex_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        await self._rate_limit()
        identifier = None
        if openalex_id:
            identifier = f"W{normalize_openalex_id(openalex_id)}"
        elif doi:
            identifier = f"https://doi.org/{normalize_doi(doi)}"
        elif arxiv_id:
            identifier = f"https://arxiv.org/abs/{normalize_arxiv(arxiv_id)}"

        if not identifier:
            return None

        url = f"https://api.openalex.org/works/{identifier}"
        params = {}
        if self.api_key:
            params["api_key"] = self.api_key

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await client.get(url, params=params)
                if res.status_code == 200:
                    return self._parse_work(res.json())
                return None
        except Exception as exc:
            logger.warning(f"OpenAlex get_work error: {exc}")
            return None

    async def fetch_work_references(self, doi: Optional[str] = None, arxiv_id: Optional[str] = None) -> List[Dict[str, Any]]:
        work = await self.get_work(doi=doi, arxiv_id=arxiv_id)
        if not work or "referenced_works" not in work:
            return []

        ref_ids = work.get("referenced_works", [])[:100]
        if not ref_ids:
            return []

        results = []
        # Batch query openalex works in chunks of 50
        for i in range(0, len(ref_ids), 50):
            chunk = ref_ids[i:i+50]
            clean_chunk = [c.split("/")[-1] for c in chunk]
            filter_str = "|".join(clean_chunk)
            url = f"https://api.openalex.org/works?filter=openalex_id:{filter_str}&per-page=50"
            await self._rate_limit()
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    res = await client.get(url)
                    if res.status_code == 200:
                        items = res.json().get("results", [])
                        for item in items:
                            parsed = self._parse_work(item)
                            parsed["extraction_method"] = "openalex"
                            results.append(parsed)
            except Exception as exc:
                logger.warning(f"OpenAlex fetch_work_references batch error: {exc}")

        return results

    async def find_related_candidates(self, work_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        await self._rate_limit()
        clean_id = work_id.split("/")[-1]
        url = f"https://api.openalex.org/works?filter=related_to:{clean_id}&per-page={min(limit, 100)}"
        params = {}
        if self.api_key:
            params["api_key"] = self.api_key

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await client.get(url, params=params)
                if res.status_code == 200:
                    results = res.json().get("results", [])
                    return [self._parse_work(w) for w in results]
                return []
        except Exception as exc:
            logger.warning(f"OpenAlex find_related_candidates error: {exc}")
            return []

    async def search_candidates(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        await self._rate_limit()
        import urllib.parse
        encoded_query = urllib.parse.quote_plus(query)
        url = f"https://api.openalex.org/works?search={encoded_query}&per-page={min(limit, 50)}"
        params = {}
        if self.api_key:
            params["api_key"] = self.api_key

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await client.get(url, params=params)
                if res.status_code == 200:
                    results = res.json().get("results", [])
                    return [self._parse_work(w) for w in results]
                return []
        except Exception as exc:
            logger.warning(f"OpenAlex search_candidates error: {exc}")
            return []


    def _parse_work(self, item: Dict[str, Any]) -> Dict[str, Any]:
        openalex_id = item.get("id", "").split("/")[-1] if item.get("id") else None
        doi = item.get("doi")
        arxiv_id = None
        ids = item.get("ids", {})
        if "arxiv" in ids:
            arxiv_id = ids["arxiv"].split("/")[-1]

        authorships = item.get("authorships", [])
        authors = [a.get("author", {}).get("display_name") for a in authorships if a.get("author", {}).get("display_name")]

        biblio = item.get("biblio", {})
        year = item.get("publication_year") or biblio.get("pub_year")

        primary_loc = item.get("primary_location", {}) or {}
        source = primary_loc.get("source", {}) or {}
        venue = source.get("display_name")

        oa_info = item.get("open_access", {}) or {}
        is_oa = oa_info.get("is_oa", False)
        oa_url = oa_info.get("oa_url")

        # Inverted abstract reconstruction if present
        abstract = None
        inv_abs = item.get("abstract_inverted_index")
        if inv_abs:
            word_positions = []
            for word, positions in inv_abs.items():
                for pos in positions:
                    word_positions.append((pos, word))
            word_positions.sort(key=lambda x: x[0])
            abstract = " ".join([w for pos, w in word_positions])

        return {
            "title": item.get("display_name") or item.get("title", ""),
            "authors": authors,
            "doi": normalize_doi(doi),
            "arxiv_id": normalize_arxiv(arxiv_id),
            "openalex_id": openalex_id,
            "publication_year": year,
            "venue": venue,
            "abstract": abstract,
            "is_open_access": is_oa,
            "oa_pdf_url": oa_url,
            "referenced_works": item.get("referenced_works", [])
        }
