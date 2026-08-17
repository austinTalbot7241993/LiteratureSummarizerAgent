import asyncio
import time
import httpx
from typing import Dict, Any, List, Optional
from lea.logging import logger
from lea.resolution.identifiers import normalize_doi, normalize_arxiv, normalize_openalex_id

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_RETRIES = 3
BASE_BACKOFF_SECONDS = 1.5

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

    async def _get_with_retry(self, client: httpx.AsyncClient, url: str, **kwargs) -> httpx.Response:
        """GET with bounded exponential-backoff retry on 429/5xx.

        Every call site previously branched only on `status_code == 200`,
        so a transient rate-limit or server error was treated identically
        to "confirmed zero results" -- a single 429 silently produced "no
        candidates" for the whole discovery run instead of being retried
        (confirmed live: OpenAlex/Semantic Scholar both returned 429 mid-run
        with no OPENALEX_API_KEY/SEMANTIC_SCHOLAR_API_KEY configured, which
        put every request in the strictest unauthenticated tier). Honors a
        `Retry-After` header when the server provides one.
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
                f"OpenAlex request to {url} returned {res.status_code}; "
                f"retrying in {delay:.1f}s (attempt {attempt + 1}/{MAX_RETRIES})"
            )
            await asyncio.sleep(delay)
        return res

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
                res = await self._get_with_retry(client, url, params=params)
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
                    res = await self._get_with_retry(client, url)
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
        # NOTE: params must carry ALL query args, including `filter`/`per-page`
        # -- passing a `params=` dict to httpx alongside a URL that already
        # has its own embedded "?filter=...&per-page=..." query string
        # silently DISCARDS that embedded query entirely (confirmed live: this
        # was sending a completely unfiltered request to /works, returning
        # the same fixed ~25-paper slice of the whole OpenAlex corpus
        # regardless of work_id -- verified by requesting "related to
        # Attention Is All You Need" and getting back papers on radiation-
        # resistant cameras and RNA-seq analysis). This was the dominant root
        # cause of "irrelevant candidates" appearing for every single
        # discovery run all session, independent of any screening/critique
        # logic downstream.
        params = {"filter": f"related_to:{clean_id}", "per-page": min(limit, 100)}
        if self.api_key:
            params["api_key"] = self.api_key

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await self._get_with_retry(client, "https://api.openalex.org/works", params=params)
                if res.status_code == 200:
                    results = res.json().get("results", [])
                    return [self._parse_work(w) for w in results]
                return []
        except Exception as exc:
            logger.warning(f"OpenAlex find_related_candidates error: {exc}")
            return []

    async def search_candidates(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Plain-text search. NOTE: OpenAlex's `search=` parameter is an
        AND-of-all-terms full-text filter, not fuzzy relevance ranking
        (confirmed via its own `x_query.oql` debug field: "works where full
        text has (all these terms)"). A multi-word query that includes even
        one narrow/rare term (e.g. a specific method name like "SuSiE-RSS" or
        "TWAS") can drive the match count to exactly zero -- and, confirmed
        live, OpenAlex then silently returns an unrelated "trending/popular"
        fallback result set instead of an empty one, which looks
        indistinguishable from a real (bad) match unless the count is
        checked. If the initial query matches nothing, progressively drop
        trailing terms and retry rather than accept that fallback.
        """
        terms = query.split()
        for attempt in range(3):
            if not terms:
                return []

            result = await self._search_once(" ".join(terms), limit)
            if result is not None:
                return result

            # Zero real matches at this specificity -- drop the last
            # (least essential, in extract_search_keywords' extraction
            # order) term(s) and try a broader query instead of accepting
            # OpenAlex's fallback result set for a zero-match query.
            shortened = terms[:-3] if len(terms) > 3 else terms[:-1]
            if len(shortened) == len(terms):
                break
            terms = shortened

        return []

    async def _search_once(self, query: str, limit: int) -> Optional[List[Dict[str, Any]]]:
        """Runs a single search query. Returns None (not an empty list) when
        OpenAlex reports zero real matches, so the caller can distinguish
        "genuinely no results for this broader query" from "this specific
        query matched nothing and should be relaxed."
        """
        await self._rate_limit()
        # See the identical note in find_related_candidates: params must
        # carry the `search`/`per-page` args directly, not be embedded in the
        # URL string alongside a separate params= dict.
        params = {"search": query, "per-page": min(limit, 50)}
        if self.api_key:
            params["api_key"] = self.api_key

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await self._get_with_retry(client, "https://api.openalex.org/works", params=params)
                if res.status_code == 200:
                    data = res.json()
                    if data.get("meta", {}).get("count", 0) == 0:
                        return None
                    results = data.get("results", [])
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
