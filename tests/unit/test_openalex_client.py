import pytest
from lea.discovery.openalex import OpenAlexClient


class _FakeResponse:
    def __init__(self, json_data, status_code=200):
        self._json = json_data
        self.status_code = status_code

    def json(self):
        return self._json


class _FakeAsyncClient:
    """Captures the exact (url, params) passed to httpx.AsyncClient.get()."""

    captured_calls = []

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def get(self, url, params=None, headers=None):
        type(self).captured_calls.append({"url": url, "params": params})
        return _FakeResponse({"meta": {"count": 1}, "results": [{"id": "https://openalex.org/W1", "display_name": "Fake Result"}]})


async def test_find_related_candidates_sends_filter_param_not_embedded_in_url(monkeypatch):
    """Regression test for a real, severe production bug: `find_related_candidates`
    built a URL string with an embedded "?filter=related_to:...&per-page=..."
    query, then ALSO passed a separate `params=` dict to httpx.AsyncClient.get().
    httpx silently discards the URL's own embedded query string whenever a
    `params=` kwarg is also supplied -- so every call (with no API key
    configured, the default) sent a completely UNFILTERED request to
    /works, returning some fixed slice of the entire ~324 million-work
    OpenAlex corpus regardless of which paper's citation neighborhood was
    requested. Confirmed live: querying "related to Attention Is All You
    Need" (one of the most-cited papers in computer science) returned
    papers about radiation-resistant camera shielding and RNA-seq analysis
    -- the exact same fixed, irrelevant candidate set that appeared for
    every single discovery run all session, independent of the input paper.
    This was the dominant root cause of "irrelevant candidates" throughout
    the whole investigation, not any screening/critique logic downstream.
    """
    _FakeAsyncClient.captured_calls = []
    monkeypatch.setattr("lea.discovery.openalex.httpx.AsyncClient", _FakeAsyncClient)

    client = OpenAlexClient()
    await client.find_related_candidates("W123456", limit=10)

    assert len(_FakeAsyncClient.captured_calls) == 1
    call = _FakeAsyncClient.captured_calls[0]
    # The filter and per-page args MUST travel via the params dict (the only
    # thing httpx actually sends), never solely embedded in the URL string.
    assert call["url"] == "https://api.openalex.org/works"
    assert call["params"]["filter"] == "related_to:W123456"
    assert call["params"]["per-page"] == 10


async def test_search_candidates_sends_search_param_not_embedded_in_url(monkeypatch):
    """Same bug class, same fix, in search_candidates()."""
    _FakeAsyncClient.captured_calls = []
    monkeypatch.setattr("lea.discovery.openalex.httpx.AsyncClient", _FakeAsyncClient)

    client = OpenAlexClient()
    await client.search_candidates("linkage disequilibrium GWAS", limit=15)

    assert len(_FakeAsyncClient.captured_calls) >= 1
    call = _FakeAsyncClient.captured_calls[0]
    assert call["url"] == "https://api.openalex.org/works"
    assert call["params"]["search"] == "linkage disequilibrium GWAS"
    assert call["params"]["per-page"] == 15


async def test_search_candidates_relaxes_query_on_zero_matches(monkeypatch):
    """Regression test: OpenAlex's `search=` parameter is an AND-of-all-terms
    filter. A query where even one narrow/rare term (e.g. a specific method
    name) drives the real match count to zero must not be silently accepted
    as "no results" -- confirmed live, OpenAlex returns an unrelated
    fallback result set for a zero-match query rather than an empty one.
    search_candidates must detect meta.count == 0 and progressively drop
    trailing terms rather than return that fallback set.
    """
    calls = []

    class ZeroThenRealClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url, params=None, headers=None):
            calls.append(params["search"])
            if len(calls) == 1:
                return _FakeResponse({"meta": {"count": 0}, "results": []})
            return _FakeResponse({
                "meta": {"count": 5000},
                "results": [{"id": "https://openalex.org/W2", "display_name": "Real Match"}]
            })

    monkeypatch.setattr("lea.discovery.openalex.httpx.AsyncClient", ZeroThenRealClient)

    client = OpenAlexClient()
    results = await client.search_candidates("one two three four five six seven", limit=10)

    assert len(calls) == 2
    assert len(calls[1].split()) < len(calls[0].split())
    assert len(results) == 1
    assert results[0]["title"] == "Real Match"
