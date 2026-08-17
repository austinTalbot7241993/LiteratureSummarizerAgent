import asyncio
import pytest
from pathlib import Path
from lea.ingester.grobid_client import GrobidClient
from lea.ingester.tei_parser import TEIParser


async def test_process_fulltext_disables_header_consolidation(monkeypatch, tmp_path):
    """Regression test: GROBID's consolidateHeader=1 cross-checks the
    extracted title/author against CrossRef and fills in a "matched" DOI.
    For an anonymized/preprint submission with no real author name for
    CrossRef to match against, this can (and, confirmed live against a real
    anonymized preprint, did) confidently return a completely unrelated
    paper's DOI -- corrupting the input paper's identity for the entire
    downstream discovery pipeline, which then searches that OTHER paper's
    citation neighborhood instead of the seed paper's. consolidateCitations
    (reference-level matching) stays on since a wrong match there is
    low-impact by comparison.
    """
    captured = {}

    class FakeResponse:
        status_code = 200
        text = "<TEI/>"

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, files=None, data=None):
            captured["data"] = data
            return FakeResponse()

    monkeypatch.setattr("lea.ingester.grobid_client.httpx.AsyncClient", FakeAsyncClient)

    dummy_pdf = tmp_path / "dummy.pdf"
    dummy_pdf.write_bytes(b"%PDF-1.4 fake content")

    client = GrobidClient()
    await client.process_fulltext(str(dummy_pdf))

    assert captured["data"]["consolidateHeader"] == "0"
    assert captured["data"]["consolidateCitations"] == "1"

def test_grobid_tei_parser_fixture():
    fixture_path = Path(__file__).resolve().parent.parent / "fixtures" / "sample_grobid.tei.xml"
    with open(fixture_path, "r", encoding="utf-8") as f:
        xml_content = f.read()

    parser = TEIParser(xml_content)
    header = parser.parse_header()
    assert header["title"] == "Sample Academic Paper on Neural Attention"
    assert "Alice Smith" in header["authors"]
    assert header["doi"] == "10.1234/sample.2023.001"

    bib = parser.parse_bibliography()
    assert len(bib) == 2
    assert bib[0]["arxiv_id"] == "1706.03762"
    assert bib[1]["doi"] == "10.1109/CVPR.2016.90"
