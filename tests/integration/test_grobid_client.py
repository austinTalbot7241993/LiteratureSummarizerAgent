import asyncio
import pytest
from pathlib import Path
from lea.ingester.grobid_client import GrobidClient
from lea.ingester.tei_parser import TEIParser

def test_grobid_tei_parser_fixture():
    fixture_path = Path("tests/fixtures/sample_grobid.tei.xml")
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
