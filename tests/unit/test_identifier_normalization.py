import pytest
from lea.resolution.identifiers import (
    normalize_doi, normalize_arxiv, normalize_openalex_id, normalize_s2_id, normalize_title
)

def test_normalize_doi():
    assert normalize_doi("https://doi.org/10.1234/ABC.123") == "10.1234/abc.123"
    assert normalize_doi("DOI:10.1000/182") == "10.1000/182"
    assert normalize_doi(None) is None

def test_normalize_arxiv():
    assert normalize_arxiv("arXiv:2301.12345v2") == "2301.12345"
    assert normalize_arxiv("https://arxiv.org/abs/2106.00001") == "2106.00001"
    assert normalize_arxiv(None) is None

def test_normalize_openalex_id():
    assert normalize_openalex_id("https://openalex.org/w2741809807") == "W2741809807"
    assert normalize_openalex_id("W12345") == "W12345"

def test_normalize_title():
    assert normalize_title("Attention Is All You Need!") == "attention is all you need"
    assert normalize_title("Deep  Residual   Learning...") == "deep residual learning"
