import pytest
from lea.bibliography.bibtex import generate_bibtex, generate_bibtex_key

def test_generate_bibtex_key():
    meta = {
        "title": "Attention Is All You Need",
        "authors": ["Ashish Vaswani", "Noam Shazeer"],
        "publication_year": 2017
    }
    key = generate_bibtex_key(meta)
    assert key == "vaswani2017attention"

def test_generate_bibtex_string():
    meta = {
        "title": "Attention Is All You Need",
        "authors": ["Ashish Vaswani", "Noam Shazeer"],
        "publication_year": 2017,
        "venue": "NeurIPS",
        "arxiv_id": "1706.03762"
    }
    bibtex = generate_bibtex(meta)
    assert "@article{vaswani2017attention," in bibtex
    assert "author = {Ashish Vaswani and Noam Shazeer}" in bibtex
    assert "journal = {NeurIPS}" in bibtex
