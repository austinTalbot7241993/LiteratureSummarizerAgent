import pytest
# pyrefly: ignore [missing-import]
from lea.resolution.matcher import exact_id_match, fuzzy_title_match, is_same_paper

def test_exact_id_match():
    p1 = {"doi": "10.1038/s41586-020-2649-2"}
    p2 = {"doi": "https://doi.org/10.1038/s41586-020-2649-2"}
    assert exact_id_match(p1, p2) is True

    p3 = {"arxiv_id": "2301.12345v1"}
    p4 = {"arxiv_id": "2301.12345"}
    assert exact_id_match(p3, p4) is True

def test_fuzzy_title_match():
    m1 = {"title": "Attention Is All You Need", "year": 2017}
    m2 = {"title": "Attention Is All You Need!", "year": 2017}
    assert fuzzy_title_match(m1, m2, similarity_threshold=0.96) is True

    m3 = {"title": "Attention Is All You Need", "year": 2017}
    m4 = {"title": "Attention Is All You Need", "year": 2022}
    assert fuzzy_title_match(m3, m4, year_tolerance=1) is False

def test_is_same_paper():
    p1 = {"title": "Generative Adversarial Nets", "arxiv_id": "1406.2661"}
    p2 = {"title": "GANs paper", "arxiv_id": "1406.2661"}
    assert is_same_paper(p1, p2) is True
