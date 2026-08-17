import pytest
from pathlib import Path
from lea.ingester.pdf_parser import PDFParser

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_fallback_title_merges_wrapped_title_lines():
    """Regression test for a real production bug: a real user-provided PDF
    (HapSpin.pdf) had a title wrapping across two lines with no blank-line
    separator ("...Estimation for" / "Low-Frequency Variants..."), and the
    naive `lines[0]`-only extraction truncated it mid-phrase. A truncated,
    grammatically incomplete title fed into OpenAlex's title search returned
    generic unrelated "hub" papers instead of real neighbors -- an empty,
    useless literature search traced all the way back to this one-line
    title bug. `tests/fixtures/sample_real_paper.pdf` has the same real-world
    shape (a running-header banner, then a 3-line wrapped title, then
    author/affiliation lines), so it's used here instead of a synthetic
    fixture.
    """
    parser = PDFParser(str(FIXTURES / "sample_real_paper.pdf"))
    meta = parser.parse_fallback_metadata()
    title = meta["title"]
    assert "Combining Bayesian and Frequentist Inference" in title
    assert "Copy" in title  # the continuation line must not be dropped


def test_fallback_title_skips_running_header_banner():
    parser = PDFParser(str(FIXTURES / "sample_real_paper.pdf"))
    meta = parser.parse_fallback_metadata()
    title = meta["title"]
    assert "Preprint" not in title
    assert "arXiv:" not in title


def test_fallback_title_does_not_include_author_or_affiliation_lines():
    parser = PDFParser(str(FIXTURES / "sample_real_paper.pdf"))
    meta = parser.parse_fallback_metadata()
    title = meta["title"]
    assert "Talbot" not in title
    assert "Pillar" not in title


def test_fallback_abstract_extracts_real_abstract_not_title_page_noise():
    """Regression test: the old fallback abstract was `text[:1000]` -- the
    first 1000 characters of the WHOLE document, mixing the title, authors,
    and affiliation lines in with (or instead of) the actual abstract. This
    fed a garbled, noisy "abstract" into abstract screening and (after the
    target_abstract fix) into summarization/critique prompts for the seed
    paper.
    """
    parser = PDFParser(str(FIXTURES / "sample_real_paper.pdf"))
    meta = parser.parse_fallback_metadata()
    abstract = meta["abstract"]
    assert "amplicon panels" in abstract or "CNV" in abstract
    assert "Combining Bayesian" not in abstract
    assert "Talbot" not in abstract


def test_fallback_metadata_on_hapspin_pdf_if_available():
    """Direct regression check against the exact PDF that surfaced this bug,
    when available locally (it's a user-provided file, not a repo fixture,
    so this is skipped in environments that don't have it).
    """
    hapspin_path = Path.home() / "Downloads" / "HapSpin.pdf"
    if not hapspin_path.exists():
        pytest.skip("HapSpin.pdf not present in this environment")

    parser = PDFParser(str(hapspin_path))
    meta = parser.parse_fallback_metadata()
    assert meta["title"] == (
        "HapSpin: Regularized Linkage Disequilibrium Matrix Estimation for "
        "Low-Frequency Variants and Small Reference Panels"
    )
    assert "Linkage disequilibrium" in meta["abstract"]
