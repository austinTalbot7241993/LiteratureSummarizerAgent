from typing import Dict, Any, List, Optional
from bs4 import BeautifulSoup
from lea.logging import logger

class TEIParser:
    def __init__(self, xml_content: str):
        self.soup = BeautifulSoup(xml_content, "xml")

    def parse_header(self) -> Dict[str, Any]:
        # Scope every lookup to <teiHeader> specifically, never the whole
        # document. GROBID emits the paper's own metadata inside <teiHeader>
        # and everything it cites inside <text><back><listBibl> -- searching
        # the whole soup for the first matching tag conflates the two. This
        # is not hypothetical: for a paper with no DOI of its own (e.g. an
        # unpublished arXiv preprint), self.soup.find("idno", type="DOI")
        # returns the FIRST bibliography entry's DOI instead of None,
        # silently misattributing a cited work's identity as the paper's
        # own (confirmed live: <teiHeader> genuinely has no DOI idno here,
        # but <listBibl><biblStruct xml:id="b0"> -- reference #1 -- does).
        # Falls back to the whole soup only if <teiHeader> is somehow
        # missing, which should not happen for valid GROBID TEI output.
        header = self.soup.find("teiHeader") or self.soup

        title_node = header.find("title", type="main") or header.find("title")
        title = title_node.get_text(strip=True) if title_node else "Untitled"

        abstract_node = header.find("abstract")
        abstract = abstract_node.get_text(" ", strip=True) if abstract_node else ""

        authors = []
        for author_node in header.find_all("author"):
            persname = author_node.find("persName")
            if persname:
                first = persname.find("forename", type="first")
                surname = persname.find("surname")
                name_parts = []
                if first and first.text:
                    name_parts.append(first.text.strip())
                if surname and surname.text:
                    name_parts.append(surname.text.strip())
                if name_parts:
                    authors.append(" ".join(name_parts))

        doi = None
        idno_doi = header.find("idno", type="DOI") or header.find("idno", type="doi")
        if idno_doi:
            doi = idno_doi.get_text(strip=True)

        arxiv_id = None
        idno_arxiv = header.find("idno", type="arXiv") or header.find("idno", type="arxiv")
        if idno_arxiv:
            arxiv_id = idno_arxiv.get_text(strip=True)

        year = None
        date_node = header.find("date", type="published") or header.find("date")
        if date_node and date_node.has_attr("when"):
            try:
                year = int(date_node["when"][:4])
            except ValueError:
                pass

        return {
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "doi": doi,
            "arxiv_id": arxiv_id,
            "publication_year": year
        }

    def parse_bibliography(self) -> List[Dict[str, Any]]:
        entries = []
        container = self.soup.find("listBibl") or self.soup.find("back") or self.soup
        for struct in container.find_all("biblStruct"):
            ref_title_node = struct.find("title", level="a") or struct.find("title", level="m")
            ref_title = ref_title_node.get_text(strip=True) if ref_title_node else None

            ref_authors = []
            for author_node in struct.find_all("author"):
                persname = author_node.find("persName")
                if persname:
                    first = persname.find("forename", type="first")
                    surname = persname.find("surname")
                    name_parts = []
                    if first and first.text:
                        name_parts.append(first.text.strip())
                    if surname and surname.text:
                        name_parts.append(surname.text.strip())
                    if name_parts:
                        ref_authors.append(" ".join(name_parts))

            doi = None
            idno_doi = struct.find("idno", type="DOI") or struct.find("idno", type="doi")
            if idno_doi:
                doi = idno_doi.get_text(strip=True)

            arxiv_id = None
            idno_arxiv = struct.find("idno", type="arXiv") or struct.find("idno", type="arxiv")
            if idno_arxiv:
                arxiv_id = idno_arxiv.get_text(strip=True)

            year = None
            date_node = struct.find("date")
            if date_node and date_node.has_attr("when"):
                try:
                    year = int(date_node["when"][:4])
                except ValueError:
                    pass

            raw_text = struct.get_text(" ", strip=True)

            entries.append({
                "raw_citation": raw_text,
                "title": ref_title,
                "authors": ref_authors,
                "doi": doi,
                "arxiv_id": arxiv_id,
                "year": year,
                "extraction_method": "grobid"
            })
        return entries
