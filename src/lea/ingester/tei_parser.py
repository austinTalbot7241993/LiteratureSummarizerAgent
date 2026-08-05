from typing import Dict, Any, List, Optional
from bs4 import BeautifulSoup
from lea.logging import logger

class TEIParser:
    def __init__(self, xml_content: str):
        self.soup = BeautifulSoup(xml_content, "xml")

    def parse_header(self) -> Dict[str, Any]:
        title_node = self.soup.find("title", type="main") or self.soup.find("title")
        title = title_node.get_text(strip=True) if title_node else "Untitled"

        abstract_node = self.soup.find("abstract")
        abstract = abstract_node.get_text(" ", strip=True) if abstract_node else ""

        authors = []
        for author_node in self.soup.find_all("author"):
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
        idno_doi = self.soup.find("idno", type="DOI") or self.soup.find("idno", type="doi")
        if idno_doi:
            doi = idno_doi.get_text(strip=True)

        arxiv_id = None
        idno_arxiv = self.soup.find("idno", type="arXiv") or self.soup.find("idno", type="arxiv")
        if idno_arxiv:
            arxiv_id = idno_arxiv.get_text(strip=True)

        year = None
        date_node = self.soup.find("date", type="published") or self.soup.find("date")
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
