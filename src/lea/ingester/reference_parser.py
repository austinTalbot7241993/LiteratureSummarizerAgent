from typing import Dict, Any, List, Tuple, Optional
from lea.ingester.grobid_client import GrobidClient
from lea.ingester.tei_parser import TEIParser
from lea.ingester.pdf_parser import PDFParser
from lea.exceptions import IngestError, BibliographyExtractionError
from lea.logging import logger

class ReferenceParser:
    def __init__(self, grobid_url: str = "http://localhost:8070", config: Any = None, grobid_timeout: float = 120.0):
        self.grobid_client = GrobidClient(base_url=grobid_url, timeout=grobid_timeout)
        self.config = config

    async def extract_references(
        self,
        pdf_path: str,
        doi: Optional[str] = None,
        arxiv_id: Optional[str] = None,
        openalex_client: Any = None
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Extracts references using multi-stage fallback:
        Stage 1: GROBID microservice
        Stage 2: PyMuPDF + RegEx extraction
        Stage 3: OpenAlex Reference API
        Returns tuple of (references_list, status_string: 'complete' | 'incomplete').
        """
        references = []
        status = "incomplete"

        # Stage 1: GROBID
        try:
            if await self.grobid_client.is_alive():
                logger.info("Attempting GROBID reference extraction...")
                tei_xml = await self.grobid_client.process_fulltext(pdf_path)
                parser = TEIParser(tei_xml)
                grobid_refs = parser.parse_bibliography()
                if grobid_refs:
                    logger.info(f"GROBID extracted {len(grobid_refs)} references.")
                    return grobid_refs, "complete"
        except Exception as exc:
            logger.warning(f"GROBID reference extraction failed/skipped: {exc}")

        # Stage 2: PyMuPDF + RegEx
        try:
            logger.info("Attempting PyMuPDF RegEx reference extraction...")
            pdf_parser = PDFParser(pdf_path)
            regex_refs = pdf_parser.extract_regex_references()
            if regex_refs:
                logger.info(f"PyMuPDF extracted {len(regex_refs)} reference identifier matches.")
                references.extend(regex_refs)
                status = "complete"
        except Exception as exc:
            logger.warning(f"PyMuPDF reference extraction failed: {exc}")

        # Stage 3: OpenAlex Reference API
        if openalex_client and (doi or arxiv_id):
            try:
                logger.info("Attempting OpenAlex Reference API fallback...")
                oa_refs = await openalex_client.fetch_work_references(doi=doi, arxiv_id=arxiv_id)
                if oa_refs:
                    logger.info(f"OpenAlex API extracted {len(oa_refs)} references.")
                    references.extend(oa_refs)
                    status = "complete"
            except Exception as exc:
                logger.warning(f"OpenAlex reference fallback failed: {exc}")

        if not references:
            status = "incomplete"

        return references, status
