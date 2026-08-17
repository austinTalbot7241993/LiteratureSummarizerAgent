import httpx
from pathlib import Path
from typing import Optional
from lea.exceptions import IngestError
from lea.logging import logger

class GrobidClient:
    def __init__(self, base_url: str = "http://localhost:8070", timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def is_alive(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                res = await client.get(f"{self.base_url}/api/isalive")
                return res.status_code == 200
        except Exception:
            return False

    async def process_fulltext(self, pdf_path: str) -> str:
        url = f"{self.base_url}/api/processFulltextDocument"
        path = Path(pdf_path)
        if not path.exists():
            raise IngestError(f"PDF file not found at {pdf_path}")

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                with open(path, "rb") as f:
                    files = {"input": (path.name, f, "application/pdf")}
                    data = {
                        "generateIDs": "1",
                        # Header consolidation cross-checks the extracted title/
                        # author against CrossRef and fills in a "matched" DOI.
                        # For anonymized/preprint submissions with no real author
                        # name for CrossRef to match against, this can (and, on
                        # a real anonymized preprint, did) confidently return a
                        # completely unrelated paper's DOI -- which then corrupts
                        # the input paper's identity for the entire downstream
                        # discovery pipeline (it starts searching that OTHER
                        # paper's citation neighborhood instead). Reference-level
                        # consolidation is left on: a wrong match on one citation
                        # among many is low-impact, unlike corrupting the seed
                        # paper's own identity.
                        "consolidateHeader": "0",
                        "consolidateCitations": "1"
                    }
                    res = await client.post(url, files=files, data=data)
                    if res.status_code != 200:
                        raise IngestError(f"GROBID returned status {res.status_code}: {res.text[:200]}")
                    return res.text
        except httpx.RequestError as exc:
            logger.warning(f"GROBID connection error: {exc}")
            raise IngestError(f"GROBID client failed to connect to {url}: {exc}")
        except Exception as exc:
            raise IngestError(f"GROBID processing failed: {exc}")

    async def process_references(self, pdf_path: str) -> str:
        url = f"{self.base_url}/api/processReferences"
        path = Path(pdf_path)
        if not path.exists():
            raise IngestError(f"PDF file not found at {pdf_path}")

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                with open(path, "rb") as f:
                    files = {"input": (path.name, f, "application/pdf")}
                    data = {"consolidateCitations": "1"}
                    res = await client.post(url, files=files, data=data)
                    if res.status_code != 200:
                        raise IngestError(f"GROBID processReferences returned status {res.status_code}")
                    return res.text
        except Exception as exc:
            raise IngestError(f"GROBID processReferences failed: {exc}")
