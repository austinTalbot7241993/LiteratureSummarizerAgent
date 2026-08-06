import httpx
from pathlib import Path
from typing import Optional, Dict, Any
from lea.acquisition.open_access import OpenAccessResolver
from lea.acquisition.validation import validate_pdf_bytes
from lea.exceptions import AcquisitionError
from lea.logging import logger

class PDFDownloader:
    def __init__(self, cache_dir: str = ".cache/lea/pdfs", max_pdf_bytes: int = 104857600, user_agent: str = "LEA/0.1 scholarly-research-agent", unpaywall_email: Optional[str] = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_pdf_bytes = max_pdf_bytes
        self.user_agent = user_agent
        self.oa_resolver = OpenAccessResolver(unpaywall_email=unpaywall_email)
        if unpaywall_email:
            logger.info(f"Unpaywall resolution enabled with email: {unpaywall_email}")
        else:
            logger.warning("Unpaywall email not set — OA resolution limited to oa_pdf_url and arXiv IDs only.")

    async def download_candidate_pdf(self, candidate_meta: Dict[str, Any]) -> Optional[str]:
        oa_url = await self.oa_resolver.resolve_oa_url(candidate_meta)
        if not oa_url:
            logger.info(f"No open-access PDF URL resolved for candidate: {candidate_meta.get('title')}")
            return None

        # Filename based on DOI / arXiv / S2 ID or title hash
        file_id = candidate_meta.get("arxiv_id") or candidate_meta.get("doi") or candidate_meta.get("s2_id") or "cand"
        safe_name = "".join(c if c.isalnum() else "_" for c in str(file_id))[:50] + ".pdf"
        target_path = self.cache_dir / safe_name

        if target_path.exists():
            logger.info(f"Using cached PDF at {target_path}")
            return str(target_path)

        logger.info(f"Downloading OA PDF from {oa_url}...")
        headers = {"User-Agent": self.user_agent}

        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                res = await client.get(oa_url, headers=headers)
                if res.status_code != 200:
                    logger.warning(f"Download failed with status {res.status_code} for {oa_url}")
                    return None

                content_type = res.headers.get("content-type")
                validate_pdf_bytes(res.content, content_type=content_type, max_bytes=self.max_pdf_bytes)

                with open(target_path, "wb") as f:
                    f.write(res.content)

                logger.info(f"Successfully saved PDF to {target_path}")
                return str(target_path)
        except Exception as exc:
            logger.warning(f"Failed to download or validate PDF from {oa_url}: {exc}")
            return None
