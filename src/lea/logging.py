import logging
import sys
from rich.logging import RichHandler

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
        force=True
    )
    logger = logging.getLogger("lea")
    logger.setLevel(numeric_level)
    return logger

logger = logging.getLogger("lea")
