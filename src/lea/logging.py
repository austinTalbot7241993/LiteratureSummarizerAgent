import os
import logging
import sys
from rich.logging import RichHandler

def setup_logging(log_level: str = "INFO", log_file: str = "lea_run.log") -> logging.Logger:
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(file_formatter)

    console_handler = RichHandler(rich_tracebacks=True, show_path=False)
    console_handler.setLevel(numeric_level)

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    root_logger.handlers = [console_handler, file_handler]

    logger = logging.getLogger("lea")
    logger.setLevel(numeric_level)
    logger.info(f"Logging initialized. Writing execution logs to: {os.path.abspath(log_file)}")
    return logger

logger = logging.getLogger("lea")
