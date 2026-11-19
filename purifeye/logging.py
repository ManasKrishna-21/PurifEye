"""Structured logging for PurifEye."""

import logging
import sys
from pathlib import Path


def setup_logger(name: str = "purifeye", log_file: Path | str | None = None, level: int = logging.INFO) -> logging.Logger:
    """Configure a standard logger with clean formatting."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        if log_file:
            path = Path(log_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(str(path), encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)

    return logger
