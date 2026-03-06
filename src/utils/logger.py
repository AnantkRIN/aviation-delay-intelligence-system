"""Structured logging for the aviation operations system."""

from __future__ import annotations

import logging
import pathlib
from typing import Optional

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
LOGS_DIR = PROJECT_ROOT / "logs"


def setup_logging(
    log_file: Optional[str] = "simulation.log",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Configure structured logging to file and console.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / log_file if log_file else None

    logger = logging.getLogger("aviation_ops")
    logger.setLevel(level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(levelname)s] %(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if log_path:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def get_logger(name: str = "aviation_ops") -> logging.Logger:
    """Get or create the aviation operations logger."""
    return logging.getLogger(name)
