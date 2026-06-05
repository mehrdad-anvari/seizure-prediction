"""Logging utilities.

The original repository logged to both console and a run-specific file.
This module provides the same behavior for the new library structure.
"""

from __future__ import annotations

import logging
import os
from typing import Optional


def setup_logging(
    run_dir: str,
    *,
    filename: str = "training.log",
    level: int = logging.INFO,
    logger_name: str = "seizure_pred",
) -> logging.Logger:
    """Configure logging to both console and a file under ``run_dir/logs``.

    This function is idempotent for a given logger_name: it clears existing
    handlers to avoid duplicate log lines when called multiple times.
    """

    log_dir = os.path.join(run_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, filename)

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False

    # Clear handlers to avoid duplicate outputs in notebooks / repeated calls.
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.debug("Logging initialized. log_path=%s", log_path)
    return logger
