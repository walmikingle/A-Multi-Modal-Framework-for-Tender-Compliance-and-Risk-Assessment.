import logging
from datetime import datetime
from pathlib import Path

from .config import DATA_DIR, PARSER


def get_logger(
    name="rag"
):
    """
    Create a unique log file for each application run.

    Log directory:
        data/logs/

    Example:
        rag_docling_20260902_192259.log
    """

    log_dir = (
        Path(DATA_DIR)
        / "logs"
    )

    log_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    logger = logging.getLogger(
        name
    )

    # Prevent duplicate handlers.
    if logger.handlers:
        return logger

    logger.setLevel(
        logging.INFO
    )

    # -------------------------------------------------
    # Generate meaningful unique filename
    # -------------------------------------------------

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    parser_name = (
        str(PARSER)
        .strip()
        .lower()
    )

    log_filename = (
        f"rag_{parser_name}_{timestamp}.log"
    )

    log_path = (
        log_dir
        / log_filename
    )

    # -------------------------------------------------
    # Log format
    # -------------------------------------------------

    formatter = logging.Formatter(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    )

    # -------------------------------------------------
    # File handler
    # -------------------------------------------------

    file_handler = logging.FileHandler(
        log_path,
        encoding="utf-8"
    )

    file_handler.setLevel(
        logging.INFO
    )

    file_handler.setFormatter(
        formatter
    )

    # -------------------------------------------------
    # Console handler
    # -------------------------------------------------

    console_handler = (
        logging.StreamHandler()
    )

    console_handler.setLevel(
        logging.INFO
    )

    console_handler.setFormatter(
        formatter
    )

    # -------------------------------------------------
    # Register handlers
    # -------------------------------------------------

    logger.addHandler(
        file_handler
    )

    logger.addHandler(
        console_handler
    )

    # Prevent messages from being
    # duplicated by the root logger.
    logger.propagate = False

    # Useful startup entry.
    logger.info(
        "Log file created | "
        f"File={log_filename} | "
        f"Parser={parser_name}"
    )

    return logger


logger = get_logger()