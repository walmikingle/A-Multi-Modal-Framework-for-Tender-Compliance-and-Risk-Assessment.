import logging
from datetime import datetime
from pathlib import Path

from .config import DATA_DIR, PARSER


def get_logger(
    name="rag"
):
    """
    Create a unique log file for each application run.

    Detailed INFO/DEBUG logs are written to the log file.
    Only WARNING/ERROR/CRITICAL messages are shown in the console.

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
        logging.DEBUG
    )

    # -------------------------------------------------
    # Generate unique log filename
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
    # Keep detailed technical information in the file.

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
    # Only show important warnings/errors in terminal.

    console_handler = (
        logging.StreamHandler()
    )

    console_handler.setLevel(
        logging.WARNING
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

    # Prevent duplicate output through root logger.
    logger.propagate = False

    # Startup information goes to the log file,
    # but will NOT appear in the manager-facing terminal.
    logger.info(
        "Log file created | "
        f"File={log_filename} | "
        f"Parser={parser_name}"
    )

    return logger


logger = get_logger()