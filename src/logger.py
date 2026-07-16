import logging
import sys
from pathlib import Path

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File
    fh = logging.FileHandler(LOG_DIR / "forecast.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def get_api_logger(name: str) -> logging.Logger:
    """API-specific logger with rotation — prevents log files growing forever."""
    from logging.handlers import RotatingFileHandler
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    # Rotate at 10MB, keep 5 files
    fh = RotatingFileHandler(
        LOG_DIR / "api_requests.log",
        maxBytes  = 10 * 1024 * 1024,
        backupCount = 5
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger
