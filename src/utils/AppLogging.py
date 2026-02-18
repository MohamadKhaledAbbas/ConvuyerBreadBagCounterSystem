"""
Centralized logging configuration for ConvuyerBreadBagCounterSystem.

Provides standard logging with configurable file retention.
Track event details are stored in the database (track_event_details table),
so structured JSON logging is no longer needed.
"""

import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

# Default log retention in days
LOG_RETENTION_DAYS = 3


def setup_logging(log_dir: str = "data/logs") -> logging.Logger:
    """Setup application logging with file and console handlers."""
    # Create log directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Clean up old log files on startup
    _cleanup_old_logs(log_dir, retention_days=LOG_RETENTION_DAYS)

    # Generate timestamped log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"convuyer_counter_{timestamp}.log")

    # Create logger
    logger = logging.getLogger("ConvuyerBreadBagCounter")
    logger.setLevel(logging.DEBUG)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # File handler - detailed logs
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)

    # Console handler - info and above
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def _cleanup_old_logs(log_dir: str, retention_days: int = 7):
    """
    Delete log files older than retention_days.

    Args:
        log_dir: Directory containing log files
        retention_days: Number of days to keep log files (default: 7)
    """
    cutoff = time.time() - (retention_days * 86400)
    deleted = 0
    try:
        for log_file in Path(log_dir).glob("convuyer_counter_*.log"):
            if log_file.stat().st_mtime < cutoff:
                log_file.unlink()
                deleted += 1
    except Exception:
        pass  # Don't fail startup over log cleanup
    if deleted > 0:
        # Can't use logger here (not created yet), use print
        print(f"[LogRetention] Deleted {deleted} log file(s) older than {retention_days} days")


# Global logger instance
logger = setup_logging()


def get_log_file_paths() -> Dict[str, str]:
    """Get paths to current log files."""
    log_dir = "data/logs"
    if not os.path.exists(log_dir):
        return {}

    files = sorted(Path(log_dir).glob("convuyer_counter_*.log"), reverse=True)
    if files:
        return {"main_log": str(files[0])}
    return {}
