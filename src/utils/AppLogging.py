"""
Centralized logging configuration for ConvuyerBreadBagCounterSystem.

Production-grade logging with:
- Size-based rotation (RotatingFileHandler) with configurable max size
- Automatic backup rotation with configurable backup count
- Age-based retention that cleans up old rotated/compressed logs
- Separate error log file for quick issue triage
- Compressed (.gz) archived rotated logs to save disk space
- Thread-safe log cleanup
- Consistent formatters across all handlers

Track event details are stored in the database (track_event_details table),
so structured JSON logging is not duplicated here.
"""

import gzip
import logging
import logging.handlers
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict

# ============================================================================
# Logging Configuration Constants
# ============================================================================

LOG_DIR: str = "data/logs"

# Retention
LOG_RETENTION_DAYS: int = 7  # Days to keep rotated/compressed logs

# Rotation – size-based
LOG_MAX_BYTES: int = 20 * 1024 * 1024  # 20 MB per log file before rotation
LOG_BACKUP_COUNT: int = 10              # Keep up to 10 rotated backups per log type

# Error log rotation
ERROR_LOG_MAX_BYTES: int = 5 * 1024 * 1024  # 5 MB per error log
ERROR_LOG_BACKUP_COUNT: int = 3              # Keep up to 3 error log backups

# Formatter strings
_DETAILED_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
_CONSOLE_FMT = "%(asctime)s | %(levelname)-8s | %(message)s"
_DETAILED_DATEFMT = "%Y-%m-%d %H:%M:%S"
_CONSOLE_DATEFMT = "%H:%M:%S"


# ============================================================================
# Custom namer / rotator for gzip compression of rotated logs
# ============================================================================

def _namer(name: str) -> str:
    """Append .gz to rotated log filenames so they compress on rotation."""
    return name + ".gz"


def _rotator(source: str, dest: str) -> None:
    """Compress rotated log file with gzip."""
    try:
        with open(source, "rb") as f_in:
            with gzip.open(dest, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)  # type: ignore[arg-type]
        os.remove(source)
    except Exception:
        # Fallback: plain rename if compression fails
        try:
            os.rename(source, dest)
        except Exception:
            pass


# ============================================================================
# Setup
# ============================================================================

def setup_logging(
    log_dir: str = LOG_DIR,
    log_max_bytes: int = LOG_MAX_BYTES,
    log_backup_count: int = LOG_BACKUP_COUNT,
    error_log_max_bytes: int = ERROR_LOG_MAX_BYTES,
    error_log_backup_count: int = ERROR_LOG_BACKUP_COUNT,
    retention_days: int = LOG_RETENTION_DAYS,
    console_level: int = logging.INFO,
) -> logging.Logger:
    """
    Setup production-grade application logging.

    Features:
        - **Rotating main log** – rolls over when *log_max_bytes* is exceeded,
          keeping up to *log_backup_count* compressed (.gz) backups.
        - **Rotating error log** – captures WARNING+ messages separately for
          quick triage.
        - **Console handler** – coloured-by-level output to stdout.
        - **Old-log cleanup** – removes rotated/compressed files older than
          *retention_days* on every startup.

    Args:
        log_dir: Directory for log files.
        log_max_bytes: Max bytes per main log file before rotation.
        log_backup_count: Number of rotated main-log backups to keep.
        error_log_max_bytes: Max bytes per error log file before rotation.
        error_log_backup_count: Number of rotated error-log backups to keep.
        retention_days: Days to keep old (rotated) log files.
        console_level: Minimum level for console output.

    Returns:
        Configured root application logger.
    """
    # Ensure log directory exists
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Proactive cleanup of aged log artefacts
    _cleanup_old_logs(log_dir, retention_days=retention_days)

    # Logger setup
    app_logger = logging.getLogger("ConvuyerBreadBagCounter")
    app_logger.setLevel(logging.DEBUG)

    # Prevent duplicate handlers on re-import / hot-reload
    if app_logger.handlers:
        return app_logger

    # --- Formatters ---
    detailed_formatter = logging.Formatter(_DETAILED_FMT, datefmt=_DETAILED_DATEFMT)
    console_formatter = logging.Formatter(_CONSOLE_FMT, datefmt=_CONSOLE_DATEFMT)

    # --- 1. Rotating main log (DEBUG+) ---
    main_log_path = os.path.join(log_dir, "convuyer_counter.log")
    main_handler = logging.handlers.RotatingFileHandler(
        main_log_path,
        maxBytes=log_max_bytes,
        backupCount=log_backup_count,
        encoding="utf-8",
    )
    main_handler.setLevel(logging.DEBUG)
    main_handler.setFormatter(detailed_formatter)
    main_handler.namer = _namer
    main_handler.rotator = _rotator
    app_logger.addHandler(main_handler)

    # --- 2. Rotating error log (WARNING+) ---
    error_log_path = os.path.join(log_dir, "convuyer_counter_error.log")
    error_handler = logging.handlers.RotatingFileHandler(
        error_log_path,
        maxBytes=error_log_max_bytes,
        backupCount=error_log_backup_count,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.WARNING)
    error_handler.setFormatter(detailed_formatter)
    error_handler.namer = _namer
    error_handler.rotator = _rotator
    app_logger.addHandler(error_handler)

    # --- 3. Console handler (INFO+ by default) ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)
    app_logger.addHandler(console_handler)

    # Startup banner
    app_logger.info(
        "[Logging] Initialised — main=%s (max %s MB × %d backups), "
        "error=%s (max %s MB × %d backups), retention=%d days",
        main_log_path,
        round(log_max_bytes / (1024 * 1024), 1),
        log_backup_count,
        error_log_path,
        round(error_log_max_bytes / (1024 * 1024), 1),
        error_log_backup_count,
        retention_days,
    )

    return app_logger


# ============================================================================
# Retention / cleanup
# ============================================================================

def _cleanup_old_logs(log_dir: str, retention_days: int = LOG_RETENTION_DAYS) -> None:
    """
    Delete log files (including compressed rotated backups) older than
    *retention_days*.

    Scans for:
        - ``convuyer_counter*.log``
        - ``convuyer_counter*.log.*``  (rotated backups)
        - ``convuyer_counter*.gz``     (compressed backups)

    This runs at startup before the logger is created, so any diagnostic
    output uses ``print()``.

    Args:
        log_dir: Directory containing log files.
        retention_days: Number of days to keep files.
    """
    cutoff = time.time() - (retention_days * 86400)
    deleted = 0
    patterns = [
        "convuyer_counter*.log",
        "convuyer_counter*.log.*",
        "convuyer_counter*.gz",
    ]
    try:
        log_path = Path(log_dir)
        for pattern in patterns:
            for log_file in log_path.glob(pattern):
                try:
                    if log_file.stat().st_mtime < cutoff:
                        log_file.unlink()
                        deleted += 1
                except OSError:
                    pass  # File may have been removed concurrently
    except Exception:
        pass  # Never fail startup over log cleanup

    if deleted > 0:
        print(f"[LogRetention] Deleted {deleted} log file(s) older than {retention_days} days")


# ============================================================================
# Global logger instance
# ============================================================================

logger = setup_logging()


# ============================================================================
# Utility helpers
# ============================================================================

def get_log_file_paths() -> Dict[str, str]:
    """
    Get paths to the current (active) log files.

    Returns:
        Dictionary with keys ``main_log`` and optionally ``error_log``
        pointing to the active log file paths.
    """
    log_dir = LOG_DIR
    if not os.path.exists(log_dir):
        return {}

    result: Dict[str, str] = {}

    main_log = os.path.join(log_dir, "convuyer_counter.log")
    if os.path.isfile(main_log):
        result["main_log"] = main_log

    error_log = os.path.join(log_dir, "convuyer_counter_error.log")
    if os.path.isfile(error_log):
        result["error_log"] = error_log

    return result


def reconfigure_console_level(level: int = logging.INFO) -> None:
    """
    Change the console handler log level at runtime.

    Useful for temporarily enabling DEBUG output on the console for
    live troubleshooting without restarting the application.

    Args:
        level: New logging level for the console handler.
    """
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler, logging.FileHandler
        ):
            handler.setLevel(level)
            logger.info("[Logging] Console level changed to %s", logging.getLevelName(level))
            break
