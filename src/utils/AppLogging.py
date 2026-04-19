"""
Centralized logging configuration for ConvuyerBreadBagCounterSystem.

Production-grade logging with:
- Size-based rotation (RotatingFileHandler) with configurable max size
- Automatic backup rotation with configurable backup count
- Age-based retention that cleans up old rotated/compressed logs
- Separate error log file for quick issue triage
- Compressed (.gz) archived rotated logs to save disk space
- **Non-blocking** gzip compression via background thread to prevent
  frame pipeline stalls during log rotation
- Thread-safe log cleanup
- Consistent formatters across all handlers

Track event details are stored in the database (track_event_details table),
so structured JSON logging is not duplicated here.

Environment overrides:
- APP_LOG_BASENAME: per-process log file basename (default: convuyer_counter)
- APP_LOG_CONSOLE_STREAM: stdout, stderr, or none (default: stdout)
- APP_LOG_MAX_BYTES: max bytes before main log rotation
- APP_LOG_BACKUP_COUNT: number of rotated main log backups
- APP_ERROR_LOG_MAX_BYTES: max bytes before error log rotation
- APP_ERROR_LOG_BACKUP_COUNT: number of rotated error log backups
- APP_LOG_RETENTION_DAYS: days to retain rotated/compressed logs
"""

import gzip
import logging
import logging.handlers
import os
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Optional

from src.config.paths import LOG_DIR


def _resolve_log_basename() -> str:
    """Return the per-process log basename used for active log files."""
    name = (os.getenv("APP_LOG_BASENAME", "convuyer_counter") or "").strip()
    if not name:
        return "convuyer_counter"
    # Keep filenames predictable and filesystem-safe.
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)


def _resolve_console_stream() -> str:
    """Return stdout, stderr, or none for the console handler."""
    stream = (os.getenv("APP_LOG_CONSOLE_STREAM", "stdout") or "").strip().lower()
    if stream in {"stdout", "stderr", "none"}:
        return stream
    return "stdout"


def _resolve_int_env(name: str, default: int, minimum: int = 0) -> int:
    """Return a validated integer env override, or *default* when invalid."""
    raw_value = (os.getenv(name, "") or "").strip()
    if not raw_value:
        return default
    try:
        parsed_value = int(raw_value)
    except ValueError:
        return default
    if parsed_value < minimum:
        return default
    return parsed_value

# ============================================================================
# Logging Configuration Constants
# ============================================================================

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


def _compress_in_background(source: str, dest: str) -> None:
    """Background worker: gzip-compress *source* into *dest*, then remove *source*.

    Runs in a daemon thread so it never blocks the logging lock.
    """
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


def _rotator(source: str, dest: str) -> None:
    """Move the rotated log aside and compress it **without blocking**.

    The previous implementation did synchronous gzip compression while
    holding Python's logging lock.  On slow embedded storage (eMMC /
    SD card / Horizon RDK) that blocked **every** thread calling
    ``logger.*()`` for 2-4 seconds — causing gaps in frame timestamps
    and lost tracks.

    New approach:
    1. Cheaply ``rename`` *source* → intermediate path (microseconds).
    2. Spawn a daemon thread to gzip the intermediate → *dest*.
    The logging lock is released immediately after the rename.
    """
    # Use an intermediate name so the logging handler doesn't see the
    # file while the background thread is still compressing it.
    intermediate = source + ".rotating"
    try:
        os.rename(source, intermediate)
    except Exception:
        # If even the rename fails, fall back to synchronous path
        try:
            os.rename(source, dest)
        except Exception:
            pass
        return

    t = threading.Thread(
        target=_compress_in_background,
        args=(intermediate, dest),
        name="log-compressor",
        daemon=True,
    )
    t.start()


# ============================================================================
# Setup
# ============================================================================

def setup_logging(
    log_dir: Optional[str] = None,
    log_max_bytes: Optional[int] = None,
    log_backup_count: Optional[int] = None,
    error_log_max_bytes: Optional[int] = None,
    error_log_backup_count: Optional[int] = None,
    retention_days: Optional[int] = None,
    console_level: int = logging.INFO,
    log_basename: Optional[str] = None,
    console_stream: Optional[str] = None,
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
    resolved_log_dir = LOG_DIR if log_dir is None else log_dir
    resolved_log_max_bytes = (
        _resolve_int_env("APP_LOG_MAX_BYTES", LOG_MAX_BYTES, minimum=1)
        if log_max_bytes is None
        else log_max_bytes
    )
    resolved_log_backup_count = (
        _resolve_int_env("APP_LOG_BACKUP_COUNT", LOG_BACKUP_COUNT, minimum=0)
        if log_backup_count is None
        else log_backup_count
    )
    resolved_error_log_max_bytes = (
        _resolve_int_env("APP_ERROR_LOG_MAX_BYTES", ERROR_LOG_MAX_BYTES, minimum=1)
        if error_log_max_bytes is None
        else error_log_max_bytes
    )
    resolved_error_log_backup_count = (
        _resolve_int_env("APP_ERROR_LOG_BACKUP_COUNT", ERROR_LOG_BACKUP_COUNT, minimum=0)
        if error_log_backup_count is None
        else error_log_backup_count
    )
    resolved_retention_days = (
        _resolve_int_env("APP_LOG_RETENTION_DAYS", LOG_RETENTION_DAYS, minimum=0)
        if retention_days is None
        else retention_days
    )

    # Ensure log directory exists, falling back to local data/logs if the
    # primary path (e.g. an SSD mount) is not writable by this user.
    _FALLBACK_LOG_DIR = os.path.join("data", "logs")

    def _try_prepare_log_dir(directory: str) -> bool:
        """Return True if *directory* exists (or was created) and is writable."""
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            # Quick write-access probe — avoids discovering the problem only
            # when the RotatingFileHandler tries to open the log file.
            probe = os.path.join(directory, ".write_probe")
            with open(probe, "w") as _f:
                pass
            os.remove(probe)
            return True
        except (PermissionError, OSError):
            return False

    if not _try_prepare_log_dir(resolved_log_dir):
        print(
            f"[Logging] WARNING: cannot write to '{resolved_log_dir}' "
            f"(permission denied) — falling back to '{_FALLBACK_LOG_DIR}'",
            flush=True,
        )
        resolved_log_dir = _FALLBACK_LOG_DIR
        if not _try_prepare_log_dir(resolved_log_dir):
            # Last resort: console-only logging
            print(
                f"[Logging] ERROR: fallback log dir '{resolved_log_dir}' also not writable"
                " — file logging disabled, console only.",
                flush=True,
            )
            resolved_log_dir = ""

    resolved_basename = log_basename or _resolve_log_basename()
    resolved_console_stream = (console_stream or _resolve_console_stream()).lower()

    if resolved_log_dir:
        # Proactive cleanup of aged log artefacts
        _cleanup_old_logs(
            resolved_log_dir,
            retention_days=resolved_retention_days,
            log_basename=resolved_basename,
        )

    # Logger setup
    app_logger = logging.getLogger("ConvuyerBreadBagCounter")
    app_logger.setLevel(logging.DEBUG)

    # Prevent duplicate handlers on re-import / hot-reload
    if app_logger.handlers:
        return app_logger

    # --- Formatters ---
    detailed_formatter = logging.Formatter(_DETAILED_FMT, datefmt=_DETAILED_DATEFMT)
    console_formatter = logging.Formatter(_CONSOLE_FMT, datefmt=_CONSOLE_DATEFMT)

    if resolved_log_dir:
        # --- 1. Rotating main log (DEBUG+) ---
        main_log_path = os.path.join(resolved_log_dir, f"{resolved_basename}.log")
        main_handler = logging.handlers.RotatingFileHandler(
            main_log_path,
            maxBytes=resolved_log_max_bytes,
            backupCount=resolved_log_backup_count,
            encoding="utf-8",
        )
        main_handler.setLevel(logging.DEBUG)
        main_handler.setFormatter(detailed_formatter)
        main_handler.namer = _namer
        main_handler.rotator = _rotator
        app_logger.addHandler(main_handler)

        # --- 2. Rotating error log (WARNING+) ---
        error_log_path = os.path.join(resolved_log_dir, f"{resolved_basename}_error.log")
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_path,
            maxBytes=resolved_error_log_max_bytes,
            backupCount=resolved_error_log_backup_count,
            encoding="utf-8",
        )
        error_handler.setLevel(logging.WARNING)
        error_handler.setFormatter(detailed_formatter)
        error_handler.namer = _namer
        error_handler.rotator = _rotator
        app_logger.addHandler(error_handler)

    # --- 3. Console handler (INFO+ by default) ---
    if resolved_console_stream != "none":
        stream = sys.stderr if resolved_console_stream == "stderr" else sys.stdout
        console_handler = logging.StreamHandler(stream)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(console_formatter)
        app_logger.addHandler(console_handler)

    # Startup banner
    if resolved_log_dir:
        app_logger.info(
            "[Logging] Initialised — main=%s (max %s MB × %d backups), "
            "error=%s (max %s MB × %d backups), retention=%d days, console=%s",
            os.path.join(resolved_log_dir, f"{resolved_basename}.log"),
            round(resolved_log_max_bytes / (1024 * 1024), 1),
            resolved_log_backup_count,
            os.path.join(resolved_log_dir, f"{resolved_basename}_error.log"),
            round(resolved_error_log_max_bytes / (1024 * 1024), 1),
            resolved_error_log_backup_count,
            resolved_retention_days,
            resolved_console_stream,
        )
    else:
        app_logger.warning("[Logging] File logging disabled — console only (no writable log directory found)")

    return app_logger


# ============================================================================
# Retention / cleanup
# ============================================================================

def _cleanup_old_logs(
    log_dir: str,
    retention_days: int = LOG_RETENTION_DAYS,
    log_basename: Optional[str] = None,
) -> None:
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
    basename = log_basename or _resolve_log_basename()
    patterns = [
        f"{basename}*.log",
        f"{basename}*.log.*",
        f"{basename}*.gz",
        f"{basename}*.rotating",  # intermediate files from non-blocking log rotation
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
    basename = _resolve_log_basename()
    if not os.path.exists(log_dir):
        return {}

    result: Dict[str, str] = {}

    main_log = os.path.join(log_dir, f"{basename}.log")
    if os.path.isfile(main_log):
        result["main_log"] = main_log

    error_log = os.path.join(log_dir, f"{basename}_error.log")
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
