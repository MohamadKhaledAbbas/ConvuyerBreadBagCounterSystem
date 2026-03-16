"""
Database-backed log handler for monitoring.

Captures WARNING, ERROR, and CRITICAL log messages and stores them in the
``monitoring_logs`` database table so they can be queried via the health API.

The handler is intentionally lightweight:
- Delegates actual DB writes to ``DatabaseManager.insert_monitoring_log()``
  which uses the non-blocking async write queue (never blocks the logger).
- Truncates long messages to prevent DB bloat.
- Safe to attach to the root application logger alongside file/console handlers.

Usage:
    from src.logging.db_log_handler import attach_db_log_handler
    attach_db_log_handler(db)   # call once during startup
"""

import logging
from typing import Optional


# Maximum length of message/details stored in DB to prevent bloat
_MAX_MESSAGE_LEN = 1000
_MAX_DETAILS_LEN = 2000


class DatabaseLogHandler(logging.Handler):
    """
    Logging handler that writes WARNING+ records to the monitoring_logs table.

    Uses the DatabaseManager async write queue so emit() never blocks.
    """

    def __init__(self, db, level: int = logging.WARNING):
        super().__init__(level)
        self._db = db

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            if len(message) > _MAX_MESSAGE_LEN:
                message = message[:_MAX_MESSAGE_LEN] + "…"

            # Extract traceback/exception info as details
            details: Optional[str] = None
            if record.exc_info and record.exc_info[1] is not None:
                import traceback
                details = "".join(traceback.format_exception(*record.exc_info))
                if len(details) > _MAX_DETAILS_LEN:
                    details = details[:_MAX_DETAILS_LEN] + "…"

            self._db.insert_monitoring_log(
                level=record.levelname,
                source=record.name,
                message=message,
                details=details,
            )
        except Exception:
            # Never let logging failures propagate — that causes infinite recursion
            pass


def attach_db_log_handler(db) -> DatabaseLogHandler:
    """
    Attach a :class:`DatabaseLogHandler` to the application logger.

    Should be called once during server startup after the DB is initialized.

    Args:
        db: A ``DatabaseManager`` instance.

    Returns:
        The handler instance (can be used for later removal if needed).
    """
    app_logger = logging.getLogger("ConvuyerBreadBagCounter")

    # Prevent duplicate handlers on hot-reload
    for h in app_logger.handlers:
        if isinstance(h, DatabaseLogHandler):
            return h

    handler = DatabaseLogHandler(db, level=logging.WARNING)
    # Use a concise format — the full detailed format is in file logs
    handler.setFormatter(logging.Formatter("%(message)s"))
    app_logger.addHandler(handler)

    app_logger.info("[DBLogHandler] Database log handler attached (WARNING+)")
    return handler
