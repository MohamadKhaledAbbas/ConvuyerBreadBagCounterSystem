"""
Alert Service — Curated, rate-limited operational alerts.

Unlike the ``monitoring_logs`` table (which captures ALL WARNING+ log
messages), the alert service is designed for:

- **Curated** alerts: only specific, actionable conditions
- **Rate-limited**: max 1 alert per (source, message) per 5 minutes
- **Low volume**: safe to display on a dashboard without flooding

Usage:
    from src.logging.alert_service import alert_service

    # In any catch block or critical check:
    alert_service.record("ClassificationWorker", "error",
                         "Model inference failed", details="TimeoutError...")

The service writes to the existing ``monitoring_logs`` table using a
special ``[ALERT]`` prefix so the health UI can distinguish curated
alerts from raw log captures.
"""

import threading
import time
from typing import Optional, Dict, Tuple

from src.utils.AppLogging import logger


class AlertService:
    """
    Rate-limited alert recorder.

    Deduplicates alerts by (source, message_key) so the same error
    doesn't flood the database.  All alerts are written via the
    application logger at WARNING/ERROR/CRITICAL level, which means
    the existing ``DatabaseLogHandler`` picks them up and stores
    them in ``monitoring_logs`` automatically.

    The rate limit is per (source, first-60-chars-of-message) pair.
    """

    # Default cooldown: 5 minutes between identical alerts
    DEFAULT_COOLDOWN_SECONDS = 300

    def __init__(self, cooldown_seconds: int = DEFAULT_COOLDOWN_SECONDS):
        self._cooldown = cooldown_seconds
        self._last_fired: Dict[Tuple[str, str], float] = {}
        self._lock = threading.Lock()

    def record(
        self,
        source: str,
        severity: str,
        message: str,
        details: Optional[str] = None,
    ) -> bool:
        """
        Record an operational alert if rate limit allows.

        Args:
            source: Module/component name (e.g. "ConveyorCounterApp")
            severity: "warning", "error", or "critical"
            message: Human-readable alert message
            details: Optional extra context (traceback, JSON, etc.)

        Returns:
            True if the alert was actually emitted, False if rate-limited.
        """
        now = time.time()
        # Key for deduplication (source + first 60 chars of message)
        dedup_key = (source, message[:60])

        with self._lock:
            last = self._last_fired.get(dedup_key, 0)
            if now - last < self._cooldown:
                return False
            self._last_fired[dedup_key] = now

            # Prune stale entries periodically (keep dict small)
            if len(self._last_fired) > 200:
                cutoff = now - self._cooldown * 2
                self._last_fired = {
                    k: v for k, v in self._last_fired.items() if v > cutoff
                }

        # Emit via the application logger so DatabaseLogHandler stores it
        # Prefix with [ALERT] for easy filtering in the UI
        alert_msg = f"[ALERT][{source}] {message}"
        if details:
            alert_msg += f" | {details[:500]}"

        level = severity.lower()
        if level == "critical":
            logger.critical(alert_msg)
        elif level == "error":
            logger.error(alert_msg)
        else:
            logger.warning(alert_msg)

        return True

    def clear_cooldowns(self):
        """Clear all rate-limit state (for testing)."""
        with self._lock:
            self._last_fired.clear()


# ── Module-level singleton ──────────────────────────────────────────────
# Import and use directly:  from src.logging.alert_service import alert_service
alert_service = AlertService()

