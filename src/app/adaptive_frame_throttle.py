"""
Adaptive Frame Throttle — power-saving idle detection for the conveyor pipeline.

When the production line is idle (no detections for a configurable timeout,
default 30 minutes), the system degrades to processing every Nth frame
(default N=5).  This reduces CPU load by ~80 % and extends hardware lifespan
without affecting counting reliability.

As soon as ANY detection is found in a processed frame, the throttle
immediately wakes back to full processing rate.  A configurable hysteresis
window (default 60 s) prevents rapid oscillation between modes.

Usage (integrated into ConveyorCounterApp):

    throttle = AdaptiveFrameThrottle(config)

    for frame, latency in source.frames():
        frame_count += 1
        throttle.check_timeout()

        if not throttle.should_process(frame_count):
            continue  # skip processing, but keep reading to drain queue

        detections, tracks, rois = pipeline.process_frame(frame)

        if detections:
            throttle.report_activity()

Thread-safety:
    All public methods are guarded by a threading.Lock so the throttle
    can be queried safely from the FastAPI endpoint thread.
"""

import threading
import time
from typing import Dict, Any

from src.utils.AppLogging import logger


class AdaptiveFrameThrottle:
    """
    Adaptive frame processing throttle with two modes:

        FULL      – every frame is processed (normal production)
        DEGRADED  – only every Nth frame is processed (idle / power-saving)

    State machine::

        FULL ── (idle_timeout_s with no activity) ──→ DEGRADED
          ↑                                               │
          └──── (detection in processed frame) ───────────┘
                 (stays FULL for at least hysteresis_s)
    """

    # Mode constants
    MODE_FULL = "full"
    MODE_DEGRADED = "degraded"

    def __init__(
        self,
        enabled: bool = True,
        idle_timeout_s: float = 1800.0,
        skip_n: int = 5,
        hysteresis_s: float = 60.0,
    ):
        """
        Args:
            enabled:        Master switch.  When False the throttle is a no-op
                            (always returns True from should_process).
            idle_timeout_s: Seconds of zero detections before switching to
                            degraded mode.  Default 1800 (30 minutes).
            skip_n:         In degraded mode, process every Nth frame.
                            Default 5 (process 1 out of every 5 frames → ~80 %
                            reduction in processing load).
            hysteresis_s:   After waking from degraded → full, the throttle
                            stays in full mode for at least this many seconds
                            before it can degrade again.  Prevents rapid
                            oscillation from single spurious detections.
                            Default 60 seconds.
        """
        self._enabled = enabled
        self._idle_timeout_s = max(0.01, idle_timeout_s)
        self._skip_n = max(2, skip_n)
        self._hysteresis_s = max(0.0, hysteresis_s)

        self._lock = threading.Lock()
        self._mode = self.MODE_FULL
        self._last_activity_time = time.monotonic()
        # Set wake time far in the past so hysteresis doesn't block the
        # very first degradation transition after startup.
        self._last_wake_time = time.monotonic() - (self._hysteresis_s + 1.0)

        # Counters (for diagnostics / pipeline state)
        self._frames_skipped: int = 0
        self._total_frames_seen: int = 0
        self._degraded_transitions: int = 0
        self._wake_transitions: int = 0

        # Logging throttle — avoid spamming logs during degraded mode
        self._last_degraded_log_time: float = 0.0
        self._degraded_log_interval: float = 300.0  # log every 5 min in degraded

        if enabled:
            logger.info(
                f"[FrameThrottle] Initialized: idle_timeout={idle_timeout_s}s, "
                f"skip_n={skip_n}, hysteresis={hysteresis_s}s"
            )
        else:
            logger.info("[FrameThrottle] Disabled (all frames will be processed)")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def should_process(self, frame_number: int) -> bool:
        """
        Decide whether the given frame should be processed.

        In FULL mode, every frame is processed.
        In DEGRADED mode, only frames where ``frame_number % skip_n == 0``
        are processed.

        Args:
            frame_number: 1-based sequential frame counter.

        Returns:
            True if the frame should be processed, False to skip.
        """
        if not self._enabled:
            return True

        with self._lock:
            self._total_frames_seen += 1

            if self._mode == self.MODE_FULL:
                return True

            # Degraded mode — process every Nth frame
            if frame_number % self._skip_n == 0:
                return True

            self._frames_skipped += 1
            return False

    def report_activity(self):
        """
        Called when the pipeline finds detections (i.e. the conveyor is active).

        Resets the idle timer and, if in degraded mode, wakes back to full
        processing rate immediately.
        """
        if not self._enabled:
            return

        with self._lock:
            now = time.monotonic()
            self._last_activity_time = now

            if self._mode == self.MODE_DEGRADED:
                self._mode = self.MODE_FULL
                self._last_wake_time = now
                self._wake_transitions += 1
                logger.info(
                    f"[FrameThrottle] WAKE → FULL | Detection found in degraded mode, "
                    f"resuming full-rate processing (skipped {self._frames_skipped} frames "
                    f"during idle period)"
                )

    def check_timeout(self):
        """
        Check whether the idle timeout has elapsed and transition to degraded
        mode if appropriate.  Call this once per loop iteration.
        """
        if not self._enabled:
            return

        with self._lock:
            if self._mode != self.MODE_FULL:
                # Already degraded — periodic logging only
                self._maybe_log_degraded()
                return

            now = time.monotonic()
            idle_seconds = now - self._last_activity_time

            # Respect hysteresis: don't degrade too soon after waking
            if now - self._last_wake_time < self._hysteresis_s:
                return

            if idle_seconds >= self._idle_timeout_s:
                self._mode = self.MODE_DEGRADED
                self._degraded_transitions += 1
                idle_min = idle_seconds / 60.0
                logger.info(
                    f"[FrameThrottle] DEGRADE → processing every {self._skip_n}th frame | "
                    f"No detections for {idle_min:.1f} min "
                    f"(threshold={self._idle_timeout_s / 60:.0f} min). "
                    f"CPU load reduced ~{(1 - 1 / self._skip_n) * 100:.0f}%"
                )
                self._last_degraded_log_time = now

    def get_state(self) -> Dict[str, Any]:
        """
        Return a snapshot of the throttle state for the pipeline state file
        (consumed by the FastAPI /counts endpoint).

        Thread-safe.
        """
        with self._lock:
            now = time.monotonic()
            idle_seconds = now - self._last_activity_time
            return {
                "enabled": self._enabled,
                "mode": self._mode,
                "idle_seconds": round(idle_seconds, 1),
                "idle_timeout_s": self._idle_timeout_s,
                "skip_n": self._skip_n,
                "hysteresis_s": self._hysteresis_s,
                "frames_skipped": self._frames_skipped,
                "total_frames_seen": self._total_frames_seen,
                "degraded_transitions": self._degraded_transitions,
                "wake_transitions": self._wake_transitions,
            }

    @property
    def mode(self) -> str:
        """Current mode (thread-safe)."""
        with self._lock:
            return self._mode

    @property
    def is_degraded(self) -> bool:
        """True if currently in degraded (idle) mode."""
        with self._lock:
            return self._mode == self.MODE_DEGRADED

    @property
    def frames_skipped(self) -> int:
        """Total frames skipped since start (thread-safe)."""
        with self._lock:
            return self._frames_skipped

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _maybe_log_degraded(self):
        """Periodic status log while in degraded mode (every 5 min)."""
        now = time.monotonic()
        if now - self._last_degraded_log_time >= self._degraded_log_interval:
            idle_min = (now - self._last_activity_time) / 60.0
            logger.info(
                f"[FrameThrottle] Still DEGRADED | idle for {idle_min:.0f} min, "
                f"skipped {self._frames_skipped} frames total, "
                f"processing every {self._skip_n}th frame"
            )
            self._last_degraded_log_time = now



