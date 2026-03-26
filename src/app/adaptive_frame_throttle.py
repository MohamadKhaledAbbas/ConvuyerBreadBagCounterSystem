"""
Adaptive Frame Throttle — idle power-saving mode coordinator.

When the production line is idle (no confirmed tracks for a configurable
timeout, default 15 minutes), the system switches to DEGRADED mode.

In DEGRADED mode the pipeline-wide sentinel (SpoolProcessorNode) publishes
one probe frame per second from the latest disk segment, reducing VPU/CPU
to ~6 % of full-rate usage.  The app processes every probe frame it receives
— no additional app-side frame skipping is applied.  The spool processor is
the sole rate limiter in idle mode.

Two-Signal Architecture:
    Signal A — ``report_detection()``  (fast-path wake)
        A detection **inside the conveyor ROI** immediately wakes the system
        from DEGRADED to FULL mode.  Does NOT reset the idle timer.
        This guarantees zero bag skips — worst-case latency to first detection
        in DEGRADED mode equals one sentinel probe interval (~1 s).

        Important: callers MUST pre-filter detections to the conveyor ROI
        before calling this method.  Outside-belt detections (operator hands,
        table-edge reflections, etc.) must never reach here.

    Signal B — ``report_activity()``  (noise-filtered stay-alive)
        Called ONLY when confirmed tracks (hits >= min_track_duration_frames)
        or ghost tracks exist.  Resets the idle timer.  Also wakes from
        DEGRADED if still in that mode (belt-and-suspenders).  Confirmed
        tracks are inherently ROI-scoped because they are built from
        ROI-filtered detections in PipelineCore.

This separation prevents environmental noise (reflections, vibrations) from
resetting the idle timer while ensuring real bags wake the system instantly.

Thread-safety:
    All public methods are guarded by a threading.Lock so the throttle
    can be queried safely from the FastAPI endpoint thread.
"""

import threading
import time
from typing import Callable, Dict, Any, Optional

from src.utils.AppLogging import logger


class AdaptiveFrameThrottle:
    """
    Adaptive power-save coordinator with two modes:

        FULL      – normal production; SpoolProcessor sends frames at full rate.
        DEGRADED  – idle mode; SpoolProcessor sends 1 sentinel probe frame/sec.
                    The app processes every frame it receives — no extra skipping.

    State machine::

        FULL ── (idle_timeout_s with no Signal B) ──→ DEGRADED
          ↑                                               │
          └──── (Signal A: detection in any frame) ───────┘
          └──── (Signal B: confirmed/ghost tracks)  ──────┘
                 (stays FULL for at least hysteresis_s)

    Two-signal design:
        Signal A (report_detection) — FAST wake, does NOT reset idle timer
        Signal B (report_activity)  — Noise-filtered, DOES reset idle timer
    """

    # Mode constants
    MODE_FULL = "full"
    MODE_DEGRADED = "degraded"

    def __init__(
        self,
        enabled: bool = True,
        idle_timeout_s: float = 900.0,
        hysteresis_s: float = 60.0,
        on_mode_change: Optional[Callable[[str], None]] = None,
    ):
        """
        Args:
            enabled:        Master switch.  When False the throttle is a no-op
                            (always FULL, no transitions, no callbacks).
            idle_timeout_s: Seconds of zero confirmed-track activity before
                            switching to degraded mode.  Default 900 (15 min).
            hysteresis_s:   After waking from degraded → full, the throttle
                            stays in full mode for at least this many seconds
                            before it can degrade again.  Prevents rapid
                            oscillation from single spurious detections.
                            Default 60 seconds.
            on_mode_change: Optional callback invoked **outside the lock**
                            whenever the mode transitions (FULL↔DEGRADED).
                            Receives the new mode string ("full" / "degraded").
                            Used by ConveyorCounterApp to write the shared
                            throttle state file for cross-process coordination
                            with SpoolProcessorNode.
        """
        self._enabled = enabled
        self._idle_timeout_s = max(0.01, idle_timeout_s)
        self._hysteresis_s = max(0.0, hysteresis_s)

        self._lock = threading.Lock()
        self._mode = self.MODE_FULL
        self._last_activity_time = time.monotonic()
        # Set wake time far in the past so hysteresis doesn't block the
        # very first degradation transition after startup.
        self._last_wake_time = time.monotonic() - (self._hysteresis_s + 1.0)

        # Counters (for diagnostics / pipeline state)
        self._degraded_transitions: int = 0
        self._wake_transitions: int = 0

        # Track which signal caused the last wake for diagnostics
        self._last_wake_signal: str = ""

        # Timestamp when we last entered DEGRADED mode (None when in FULL mode).
        # Used to compute degraded_since_seconds in get_state().
        self._degraded_since: float | None = None

        # Logging throttle — avoid spamming logs during degraded mode
        self._last_degraded_log_time: float = 0.0
        self._degraded_log_interval: float = 300.0  # log every 5 min in degraded

        # Cross-process mode-change callback (invoked outside the lock)
        self._on_mode_change: Optional[Callable[[str], None]] = on_mode_change

        if enabled:
            logger.info(
                f"[FrameThrottle] Initialized: idle_timeout={idle_timeout_s}s, "
                f"hysteresis={hysteresis_s}s | "
                f"Sentinel probe rate controlled by SpoolProcessorNode"
            )
        else:
            logger.info("[FrameThrottle] Disabled (pipeline always runs at full rate)")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def report_detection(self):
        """
        Signal A — Fast-path wake signal.

        Called when at least one detection that falls **inside the conveyor
        ROI** is found in a processed frame.  Callers are responsible for
        pre-filtering detections to the conveyor ROI before invoking this
        method; outside-belt detections (operator hands, reflections, etc.)
        must not reach here.

        If in DEGRADED mode, immediately wakes to FULL mode.
        Does NOT reset the idle timer (``_last_activity_time``).

        This is the fast-path guard that guarantees zero bag skips.
        Spurious in-ROI detections cause at most hysteresis_s seconds of
        FULL mode before re-degrading, because the idle timer is not touched.
        """
        if not self._enabled:
            return

        _fire_callback = False
        with self._lock:
            if self._mode == self.MODE_DEGRADED:
                now = time.monotonic()
                self._mode = self.MODE_FULL
                self._degraded_since = None
                self._last_wake_time = now
                self._wake_transitions += 1
                self._last_wake_signal = "detection"
                idle_min = (now - self._last_activity_time) / 60.0
                logger.info(
                    f"[FrameThrottle] WAKE → FULL (Signal A: detection) | "
                    f"Sentinel → full-rate processing | "
                    f"idle_timer NOT reset (was {idle_min:.1f} min)"
                )
                _fire_callback = True

        # Invoke callback outside the lock to avoid deadlocks
        if _fire_callback and self._on_mode_change is not None:
            try:
                self._on_mode_change(self.MODE_FULL)
            except Exception as e:
                logger.warning(f"[FrameThrottle] on_mode_change callback error: {e}")

    def report_activity(self):
        """
        Signal B — Noise-filtered stay-alive signal.

        Called ONLY when confirmed tracks (hits >= min_track_duration_frames)
        or ghost tracks exist — i.e. real, multi-frame validated objects
        are present on the conveyor.

        Effects:
          1. Always resets ``_last_activity_time`` (keeps FULL mode alive)
          2. If in DEGRADED mode, also wakes to FULL (belt-and-suspenders)

        Because confirmed tracks require 5+ consecutive detector matches,
        environmental noise (reflections, vibrations) cannot trigger this
        signal.  This makes the idle timer highly resistant to false resets.
        """
        if not self._enabled:
            return

        _fire_callback = False
        with self._lock:
            now = time.monotonic()
            self._last_activity_time = now

            if self._mode == self.MODE_DEGRADED:
                self._mode = self.MODE_FULL
                self._degraded_since = None
                self._last_wake_time = now
                self._wake_transitions += 1
                self._last_wake_signal = "confirmed_track"
                logger.info(
                    f"[FrameThrottle] WAKE → FULL (Signal B: confirmed track) | "
                    f"Sentinel → full-rate processing + timer reset"
                )
                _fire_callback = True
            else:
                # Already FULL — if the last wake was Signal A (detection),
                # the follow-up Signal B confirms it was a real bag.
                if self._last_wake_signal == "detection":
                    self._last_wake_signal = "confirmed_track"

        # Invoke callback outside the lock to avoid deadlocks
        if _fire_callback and self._on_mode_change is not None:
            try:
                self._on_mode_change(self.MODE_FULL)
            except Exception as e:
                logger.warning(f"[FrameThrottle] on_mode_change callback error: {e}")

    def check_timeout(self):
        """
        Check whether the idle timeout has elapsed and transition to degraded
        mode if appropriate.  Call this once per loop iteration.
        """
        if not self._enabled:
            return

        _fire_callback = False
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
                self._degraded_since = now
                self._degraded_transitions += 1
                idle_min = idle_seconds / 60.0
                logger.info(
                    f"[FrameThrottle] DEGRADE → sentinel probe mode | "
                    f"No confirmed tracks for {idle_min:.1f} min "
                    f"(threshold={self._idle_timeout_s / 60:.0f} min). "
                    f"SpoolProcessor switching to 1 probe frame/sec"
                )
                self._last_degraded_log_time = now
                _fire_callback = True

        # Invoke callback outside the lock to avoid deadlocks
        if _fire_callback and self._on_mode_change is not None:
            try:
                self._on_mode_change(self.MODE_DEGRADED)
            except Exception as e:
                logger.warning(f"[FrameThrottle] on_mode_change callback error: {e}")

    def get_state(self) -> Dict[str, Any]:
        """
        Return a snapshot of the throttle state for the pipeline state file
        (consumed by the FastAPI /counts and /health endpoints).

        Thread-safe.
        """
        with self._lock:
            now = time.monotonic()
            idle_seconds = now - self._last_activity_time

            # How long have we been in DEGRADED mode (None when FULL)
            degraded_since_seconds = (
                round(now - self._degraded_since, 1)
                if self._degraded_since is not None else None
            )

            # Seconds remaining before degradation (None when already DEGRADED)
            time_until_degrade_s = (
                round(max(0.0, self._idle_timeout_s - idle_seconds), 1)
                if self._mode == self.MODE_FULL else None
            )

            # Idle progress as 0–100 % (useful for a progress bar in the UI)
            idle_percent = round(
                min(100.0, idle_seconds / self._idle_timeout_s * 100.0), 1
            )

            return {
                "enabled": self._enabled,
                "mode": self._mode,
                "idle_seconds": round(idle_seconds, 1),
                "idle_timeout_s": self._idle_timeout_s,
                "idle_percent": idle_percent,
                "time_until_degrade_s": time_until_degrade_s,
                "degraded_since_seconds": degraded_since_seconds,
                "hysteresis_s": self._hysteresis_s,
                "degraded_transitions": self._degraded_transitions,
                "wake_transitions": self._wake_transitions,
                "last_wake_signal": self._last_wake_signal,
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _maybe_log_degraded(self):
        """Periodic status log while in degraded mode (every 5 min)."""
        now = time.monotonic()
        if now - self._last_degraded_log_time >= self._degraded_log_interval:
            idle_min = (now - self._last_activity_time) / 60.0
            logger.info(
                f"[FrameThrottle] Still DEGRADED (sentinel mode) | "
                f"idle for {idle_min:.0f} min | "
                f"wake_transitions={self._wake_transitions}"
            )
            self._last_degraded_log_time = now

