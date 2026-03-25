"""
Adaptive Frame Throttle — power-saving idle detection for the conveyor pipeline.

When the production line is idle (no confirmed tracks for a configurable timeout,
default 15 minutes), the system degrades to processing every Nth frame
(default N=5).  This reduces CPU load by ~80 % and extends hardware lifespan
without affecting counting reliability.

Two-Signal Architecture:
    Signal A — ``report_detection()``  (fast-path wake)
        A detection **inside the conveyor ROI** immediately wakes the system
        from DEGRADED to FULL mode.  Does NOT reset the idle timer.
        This guarantees zero bag skips — worst-case latency to first detection
        in DEGRADED mode is (skip_n - 1) frames (~235 ms at 17 FPS, skip_n=5).

        Important: callers MUST pre-filter detections to the conveyor ROI
        before deciding whether to call this method.  Outside-belt detections
        (operator hands, table-edge reflections, etc.) must never reach here.
        When ``conveyor_roi_enabled=True`` the pipeline already drops
        out-of-ROI detections before returning them; when the flag is False
        (debug/test mode), the caller must apply the ROI bounds manually.

    Signal B — ``report_activity()``  (noise-filtered stay-alive)
        Called ONLY when confirmed tracks (hits >= min_track_duration_frames)
        or ghost tracks exist.  Resets the idle timer.  Also wakes from
        DEGRADED if still in that mode (belt-and-suspenders).  Confirmed
        tracks are inherently ROI-scoped because they are built from
        ROI-filtered detections in PipelineCore.

This separation prevents environmental noise (reflections, vibrations) from
resetting the idle timer while ensuring real bags wake the system instantly.

Usage (integrated into ConveyorCounterApp):

    throttle = AdaptiveFrameThrottle(config)

    for frame, latency in source.frames():
        frame_count += 1
        throttle.check_timeout()

        if not throttle.should_process(frame_count):
            continue  # skip processing, but keep reading to drain queue

        detections, tracks, rois = pipeline.process_frame(frame)

        # Signal A: fast-path wake on in-ROI detection only.
        # When conveyor_roi_enabled=True, `detections` is already filtered;
        # otherwise apply the ROI bounds explicitly before calling this.
        roi_detections = _filter_to_roi(detections, tracking_config)
        if roi_detections:
            throttle.report_detection()

        # Signal B: noise-filtered timer reset on confirmed/ghost tracks.
        # Tracks are inherently ROI-scoped (built from filtered detections).
        confirmed = tracker.get_confirmed_tracks()
        ghosts_active = len(tracker.ghost_tracks) > 0
        if confirmed or ghosts_active:
            throttle.report_activity()

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
    Adaptive frame processing throttle with two modes:

        FULL      – every frame is processed (normal production)
        DEGRADED  – only every Nth frame is processed (idle / power-saving)

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
        skip_n: int = 5,
        hysteresis_s: float = 60.0,
        on_mode_change: Optional[Callable[[str], None]] = None,
    ):
        """
        Args:
            enabled:        Master switch.  When False the throttle is a no-op
                            (always returns True from should_process).
            idle_timeout_s: Seconds of zero confirmed-track activity before
                            switching to degraded mode.  Default 900 (15 min).
            skip_n:         In degraded mode, process every Nth frame.
                            Default 5 (process 1 out of every 5 frames → ~80 %
                            reduction in processing load).
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

        # Track which signal caused the last wake for diagnostics
        self._last_wake_signal: str = ""

        # Detection-only wakes (Signal A without Signal B follow-up)
        # These indicate spurious noise detections.
        self._detection_only_wakes: int = 0

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
                f"[FrameThrottle] Initialized (two-signal): idle_timeout={idle_timeout_s}s, "
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
        Spurious in-ROI detections cause at most a 60-second FULL wake
        (hysteresis) before re-degrading, because the idle timer is not
        touched.
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
                self._detection_only_wakes += 1
                idle_min = (now - self._last_activity_time) / 60.0
                logger.info(
                    f"[FrameThrottle] WAKE → FULL (Signal A: detection) | "
                    f"Resuming full-rate processing | "
                    f"idle_timer NOT reset (was {idle_min:.1f} min) | "
                    f"skipped {self._frames_skipped} frames during idle period"
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
                    f"Resuming full-rate processing + timer reset | "
                    f"skipped {self._frames_skipped} frames during idle period"
                )
                _fire_callback = True
            else:
                # Already FULL — if the last wake was detection-only (Signal A),
                # the follow-up Signal B confirms it was a real bag.
                # Decrement detection_only_wakes since it's now validated.
                if self._last_wake_signal == "detection" and self._detection_only_wakes > 0:
                    self._detection_only_wakes -= 1
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
                    f"[FrameThrottle] DEGRADE → processing every {self._skip_n}th frame | "
                    f"No confirmed tracks for {idle_min:.1f} min "
                    f"(threshold={self._idle_timeout_s / 60:.0f} min). "
                    f"CPU load reduced ~{(1 - 1 / self._skip_n) * 100:.0f}%"
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
                "skip_n": self._skip_n,
                "hysteresis_s": self._hysteresis_s,
                "frames_skipped": self._frames_skipped,
                "total_frames_seen": self._total_frames_seen,
                "degraded_transitions": self._degraded_transitions,
                "wake_transitions": self._wake_transitions,
                "last_wake_signal": self._last_wake_signal,
                "detection_only_wakes": self._detection_only_wakes,
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

