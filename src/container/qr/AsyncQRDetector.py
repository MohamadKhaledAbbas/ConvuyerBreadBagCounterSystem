"""
Asynchronous wrapper around :class:`QRCodeDetector`.

The WeChat CNN QR detector takes 80–150 ms per call on RDK, which is
larger than the main loop's per-frame budget at 20 FPS (50 ms).  Running
it inline therefore stalls the producer/consumer pipeline and starves the
ROS2 frame queue, leading to dropped frames and laggy event recording.

This module decouples detection from the main loop using a single
background worker thread with **latest-frame-wins** semantics:

* :meth:`submit` is non-blocking and stores the frame in a 1-slot inbox.
  If a previous frame was still waiting it is *overwritten* (and counted
  as ``dropped_overwritten``).
* The worker pops the inbox, runs the (optional) motion gate and the
  CNN detector, then publishes the result into a 1-slot outbox.
* :meth:`poll_result` is non-blocking and returns the most recent
  unconsumed result (or ``None``).  If a previous result was still waiting
  it is overwritten and counted as ``dropped_unconsumed`` — that means the
  main loop is consuming results faster than the BG can produce them
  (rare in practice; the opposite is the typical case).

Design properties:
    * Bounded memory: at most 1 pending frame and 1 pending result in flight.
    * Wait-free producer (main loop): :meth:`submit` and :meth:`poll_result`
      acquire a mutex briefly but never block on detection work.
    * Stickiness: after a detection finds at least one QR, the next
      ``_FORCE_DETECT_FRAMES`` worker passes bypass the motion gate.
      This prevents a stationary container with a visible QR from being
      gated out by a calm scene.
    * Observability: per-pass timing, drop counters, and motion-gate
      counters are exposed via :attr:`stats` and :meth:`get_stats`.

The class is intentionally agnostic of the main loop's frame counter: the
caller passes ``seq`` to :meth:`submit` and gets it back on the result so
it can compute pipeline lag.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.container.qr.QRCodeDetector import QRCodeDetector, QRDetection
from src.utils.AppLogging import logger


# Toggle verbose per-pass logging by setting ASYNC_QR_DEBUG=1 (or 2 for every pass).
_ASYNC_QR_DEBUG = int(os.environ.get('ASYNC_QR_DEBUG', '0') or '0')


@dataclass(frozen=True)
class AsyncQRResult:
    """Result of one background detection pass."""

    detections: List[QRDetection]
    frame_seq: int          # main-loop frame counter for the processed frame
    elapsed_ms: float       # wall-clock time spent inside detector.detect_all()
    submit_mono: float      # time.monotonic() at submit
    done_mono: float        # time.monotonic() at completion
    motion_diff: float      # mean abs pixel diff used by the motion gate

    @property
    def queue_lag_ms(self) -> float:
        """Time the frame waited in the inbox before the worker picked it up."""
        return (self.done_mono - self.submit_mono) * 1000.0 - self.elapsed_ms


class AsyncQRDetector:
    """Background-threaded wrapper around :class:`QRCodeDetector`.

    Args:
        detector: a fully-initialised :class:`QRCodeDetector` instance.
        motion_threshold: mean abs grayscale pixel diff (0–255 scale) below
            which the worker skips the CNN detector entirely.  ``0.0``
            disables the motion gate.

    Thread model:
        Exactly one worker thread is spawned by :meth:`start`.  All public
        methods are safe to call from the main loop concurrently with the
        worker.  No public method blocks on detection work.
    """

    # Number of consecutive worker passes after a positive detection that
    # bypass the motion gate.  Stops a stationary-but-visible QR from being
    # gated out as "no motion" once the scene calms down.
    _FORCE_DETECT_FRAMES: int = 5

    # Downsample size used by the motion gate.  Tiny images keep the gate
    # well under 1 ms even on RDK.
    _GATE_W: int = 160
    _GATE_H: int = 90

    # Worker queue wait timeout.  Short enough to react to stop() promptly,
    # long enough to avoid burning CPU when idle.
    _WAIT_TIMEOUT_S: float = 0.25

    # Join timeout used during stop().  Detection passes can run >100 ms
    # so we give a small grace period.
    _JOIN_TIMEOUT_S: float = 3.0

    def __init__(
        self,
        detector: QRCodeDetector,
        motion_threshold: float = 0.0,
    ) -> None:
        if detector is None:
            raise ValueError("AsyncQRDetector requires a non-None detector")
        self._detector = detector
        self._motion_threshold = max(0.0, float(motion_threshold))

        # Synchronisation primitives.  A single condition variable guards
        # both the inbox and outbox; contention is negligible.
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

        # 1-slot inbox: (frame_ref, frame_seq, submit_monotonic) or None.
        # Frames are passed by reference; the producer guarantees not to
        # mutate the buffer after submit().
        self._pending: Optional[Tuple[np.ndarray, int, float]] = None

        # 1-slot outbox: latest result not yet polled.
        self._result: Optional[AsyncQRResult] = None

        # Worker control.
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._started = False

        # Worker-only state (never touched by the main loop).
        self._prev_gate_gray: Optional[np.ndarray] = None
        self._force_detect_remaining: int = 0

        # Stats counters (protected by self._lock).
        self._submitted: int = 0
        self._processed: int = 0
        self._with_detections: int = 0
        self._skipped_motion: int = 0
        self._dropped_overwritten: int = 0     # producer overwrote unconsumed inbox
        self._dropped_unconsumed: int = 0      # worker overwrote unconsumed outbox
        self._errors: int = 0
        self._total_elapsed_ms: float = 0.0
        self._max_elapsed_ms: float = 0.0
        self._last_elapsed_ms: float = 0.0
        self._started_mono: float = 0.0

        logger.info(
            f"[AsyncQRDetector] Initialised engine={detector.engine_name} "
            f"motion_threshold={self._motion_threshold:.2f}"
        )

    # ── Lifecycle ──────────────────────────────────────────────────────

    def start(self) -> None:
        """Spawn the background worker thread.  Idempotent."""
        if self._started:
            logger.warning("[AsyncQRDetector] start() called twice — ignored")
            return
        self._stop_event.clear()
        self._started_mono = time.monotonic()
        self._thread = threading.Thread(
            target=self._worker_loop,
            name="AsyncQRDetector",
            daemon=True,
        )
        self._thread.start()
        self._started = True
        logger.info("[AsyncQRDetector] Worker thread started")

    def stop(self) -> None:
        """Signal the worker to exit and join it (best-effort)."""
        if not self._started:
            return
        self._stop_event.set()
        with self._cond:
            self._cond.notify_all()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=self._JOIN_TIMEOUT_S)
            if t.is_alive():
                logger.warning(
                    f"[AsyncQRDetector] Worker did not exit within "
                    f"{self._JOIN_TIMEOUT_S}s — leaving as daemon"
                )
        self._started = False
        # Final summary so operators can see lifetime stats in the log tail.
        snap = self.stats
        logger.info(
            "[AsyncQRDetector] Stopped. "
            f"submitted={snap['submitted']} "
            f"processed={snap['processed']} "
            f"with_detections={snap['with_detections']} "
            f"skipped_motion={snap['skipped_motion']} "
            f"dropped_overwritten={snap['dropped_overwritten']} "
            f"dropped_unconsumed={snap['dropped_unconsumed']} "
            f"errors={snap['errors']} "
            f"avg_ms={snap['avg_elapsed_ms']:.1f} "
            f"max_ms={snap['max_elapsed_ms']:.1f}"
        )

    def __enter__(self) -> "AsyncQRDetector":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # ── Producer API (called from main loop) ───────────────────────────

    def submit(self, frame: np.ndarray, seq: int) -> None:
        """Hand a frame to the worker.

        Non-blocking.  If the worker has not yet picked up the previous
        frame, that frame is overwritten and counted as
        ``dropped_overwritten``.

        Args:
            frame: BGR image.  The producer must not mutate the buffer
                after this call until at least one further frame has been
                submitted (i.e. the buffer is logically owned by the
                detector until replaced).
            seq: monotonic main-loop frame counter, returned on the result.
        """
        if frame is None or frame.size == 0:
            return
        if not self._started:
            # Defensive: silently no-op so a misordered start/stop never
            # crashes the main loop.  Logged once.
            if not getattr(self, "_warned_not_started", False):
                logger.warning(
                    "[AsyncQRDetector] submit() called before start() — frame dropped"
                )
                self._warned_not_started = True
            return

        ts = time.monotonic()
        with self._cond:
            if self._pending is not None:
                self._dropped_overwritten += 1
                if _ASYNC_QR_DEBUG:
                    logger.info(
                        f"[AsyncQR] Overwriting unprocessed inbox frame "
                        f"old_seq={self._pending[1]} new_seq={seq}"
                    )
            self._pending = (frame, int(seq), ts)
            self._submitted += 1
            self._cond.notify()

    def poll_result(self) -> Optional[AsyncQRResult]:
        """Return the most recent unconsumed result (or ``None``).

        Non-blocking.  Each result is returned at most once: a successful
        poll empties the outbox.
        """
        with self._lock:
            r = self._result
            self._result = None
        return r

    # ── Stats / introspection ─────────────────────────────────────────

    @property
    def stats(self) -> dict:
        """Snapshot of internal counters.  Cheap; safe to call per frame."""
        with self._lock:
            n = max(1, self._processed)
            return {
                'submitted': self._submitted,
                'processed': self._processed,
                'with_detections': self._with_detections,
                'skipped_motion': self._skipped_motion,
                'dropped_overwritten': self._dropped_overwritten,
                'dropped_unconsumed': self._dropped_unconsumed,
                'errors': self._errors,
                'avg_elapsed_ms': self._total_elapsed_ms / n,
                'max_elapsed_ms': self._max_elapsed_ms,
                'last_elapsed_ms': self._last_elapsed_ms,
                'pending': self._pending is not None,
                'has_unconsumed_result': self._result is not None,
            }

    def get_stats(self) -> dict:
        """Stats dict shaped for the health endpoint.  Adds a few derived metrics."""
        snap = self.stats
        uptime = max(1e-3, time.monotonic() - self._started_mono) if self._started_mono else 0.0
        snap.update({
            'engine': self._detector.engine_name,
            'motion_threshold': self._motion_threshold,
            'uptime_s': uptime,
            'detect_fps': snap['processed'] / uptime if uptime > 0 else 0.0,
        })
        return snap

    # ── Worker ─────────────────────────────────────────────────────────

    def _worker_loop(self) -> None:
        """Background thread main loop."""
        logger.info("[AsyncQRDetector] Worker loop entering")
        try:
            while not self._stop_event.is_set():
                # Pull next frame from inbox, blocking briefly.
                with self._cond:
                    while self._pending is None and not self._stop_event.is_set():
                        self._cond.wait(timeout=self._WAIT_TIMEOUT_S)
                    if self._stop_event.is_set():
                        break
                    frame, seq, submit_mono = self._pending
                    self._pending = None

                self._process_one(frame, seq, submit_mono)
        except Exception as e:
            # A crash in the worker would silently kill detection; surface it.
            logger.error(
                f"[AsyncQRDetector] Worker loop crashed: {e}", exc_info=True
            )
        finally:
            logger.info("[AsyncQRDetector] Worker loop exiting")

    def _process_one(
        self,
        frame: np.ndarray,
        seq: int,
        submit_mono: float,
    ) -> None:
        """Run motion gate + detection on a single frame and publish result."""
        # Motion gate runs outside the lock to keep the producer wait-free.
        try:
            do_detect, mean_diff = self._motion_gate(frame)
        except Exception as e:
            logger.warning(
                f"[AsyncQRDetector] Motion gate error (treating as motion): {e}"
            )
            do_detect, mean_diff = True, -1.0

        if not do_detect:
            with self._lock:
                self._skipped_motion += 1
            if _ASYNC_QR_DEBUG >= 2:
                logger.debug(
                    f"[AsyncQR] Skipped (motion={mean_diff:.2f} < "
                    f"{self._motion_threshold:.2f}) seq={seq}"
                )
            return

        # Run the (expensive) CNN detection.
        t0 = time.monotonic()
        try:
            detections = self._detector.detect_all(frame)
        except Exception as e:
            with self._lock:
                self._errors += 1
            logger.error(
                f"[AsyncQRDetector] detect_all() failed for seq={seq}: {e}",
                exc_info=True,
            )
            return
        elapsed_ms = (time.monotonic() - t0) * 1000.0

        # Refresh stickiness window so a brief motion lull doesn't gate
        # out a confirmed-visible QR.
        if detections:
            self._force_detect_remaining = self._FORCE_DETECT_FRAMES
        elif self._force_detect_remaining > 0:
            self._force_detect_remaining -= 1

        result = AsyncQRResult(
            detections=detections,
            frame_seq=seq,
            elapsed_ms=elapsed_ms,
            submit_mono=submit_mono,
            done_mono=time.monotonic(),
            motion_diff=mean_diff,
        )

        # Publish + update stats under the lock.
        with self._lock:
            self._processed += 1
            self._total_elapsed_ms += elapsed_ms
            self._last_elapsed_ms = elapsed_ms
            if elapsed_ms > self._max_elapsed_ms:
                self._max_elapsed_ms = elapsed_ms
            if detections:
                self._with_detections += 1
            if self._result is not None:
                # Main loop hasn't polled the previous result.  Overwrite
                # so we always serve freshness, but count it.
                self._dropped_unconsumed += 1
                if _ASYNC_QR_DEBUG:
                    logger.info(
                        f"[AsyncQR] Overwriting unconsumed result "
                        f"old_seq={self._result.frame_seq} new_seq={seq}"
                    )
            self._result = result

        if _ASYNC_QR_DEBUG >= 2 or (_ASYNC_QR_DEBUG and detections):
            logger.info(
                f"[AsyncQR] seq={seq} elapsed={elapsed_ms:.0f}ms "
                f"motion={mean_diff:.2f} found={len(detections)} "
                f"queue_lag={result.queue_lag_ms:.0f}ms"
            )

    # ── Motion gate ────────────────────────────────────────────────────

    def _motion_gate(self, frame: np.ndarray) -> Tuple[bool, float]:
        """Return (run_detect, mean_diff).

        ``run_detect`` is False only when:
            * a non-zero motion threshold is configured, AND
            * we have a previous gate frame to diff against, AND
            * the mean absolute diff is below the threshold, AND
            * we are not inside the post-detection stickiness window.
        """
        # Downsample + grayscale for a cheap, noise-tolerant diff.
        try:
            small = cv2.resize(
                frame, (self._GATE_W, self._GATE_H),
                interpolation=cv2.INTER_AREA,
            )
            if small.ndim == 3:
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            else:
                gray = small
        except cv2.error as e:
            # Bad frame — fall back to forcing a detect; the CNN will
            # decide for itself.
            logger.debug(f"[AsyncQRDetector] resize/cvt failed: {e}")
            return True, -1.0

        prev = self._prev_gate_gray
        self._prev_gate_gray = gray

        # No threshold configured or no baseline yet → always detect.
        if self._motion_threshold <= 0.0 or prev is None or prev.shape != gray.shape:
            return True, -1.0

        # Cheap mean abs diff.  scaleAbs avoids signed overflow.
        diff = cv2.absdiff(gray, prev)
        mean_diff = float(diff.mean())

        if mean_diff >= self._motion_threshold:
            return True, mean_diff
        if self._force_detect_remaining > 0:
            return True, mean_diff
        return False, mean_diff
