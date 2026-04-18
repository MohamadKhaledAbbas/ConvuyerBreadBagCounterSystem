"""
EventFrameBuffer — smooth per-track QR-camera frame ring buffer.

Purpose
=======
The QR (overhead) camera's frames are the fallback video source when the
content camera is unavailable, and the *primary* source when
``event_video_source="qr"``.  Prior to this module, frames were buffered
only on detection ticks (every ``detect_interval``-th frame), so the
resulting clip ran at ~10 fps and looked choppy even when the camera
delivered 30 fps.

Design
======
* **Every frame** from the QR camera is considered for buffering while a
  track is active — independent of detection cadence.  Output video can
  therefore play at the full input fps (clamped to ``target_fps``).
* **Stationary dedup**: if a frame arrives while the tracked QR is
  within ``stationary_px`` pixels of the previously buffered frame, the
  **last** entry is overwritten rather than appending a new one.  This
  handles the edge case of a container that sits still in the frame for
  a long time — the buffer never overflows and the final video keeps a
  continuous "movement-only" history.
* **Hard cap** ``max_seconds * target_fps`` on buffered frames.  Once
  full, the *second-oldest* entry is evicted (preserves the first frame,
  which may be the container's earliest appearance) and the new entry
  is appended.
* **FPS throttle**: if the camera delivers faster than ``target_fps``,
  intermediate frames are dropped by walltime (same algorithm as
  :class:`ContentCameraRecorder`).
* Frames are **half-resolution BGR** by default (memory-friendly) but
  the caller controls resizing — this module only stores what it is
  handed.

Thread safety
=============
Each buffer is owned by a single producer thread (the frame-processing
loop).  Readers that take a snapshot (:meth:`snapshot_frames`) do so
under :attr:`_lock`.  Use separate instances for separate tracks.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Deque, List, Optional, Tuple

import numpy as np


# A ring-buffer entry: (monotonic_seconds, frame, center_x)
FrameEntry = Tuple[float, np.ndarray, int]


@dataclass
class EventFrameBufferConfig:
    """Runtime config for :class:`EventFrameBuffer`."""
    target_fps: float = 20.0            # sampling + output fps
    max_seconds: float = 10.0           # hard cap on buffered history
    stationary_px: int = 0              # <= this X-shift → overwrite last frame (0=disabled)
    # When True, frames arriving faster than target_fps are dropped by
    # walltime (keeps memory bounded with very high fps cameras).
    throttle_to_target_fps: bool = True

    @property
    def max_frames(self) -> int:
        return max(1, int(round(self.max_seconds * self.target_fps)))


class EventFrameBuffer:
    """Per-track rolling frame buffer optimised for smooth event videos.

    Typical usage from the main processing loop::

        buf = EventFrameBuffer(EventFrameBufferConfig(target_fps=20))
        ...
        buf.add(frame, center_x=track.last_x)                # every frame
        ...
        frames_with_ts = buf.snapshot_frames()               # on event
    """

    __slots__ = (
        "_config",
        "_ring",
        "_lock",
        "_last_accept_time",
        "_frame_interval",
        "_total_added",
        "_total_deduped",
        "_total_dropped_fps",
        "_total_evicted",
    )

    def __init__(self, config: Optional[EventFrameBufferConfig] = None):
        self._config = config or EventFrameBufferConfig()
        self._ring: Deque[FrameEntry] = deque(maxlen=self._config.max_frames)
        self._lock = Lock()
        self._last_accept_time: float = 0.0
        self._frame_interval: float = 1.0 / max(1.0, float(self._config.target_fps))
        # Diagnostics
        self._total_added: int = 0
        self._total_deduped: int = 0
        self._total_dropped_fps: int = 0
        self._total_evicted: int = 0

    # ------------------------------------------------------------------
    # Producer API
    # ------------------------------------------------------------------

    def add(
        self,
        frame: np.ndarray,
        center_x: int,
        timestamp: Optional[float] = None,
    ) -> bool:
        """Buffer ``frame`` with the QR centre-x used for stationary dedup.

        Args:
            frame: BGR image (any resolution — caller controls sizing).
            center_x: Current centre-X of the tracked QR on this frame.
            timestamp: Monotonic timestamp; defaults to :func:`time.monotonic`.

        Returns:
            ``True`` if the frame was stored (appended or overwrote the
            tail); ``False`` if it was dropped by the fps throttle.
        """
        if frame is None:
            return False
        now = timestamp if timestamp is not None else time.monotonic()

        # FPS throttle — drop frames that would push us above target_fps.
        if self._config.throttle_to_target_fps and self._last_accept_time:
            if now - self._last_accept_time < self._frame_interval * 0.9:
                self._total_dropped_fps += 1
                return False

        with self._lock:
            # Stationary dedup: overwrite the last entry when the tracked
            # point has barely moved.  This keeps the buffer from filling
            # with near-duplicates while a container is stationary.
            # Disabled when stationary_px <= 0 (e.g. global pre-roll buffer).
            if self._config.stationary_px > 0 and self._ring:
                prev_ts, _, prev_cx = self._ring[-1]
                if abs(center_x - prev_cx) <= self._config.stationary_px:
                    self._ring[-1] = (now, frame, center_x)
                    self._total_deduped += 1
                    self._last_accept_time = now
                    return True

            # Hard cap: deque has maxlen, so oldest is evicted automatically
            # when appending to a full buffer.  Track eviction for diagnostics.
            if len(self._ring) == self._ring.maxlen:
                self._total_evicted += 1

            self._ring.append((now, frame, center_x))
            self._total_added += 1
            self._last_accept_time = now
            return True

    def clear(self) -> None:
        """Empty the buffer (e.g. on track drop)."""
        with self._lock:
            self._ring.clear()
            self._last_accept_time = 0.0

    # ------------------------------------------------------------------
    # Consumer API
    # ------------------------------------------------------------------

    def snapshot_frames(
        self,
        since_monotonic: Optional[float] = None,
    ) -> List[FrameEntry]:
        """Return a list copy of the buffered entries (oldest → newest).

        Args:
            since_monotonic: If provided, only entries with timestamp
                ``>= since_monotonic`` are returned.  Use this to slice
                a precise pre-roll window.
        """
        with self._lock:
            if since_monotonic is None:
                return list(self._ring)
            return [e for e in self._ring if e[0] >= since_monotonic]

    def __len__(self) -> int:
        return len(self._ring)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return buffer stats for logging."""
        with self._lock:
            return {
                "size": len(self._ring),
                "capacity": self._ring.maxlen,
                "added": self._total_added,
                "deduped": self._total_deduped,
                "dropped_fps": self._total_dropped_fps,
                "evicted": self._total_evicted,
            }
