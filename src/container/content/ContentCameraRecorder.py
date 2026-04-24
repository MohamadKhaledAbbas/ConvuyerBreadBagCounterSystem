"""
ContentCameraRecorder — rolling-buffer + event-triggered MP4 recorder.

Purpose
=======
A second IP camera (default ``192.168.2.128``) is mounted at a 3D angle so
it can see the **contents of each container** as it passes through the
sale point.  When the QR tracker on the overhead camera fires an
event (container fully exited), we want a short video of *what was
inside it* — ideally starting a few seconds *before* the event, because
the container starts entering the content-camera's field of view before
its QR reaches the overhead camera's exit zone.

Design
======
* A dedicated **reader thread** pulls frames from the RTSP stream at a
  steady target FPS.  Each frame is pushed into a thread-safe rolling
  ``deque`` sized to ``buffer_seconds * fps`` so the last N seconds of
  video are always available in memory.
* When the main application calls :meth:`trigger_recording`, the recorder:

  1. Snapshots the current ring buffer (last ``pre_event_seconds`` of
     frames).
  2. Schedules itself to keep capturing for ``post_event_seconds`` more.
  3. A dedicated **writer worker thread** drains a queue of pending
     recordings and encodes each one to ``.mp4`` using
     ``cv2.VideoWriter``.  The main app is never blocked.

* Recording stops cleanly on :meth:`stop` — reader thread joined, writer
  worker drains pending jobs, RTSP capture released.
* **Auto-reconnect**: if the RTSP read fails, the reader backs off and
  retries, logging the state.  During a disconnect the ring buffer
  simply stops growing; old frames are still available for triggers.

Output
======
``{CONTAINER_CONTENT_VIDEOS_DIR}/{event_id}.mp4``

Thread safety
=============
The ring buffer is a ``collections.deque`` protected by ``self._lock``.
The writer queue is a ``queue.Queue``.  Public methods (:meth:`start`,
:meth:`stop`, :meth:`trigger_recording`) are safe to call from any
thread.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from queue import Queue, Empty
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np

from src.utils.AppLogging import logger
from src.container.content.SyncDebugOverlay import draw_sync_debug_overlay

# A ring-buffer entry: (monotonic_timestamp_seconds, BGR frame).
_FrameEntry = Tuple[float, np.ndarray]


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

@dataclass
class ContentRecorderConfig:
    """Runtime configuration for :class:`ContentCameraRecorder`."""

    rtsp_url: str                       # full RTSP URL (credentials embedded)
    output_dir: str                     # directory for ``{event_id}.mp4``
    buffer_seconds: float = 5.0         # ring-buffer capacity in seconds
    pre_event_seconds: float = 3.0      # seconds of pre-roll to include
    post_event_seconds: float = 2.0     # seconds to keep capturing after trigger
    target_fps: int = 15                # reader + output FPS
    reconnect_delay: float = 2.0        # seconds between RTSP reconnect attempts
    max_recording_seconds: float = 15.0 # safety cap: auto-finalize any recording longer than this
    frame_size: Optional[Tuple[int, int]] = None   # (w, h); detected from stream
    codec: str = "mp4v"                 # four-cc for ``cv2.VideoWriter``
    debug_sync_overlay: bool = False    # dev-only overlay for saved content clips

    # Masked representation of the URL for logging.
    @property
    def rtsp_url_masked(self) -> str:
        if "://" not in self.rtsp_url or "@" not in self.rtsp_url:
            return self.rtsp_url
        scheme, rest = self.rtsp_url.split("://", 1)
        creds, host = rest.split("@", 1)
        if ":" in creds:
            user = creds.split(":", 1)[0]
            return f"{scheme}://{user}:***@{host}"
        return self.rtsp_url


# --------------------------------------------------------------------------
# Pending recording job
# --------------------------------------------------------------------------

@dataclass
class _PendingRecording:
    """A recording queued for the writer worker.

    ``pre_frames`` and ``post_frames`` both hold ``_FrameEntry`` tuples
    (monotonic timestamp + frame) so the final write can interleave them
    by true capture time rather than insertion order.
    """
    event_id: str
    trigger_time: float                 # monotonic time the event was triggered
    sync_anchor_time: Optional[float]   # shared anchor used by QR/content debug overlays
    pre_frames: List[_FrameEntry]       # snapshot of ring buffer within [trigger - pre, trigger]
    capture_until: float                # monotonic deadline for post-roll
    post_frames: List[_FrameEntry]      # filled by reader thread within (trigger, trigger + post]
    done_event: threading.Event         # set when post-roll complete


# --------------------------------------------------------------------------
# Recorder
# --------------------------------------------------------------------------

class ContentCameraRecorder:
    """Background RTSP reader with a rolling ring buffer and event-triggered
    MP4 writer.

    Typical lifecycle::

        rec = ContentCameraRecorder(config)
        rec.start()
        ...
        rec.trigger_recording(event_id)   # returns immediately
        ...
        rec.stop()
    """

    # Log the reader's health every N seconds.
    _HEALTH_LOG_INTERVAL = 30.0

    def __init__(self, config: ContentRecorderConfig):
        self.config = config
        self._buffer_size = max(1, int(config.buffer_seconds * config.target_fps))
        self._pre_event_frames = max(1, int(config.pre_event_seconds * config.target_fps))
        self._post_event_frames = max(1, int(config.post_event_seconds * config.target_fps))

        # Timestamped ring buffer — each entry is (monotonic_seconds, frame).
        # Keeping timestamps lets us slice a pre-roll of exactly
        # ``pre_event_seconds`` regardless of actual fps drift, RTSP
        # stalls, or reconnects.
        self._ring: Deque[_FrameEntry] = deque(maxlen=self._buffer_size)
        self._lock = threading.Lock()

        # Consider the recorder "healthy / available" if a frame arrived
        # within this many seconds.  Used by callers deciding whether to
        # trust the recorder over the fallback path.
        self._healthy_max_age: float = max(1.0, 2.0 / max(1, config.target_fps) * 6.0 + 1.0)

        self._reader_thread: Optional[threading.Thread] = None
        self._writer_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # In-flight recordings that still need post-roll frames appended.
        self._active: List[_PendingRecording] = []
        # Jobs that are done collecting frames and ready for disk encoding.
        self._write_queue: "Queue[_PendingRecording]" = Queue()

        # Reader health counters.
        self._last_frame_time: float = 0.0
        self._connected: bool = False
        self._total_frames_read: int = 0
        self._total_reconnects: int = 0

        os.makedirs(self.config.output_dir, exist_ok=True)

    # ----- lifecycle ------------------------------------------------------

    def start(self) -> None:
        """Start the reader and writer threads."""
        if self._reader_thread is not None:
            logger.warning("[ContentRecorder] start() called but already running")
            return
        self._stop_event.clear()
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            name="content-reader",
            daemon=True,
        )
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name="content-writer",
            daemon=True,
        )
        self._reader_thread.start()
        self._writer_thread.start()
        logger.info(
            f"[ContentRecorder] Started. url={self.config.rtsp_url_masked} "
            f"buffer={self.config.buffer_seconds}s pre={self.config.pre_event_seconds}s "
            f"post={self.config.post_event_seconds}s fps={self.config.target_fps}"
        )

    def stop(self, timeout: float = 10.0) -> None:
        """Stop both threads and drain remaining recordings.

        The writer is given up to ``timeout`` seconds to finish any
        already-queued encodings — in-flight post-roll captures are
        truncated and flushed with whatever frames were collected.
        """
        logger.info("[ContentRecorder] Stopping...")
        self._stop_event.set()
        # Flush all active post-roll recordings immediately — the reader
        # won't be able to feed them any more frames.
        with self._lock:
            for rec in list(self._active):
                self._write_queue.put(rec)
            self._active.clear()
        # Wait for threads.
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=timeout)
            self._reader_thread = None
        # Signal writer to drain.
        self._write_queue.put(None)  # type: ignore[arg-type]
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=timeout)
            self._writer_thread = None
        logger.info("[ContentRecorder] Stopped.")

    # ----- public API -----------------------------------------------------

    def is_available(self) -> bool:
        """Best-effort health check: recorder running and receiving frames.

        Used by callers that may prefer this recorder but need to fall
        back to another source when the RTSP stream is down.
        """
        if self._reader_thread is None or self._stop_event.is_set():
            return False
        if not self._connected:
            return False
        if self._last_frame_time == 0.0:
            return False
        age = time.monotonic() - self._last_frame_time
        return age < self._healthy_max_age

    def trigger_recording(
        self,
        event_id: str,
        trigger_time: Optional[float] = None,
    ) -> Optional[str]:
        """Trigger a content recording for the given event.

        The ``trigger_time`` argument is the container's **entry** moment
        (``event.entry_time_monotonic``), which may be in the past by the
        time this method is called (the event fires at container exit).

        Pre-roll:  ``[trigger_time - pre_event_seconds,  NOW]``
            Captures from ``pre_event_seconds`` before the container
            entered all the way through to the exit (which is NOW), so the
            full transit is always included.

        Post-roll: ``post_event_seconds`` seconds *after* this call,
            giving a brief tail of the conveyor after the container leaves.

        Args:
            event_id: Identifier used for the output filename.
            trigger_time: Monotonic-clock anchor for the pre-roll start
                (typically the container's entry time).  Defaults to
                ``time.monotonic()`` when omitted.
        """
        if self._reader_thread is None or self._stop_event.is_set():
            logger.warning(
                f"[ContentRecorder] trigger_recording({event_id}) ignored — not running"
            )
            return None

        # t_anchor: the start reference (entry time or "now" when unknown)
        t_anchor = trigger_time if trigger_time is not None else time.monotonic()
        # t_now: the actual call time (= approximately the exit moment)
        t_now = time.monotonic()

        # Pre-roll window spans from pre_event_seconds before the anchor
        # up to the CURRENT time so the entire transit is included,
        # even when t_anchor is several seconds in the past.
        pre_cutoff = t_anchor - self.config.pre_event_seconds
        pre_end = max(t_anchor, t_now)   # safety for immediate/same-tick calls

        with self._lock:
            if len(self._ring) == 0:
                logger.warning(
                    f"[ContentRecorder] trigger_recording({event_id}) — ring buffer empty"
                )
            pre: List[_FrameEntry] = [
                entry for entry in self._ring
                if pre_cutoff <= entry[0] <= pre_end
            ]
            rec = _PendingRecording(
                event_id=event_id,
                # Reader-loop gating: frames arriving strictly after t_now
                # are treated as post-roll.  Frames already in the ring
                # buffer ([t_anchor, t_now]) are already in pre_frames.
                trigger_time=t_now,
                sync_anchor_time=t_anchor,
                pre_frames=pre,
                capture_until=t_now + self.config.post_event_seconds,
                post_frames=[],
                done_event=threading.Event(),
            )
            self._active.append(rec)

        span_s = (pre[-1][0] - pre[0][0]) if len(pre) >= 2 else 0.0
        logger.info(
            f"[ContentRecorder] Triggered event={event_id} "
            f"pre_frames={len(pre)} (~{span_s:.2f}s) "
            f"post_seconds={self.config.post_event_seconds}"
        )
        return f"{event_id}.mp4"

    # ----- begin / end model (start-of-transit) --------------------------

    def begin_event_recording(
        self,
        event_id: str,
        trigger_time: Optional[float] = None,
    ) -> None:
        """Start continuous recording for an ongoing container transit.

        Called when the QR camera's detection-count threshold is first
        met (i.e. mid-transit, not at exit).

        Pre-roll: snapshots from the ring buffer going back
        ``pre_event_seconds`` before *trigger_time*.

        Active capture: the reader thread keeps appending frames until
        :meth:`end_event_recording` is called **or**
        ``max_recording_seconds`` elapses (safety cap), whichever comes
        first.

        Args:
            event_id: Identifier used for the output filename.
            trigger_time: Monotonic-clock anchor (container entry time).
                Defaults to ``time.monotonic()``.
        """
        if self._reader_thread is None or self._stop_event.is_set():
            logger.warning(
                f"[ContentRecorder] begin_event_recording({event_id}) ignored — not running"
            )
            return

        t_anchor = trigger_time if trigger_time is not None else time.monotonic()
        t_now = time.monotonic()
        pre_cutoff = t_anchor - self.config.pre_event_seconds
        pre_end = max(t_anchor, t_now)

        with self._lock:
            # Avoid duplicate begin calls for the same event
            for rec in self._active:
                if rec.event_id == event_id:
                    logger.debug(
                        f"[ContentRecorder] begin_event_recording({event_id}) — already active"
                    )
                    return

            pre: List[_FrameEntry] = [
                entry for entry in self._ring
                if pre_cutoff <= entry[0] <= pre_end
            ]
            rec = _PendingRecording(
                event_id=event_id,
                trigger_time=t_now,
                sync_anchor_time=t_anchor,
                pre_frames=pre,
                # Safety cap — auto-finalized by the reader loop if
                # end_event_recording is never called (e.g. crash, bug).
                capture_until=t_now + self.config.max_recording_seconds,
                post_frames=[],
                done_event=threading.Event(),
            )
            self._active.append(rec)

        span_s = (pre[-1][0] - pre[0][0]) if len(pre) >= 2 else 0.0
        logger.info(
            f"[ContentRecorder] BEGIN event={event_id} "
            f"pre_frames={len(pre)} (~{span_s:.2f}s) "
            f"max_cap={self.config.max_recording_seconds}s"
        )

    def end_event_recording(self, event_id: str) -> None:
        """End an active recording started by :meth:`begin_event_recording`.

        Sets ``capture_until`` to ``now + post_event_seconds`` so the
        recording captures a brief tail after the container exits.
        If the recording already auto-finalized (max cap), this is a
        harmless no-op.
        """
        now = time.monotonic()
        with self._lock:
            for rec in self._active:
                if rec.event_id == event_id:
                    rec.capture_until = now + self.config.post_event_seconds
                    logger.info(
                        f"[ContentRecorder] END event={event_id} "
                        f"post_seconds={self.config.post_event_seconds} "
                        f"total_frames={len(rec.pre_frames) + len(rec.post_frames)}"
                    )
                    return
        # Not found — likely already auto-finalized by the max-cap.
        logger.debug(
            f"[ContentRecorder] end_event_recording({event_id}) — "
            f"not found (already finalized?)"
        )

    def get_health(self) -> dict:
        """Return diagnostic info for health monitoring."""
        now = time.monotonic()
        age = (now - self._last_frame_time) if self._last_frame_time else float("inf")
        with self._lock:
            buf = len(self._ring)
            active = len(self._active)
        return {
            "connected": self._connected,
            "last_frame_age_seconds": None if age == float("inf") else round(age, 2),
            "buffer_frames": buf,
            "buffer_capacity": self._buffer_size,
            "active_recordings": active,
            "total_frames_read": self._total_frames_read,
            "total_reconnects": self._total_reconnects,
        }

    # ----- reader thread --------------------------------------------------

    def _reader_loop(self) -> None:
        """Main reader loop — pulls frames, feeds ring buffer + active recordings."""
        frame_interval = 1.0 / max(1, self.config.target_fps)
        next_health_log = time.monotonic() + self._HEALTH_LOG_INTERVAL

        while not self._stop_event.is_set():
            cap = self._open_capture()
            if cap is None:
                # open_capture handles its own back-off; check stop flag.
                continue

            self._connected = True
            # Detect the actual stream frame size once.
            if self.config.frame_size is None:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                if w > 0 and h > 0:
                    self.config.frame_size = (w, h)

            last_read = time.monotonic()
            while not self._stop_event.is_set():
                ok, frame = cap.read()
                now = time.monotonic()

                if not ok or frame is None:
                    logger.warning(
                        "[ContentRecorder] read() returned no frame — reconnecting"
                    )
                    break

                # Drop frames if the camera is faster than our target FPS.
                if now - last_read < frame_interval * 0.9:
                    continue
                last_read = now

                self._last_frame_time = now
                self._total_frames_read += 1

                # Lazily learn the frame size from actual frames.
                if self.config.frame_size is None:
                    self.config.frame_size = (frame.shape[1], frame.shape[0])

                entry: _FrameEntry = (now, frame)
                with self._lock:
                    self._ring.append(entry)
                    # Feed any active post-roll recordings — only frames
                    # strictly after the trigger time (pre-roll already
                    # captured everything up to trigger_time) and no
                    # later than capture_until.
                    for rec in list(self._active):
                        if now <= rec.trigger_time:
                            continue
                        if now <= rec.capture_until:
                            rec.post_frames.append(entry)
                        else:
                            # Post-roll complete — queue for writing.
                            self._active.remove(rec)
                            self._write_queue.put(rec)

                # Periodic health log.
                if now >= next_health_log:
                    h = self.get_health()
                    logger.info(f"[ContentRecorder] health={h}")
                    next_health_log = now + self._HEALTH_LOG_INTERVAL

            self._connected = False
            try:
                cap.release()
            except Exception as e:
                logger.debug(f"[ContentRecorder] cap.release() failed: {e}")

            if not self._stop_event.is_set():
                self._total_reconnects += 1
                logger.info(
                    f"[ContentRecorder] Reconnecting in {self.config.reconnect_delay}s "
                    f"(total_reconnects={self._total_reconnects})"
                )
                self._stop_event.wait(self.config.reconnect_delay)

        # Exiting: ensure any active recordings are flushed.
        with self._lock:
            for rec in list(self._active):
                self._write_queue.put(rec)
            self._active.clear()

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        """Open the RTSP capture, backing off on failure."""
        # Prefer TCP transport for RTSP when supported (reduces corruption).
        os.environ.setdefault(
            "OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp"
        )
        try:
            cap = cv2.VideoCapture(self.config.rtsp_url, cv2.CAP_FFMPEG)
        except Exception as e:
            logger.error(f"[ContentRecorder] VideoCapture() threw: {e}")
            cap = None

        if cap is None or not cap.isOpened():
            logger.warning(
                f"[ContentRecorder] Failed to open {self.config.rtsp_url_masked} "
                f"— retrying in {self.config.reconnect_delay}s"
            )
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            self._stop_event.wait(self.config.reconnect_delay)
            return None

        # Small receive buffer — we want the freshest frame, not a backlog.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        logger.info(
            f"[ContentRecorder] Connected to {self.config.rtsp_url_masked}"
        )
        return cap

    # ----- writer thread --------------------------------------------------

    def _writer_loop(self) -> None:
        """Drain the write queue, encoding each pending recording to MP4."""
        while True:
            try:
                rec = self._write_queue.get(timeout=1.0)
            except Empty:
                if self._stop_event.is_set() and self._reader_thread is None:
                    break
                continue
            if rec is None:
                break
            try:
                self._write_recording(rec)
            except Exception as e:
                logger.error(
                    f"[ContentRecorder] Failed to write {rec.event_id}: {e}",
                    exc_info=True,
                )
            finally:
                rec.done_event.set()

    def _write_recording(self, rec: _PendingRecording) -> None:
        # Merge pre and post entries, sorted by timestamp to guarantee
        # monotonic playback even if the reader added frames to an
        # active recording concurrently with the trigger.
        entries = sorted(rec.pre_frames + rec.post_frames, key=lambda e: e[0])
        if not entries:
            logger.warning(
                f"[ContentRecorder] Recording {rec.event_id} has no frames — skipping"
            )
            return

        # Derive size from the first frame (handles cameras whose reported
        # size differs from the decoded frame shape).
        first_frame = entries[0][1]
        h, w = first_frame.shape[:2]
        path = os.path.join(self.config.output_dir, f"{rec.event_id}.mp4")
        tmp_path = path + ".writing.mp4"
        if self.config.debug_sync_overlay:
            anchor_mono = rec.sync_anchor_time if rec.sync_anchor_time is not None else entries[0][0]
            bare_frames = [
                draw_sync_debug_overlay(
                    frame,
                    event_id=rec.event_id,
                    camera="content",
                    capture_monotonic=ts,
                    anchor_monotonic=anchor_mono,
                )
                for ts, frame in entries
            ]
        else:
            bare_frames = [f for _, f in entries]

        # Use the actual measured sampling rate so the video plays at
        # real-time speed even when the camera delivers fewer fps than
        # target_fps (e.g. 720p main stream at 7fps vs target of 20fps).
        # Without this, each frame would be displayed for only 1/20 s
        # instead of the ~1/7 s it actually represents, making the clip
        # run 3× too fast and appear artificially short.
        if len(entries) >= 2:
            span = entries[-1][0] - entries[0][0]
            measured_fps = len(entries) / span if span > 0 else float(self.config.target_fps)
            # Clamp: never below 1 fps, never above target_fps
            write_fps = max(1.0, min(float(self.config.target_fps), measured_fps))
        else:
            write_fps = float(self.config.target_fps)

        try:
            self._write_h264_ffmpeg(bare_frames, tmp_path, write_fps, w, h)
        except Exception as e:
            logger.warning(
                f"[ContentRecorder] ffmpeg encode failed ({e}) — falling back to OpenCV"
            )
            self._write_opencv_fallback(bare_frames, tmp_path, self.config.codec, write_fps, w, h)

        if not os.path.exists(tmp_path):
            logger.error(f"[ContentRecorder] No output produced for {path}")
            return

        try:
            os.replace(tmp_path, path)
        except OSError as e:
            logger.error(f"[ContentRecorder] Rename {tmp_path} -> {path} failed: {e}")
            return

        span = entries[-1][0] - entries[0][0]
        logger.info(
            f"[ContentRecorder] Wrote {path} frames={len(entries)} "
            f"pre={len(rec.pre_frames)} post={len(rec.post_frames)} "
            f"span={span:.2f}s fps={write_fps:.1f} size={w}x{h}"
        )

    @staticmethod
    def _write_h264_ffmpeg(
        frames: list,
        out_path: str,
        fps: float,
        w: int,
        h: int,
    ) -> None:
        import shutil, subprocess
        ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin:
            raise FileNotFoundError("ffmpeg not found on PATH")
        cmd = [
            ffmpeg_bin, "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{w}x{h}", "-pix_fmt", "bgr24", "-r", str(fps),
            "-i", "pipe:0",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            out_path,
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            for f in frames:
                if f.shape[1] != w or f.shape[0] != h:
                    f = cv2.resize(f, (w, h))
                proc.stdin.write(f.tobytes())
            proc.stdin.close()
        except BrokenPipeError:
            pass
        proc.wait(timeout=120)
        if proc.returncode != 0:
            stderr = proc.stderr.read().decode(errors="replace")[-400:]
            raise RuntimeError(f"ffmpeg exited {proc.returncode}: {stderr}")

    @staticmethod
    def _write_opencv_fallback(
        frames: list,
        out_path: str,
        codec: str,
        fps: float,
        w: int,
        h: int,
    ) -> None:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError(
                f"VideoWriter failed to open {out_path} (codec={codec} {w}x{h})"
            )
        try:
            for f in frames:
                if f.shape[1] != w or f.shape[0] != h:
                    f = cv2.resize(f, (w, h))
                writer.write(f)
        finally:
            writer.release()

    # ----- context manager ------------------------------------------------

    def __enter__(self) -> "ContentCameraRecorder":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()
