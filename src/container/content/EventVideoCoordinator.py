"""
EventVideoCoordinator — chooses the camera source for an event video.

Responsibilities
================
* Decide, on each container event, whether the video clip should come
  from the QR (overhead) camera or the optional content (3D-angle)
  camera.
* If the preferred source is ``content`` but the content recorder is
  unhealthy, **fall back** transparently to the QR camera so every
  event still gets a clip.
* Encode the chosen clip asynchronously — the main detection loop is
  never blocked on ``cv2.imencode`` or ``cv2.VideoWriter`` I/O.
* Return a small, typed result so the caller can record which camera
  was used (and whether it was a fallback) in the DB metadata.

This module intentionally contains **no** OpenCV import at top level
except for ``cv2.VideoWriter``, and does **no** RTSP work — it delegates
to ``ContentCameraRecorder`` for the content camera.
"""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import Executor
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.utils.AppLogging import logger


# --------------------------------------------------------------------------
# Types
# --------------------------------------------------------------------------

SourcePreference = str  # Literal["qr", "content"] — stringly typed to avoid py<3.8 issues.


@dataclass(frozen=True)
class EventVideoResult:
    """Outcome of :meth:`EventVideoCoordinator.capture`.

    Attributes:
        camera: ``"qr"`` or ``"content"`` — which camera actually holds
            the clip.
        fallback: ``True`` when the preferred camera was ``"content"``
            but capture was routed to the QR camera because the content
            recorder was unavailable (or had no buffered frames).
        video_relpath: Path to the MP4 relative to the repository's
            ``data/`` root (e.g. ``"container_snapshots/qr3_.../video.mp4"``
            or ``"container_content_videos/qr3_....mp4"``).  ``None`` if
            no clip could be produced at all.
    """
    camera: str
    fallback: bool
    video_relpath: Optional[str]


# --------------------------------------------------------------------------
# Coordinator
# --------------------------------------------------------------------------

class EventVideoCoordinator:
    """Dispatch event-video capture to the right source.

    Thread-safe: all mutable state (``_qr_jobs_in_flight``) is accessed
    only from the owner thread when submitting jobs; the encode work
    itself runs in the supplied executor.
    """

    # QR camera fallback video is written here (alongside existing metadata.json).
    _QR_VIDEO_FILENAME = "video.mp4"

    # Legacy JPEG-frames directory name (still written for back-compat preview).
    _QR_FRAMES_DIRNAME = "frames"

    def __init__(
        self,
        *,
        source_preference: SourcePreference,
        content_recorder,                     # Optional[ContentCameraRecorder]
        qr_output_dir: str,                   # e.g. data/container_snapshots
        qr_output_relroot: str,               # relative path (e.g. "container_snapshots")
        content_output_relroot: str,          # relative path (e.g. "container_content_videos")
        qr_fps: float,                        # effective QR-frame sampling fps
        executor: Executor,                   # thread pool for async encoding
        codec: str = "mp4v",
        jpeg_quality: int = 60,
    ):
        self._preference: SourcePreference = (
            source_preference if source_preference in ("qr", "content") else "qr"
        )
        self._content = content_recorder
        self._qr_output_dir = qr_output_dir
        self._qr_output_relroot = qr_output_relroot.strip("/")
        self._content_output_relroot = content_output_relroot.strip("/")
        self._qr_fps = max(1.0, float(qr_fps))
        self._executor = executor
        self._codec = codec
        self._jpeg_quality = int(jpeg_quality)

        os.makedirs(self._qr_output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def source_preference(self) -> str:
        return self._preference

    def capture(
        self,
        *,
        event_id: str,
        trigger_monotonic_time: float,
        qr_frames: List[np.ndarray],
        metadata: dict,
    ) -> EventVideoResult:
        """Capture an event video using the chosen source.

        Args:
            event_id: Unique identifier for this event (used as output name).
            trigger_monotonic_time: ``time.monotonic()`` value when the
                event fired — passed to the content recorder so its
                pre-roll is sliced at the correct walltime.
            qr_frames: List of BGR frames buffered for this track from
                the QR camera.  Used when the QR camera is the chosen
                (or fallback) source.
            metadata: JSON-serialisable dict describing the event.  It
                is written alongside the QR-camera clip and used to
                enrich logs.  The coordinator adds ``camera`` and
                ``fallback`` keys to a shallow copy before writing.

        Returns:
            An :class:`EventVideoResult` describing the outcome.  Never
            raises on the hot path — all errors are logged and converted
            into a ``video_relpath=None`` result.
        """
        # --- Try content recorder if preferred and healthy ---------------
        if self._preference == "content":
            if self._content is not None and self._content.is_available():
                try:
                    fname = self._content.trigger_recording(
                        event_id, trigger_time=trigger_monotonic_time
                    )
                    if fname:
                        rel = f"{self._content_output_relroot}/{fname}"
                        logger.info(
                            f"[EventVideo] event={event_id} source=content "
                            f"file={rel}"
                        )
                        return EventVideoResult(
                            camera="content", fallback=False, video_relpath=rel
                        )
                    logger.warning(
                        f"[EventVideo] content recorder returned None for {event_id}"
                    )
                except Exception as e:
                    logger.error(
                        f"[EventVideo] content recorder raised for {event_id}: {e}",
                        exc_info=True,
                    )
            else:
                reason = (
                    "disabled" if self._content is None else "not available"
                )
                logger.warning(
                    f"[EventVideo] content camera {reason} for {event_id} "
                    f"— falling back to QR camera"
                )
            # Reaching here means we wanted content but couldn't get it.
            return self._submit_qr_video(
                event_id=event_id, frames=qr_frames, metadata=metadata, fallback=True,
            )

        # --- Preference = qr (or invalid) --------------------------------
        return self._submit_qr_video(
            event_id=event_id, frames=qr_frames, metadata=metadata, fallback=False,
        )

    # ------------------------------------------------------------------
    # QR-camera encoding path
    # ------------------------------------------------------------------

    def _submit_qr_video(
        self,
        *,
        event_id: str,
        frames: List[np.ndarray],
        metadata: dict,
        fallback: bool,
    ) -> EventVideoResult:
        if not frames:
            logger.warning(
                f"[EventVideo] event={event_id} source=qr — no QR frames buffered"
            )
            # Still write metadata so the DB row has something to point at.
            self._submit_qr_metadata_only(event_id=event_id, metadata=metadata,
                                          camera="qr", fallback=fallback)
            return EventVideoResult(camera="qr", fallback=fallback, video_relpath=None)

        event_dir = os.path.join(self._qr_output_dir, event_id)
        rel = f"{self._qr_output_relroot}/{event_id}/{self._QR_VIDEO_FILENAME}"

        # Enrich metadata with the chosen source — this is what the UI reads.
        full_meta = dict(metadata)
        full_meta["camera"] = "qr"
        full_meta["fallback"] = bool(fallback)
        full_meta["video_file"] = self._QR_VIDEO_FILENAME

        codec = self._codec
        fps = self._qr_fps

        def _encode_job() -> None:
            try:
                os.makedirs(event_dir, exist_ok=True)
                self._write_qr_video(event_dir, frames, fps, codec)
                self._write_metadata(event_dir, full_meta)
            except Exception as e:  # pragma: no cover — diagnostic path
                logger.error(
                    f"[EventVideo] QR encode job failed for {event_id}: {e}",
                    exc_info=True,
                )

        self._executor.submit(_encode_job)
        logger.info(
            f"[EventVideo] event={event_id} source=qr fallback={fallback} "
            f"queued frames={len(frames)} fps={fps:.1f} -> {rel}"
        )
        return EventVideoResult(camera="qr", fallback=fallback, video_relpath=rel)

    def _submit_qr_metadata_only(
        self,
        *,
        event_id: str,
        metadata: dict,
        camera: str,
        fallback: bool,
    ) -> None:
        """Write just a metadata.json for events that have no frames."""
        event_dir = os.path.join(self._qr_output_dir, event_id)
        full_meta = dict(metadata)
        full_meta["camera"] = camera
        full_meta["fallback"] = bool(fallback)
        full_meta["video_file"] = None

        def _meta_job() -> None:
            try:
                os.makedirs(event_dir, exist_ok=True)
                self._write_metadata(event_dir, full_meta)
            except Exception as e:  # pragma: no cover
                logger.error(f"[EventVideo] metadata-only write failed: {e}")

        self._executor.submit(_meta_job)

    # ------------------------------------------------------------------
    # Low-level writers (run inside executor threads)
    # ------------------------------------------------------------------

    @classmethod
    def _write_qr_video(
        cls,
        event_dir: str,
        frames: List[np.ndarray],
        fps: float,
        codec: str,  # kept for API compat; ignored when ffmpeg path is used
    ) -> None:
        """Encode ``frames`` to ``{event_dir}/video.mp4`` as H.264.

        Uses ffmpeg (libx264 + yuv420p) when available — the output is
        browser-compatible.  Falls back to OpenCV VideoWriter with the
        supplied *codec* if ffmpeg is not found on PATH.
        """
        if not frames:
            return
        h, w = frames[0].shape[:2]
        final_path = os.path.join(event_dir, cls._QR_VIDEO_FILENAME)
        tmp_path = final_path + ".writing.mp4"

        try:
            cls._write_qr_video_ffmpeg(frames, tmp_path, fps, w, h)
        except Exception as e:
            logger.warning(
                f"[EventVideo] ffmpeg encode failed ({e}) — falling back to OpenCV"
            )
            cls._write_qr_video_opencv(frames, tmp_path, fps, codec, w, h)

        if not os.path.exists(tmp_path):
            logger.error(f"[EventVideo] No output produced for {final_path}")
            return

        try:
            os.replace(tmp_path, final_path)
        except OSError as e:
            logger.error(f"[EventVideo] rename {tmp_path} -> {final_path} failed: {e}")
            return

        logger.info(
            f"[EventVideo] Wrote {final_path} frames={len(frames)} size={w}x{h}"
        )

    @staticmethod
    def _write_qr_video_ffmpeg(
        frames: List[np.ndarray],
        out_path: str,
        fps: float,
        w: int,
        h: int,
    ) -> None:
        """Pipe raw BGR frames into ffmpeg → H.264 MP4 (browser-compatible)."""
        import shutil
        import subprocess

        ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin:
            raise FileNotFoundError("ffmpeg not found on PATH")

        cmd = [
            ffmpeg_bin,
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{w}x{h}",
            "-pix_fmt", "bgr24",
            "-r", str(fps),
            "-i", "pipe:0",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",   # required for broad browser support
            "-movflags", "+faststart",  # move moov atom to front for streaming
            out_path,
        ]
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        try:
            for f in frames:
                if f.shape[1] != w or f.shape[0] != h:
                    f = cv2.resize(f, (w, h))
                proc.stdin.write(f.tobytes())
            proc.stdin.close()
        except BrokenPipeError:
            pass
        proc.wait(timeout=60)
        if proc.returncode != 0:
            stderr = proc.stderr.read().decode(errors="replace")[-400:]
            raise RuntimeError(f"ffmpeg exited {proc.returncode}: {stderr}")

    @staticmethod
    def _write_qr_video_opencv(
        frames: List[np.ndarray],
        out_path: str,
        fps: float,
        codec: str,
        w: int,
        h: int,
    ) -> None:
        """OpenCV VideoWriter fallback (mp4v — limited browser support)."""
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(out_path, fourcc, float(fps), (w, h))
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

    @staticmethod
    def _write_metadata(event_dir: str, metadata: dict) -> None:
        import json
        path = os.path.join(event_dir, "metadata.json")
        tmp = path + ".part"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)
        os.replace(tmp, path)
