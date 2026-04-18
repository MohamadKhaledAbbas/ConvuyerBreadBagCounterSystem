"""
Container Counter Application for Sale Point (صالة) monitoring.

Main orchestrator for the container tracking pipeline:
1. Receives frames from container camera (via ROS2 or OpenCV)
2. Detects QR codes using OpenCV QRCodeDetector
3. Tracks containers by QR code value (1-5)
4. Determines direction (positive/negative)
5. Triggers snapshot capture on container exit
6. Records events to database
7. Publishes state for health monitoring

Direction Rules:
- Positive (bottom→top): Filled container leaving → INCREMENT count
- Negative (top→bottom): Empty container returning → LOG only

Usage:
    app = ContainerCounterApp(config)
    app.run()
"""

import json
import os
import queue
import signal
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2

from src.container.qr.QRCodeDetector import QRCodeDetector, QRDetection
from src.container.tracking.ContainerTracker import (
    ContainerTracker,
    ContainerEvent,
    Direction,
)
from src.container.snapshot.RingBufferSnapshotter import RingBufferSnapshotter
from src.container.ContainerVisualizer import ContainerVisualizer
from src.endpoint.routes.snapshot import SnapshotWriter
from src.logging.Database import DatabaseManager
from src.utils.AppLogging import logger
from src.utils.platform import IS_RDK, PLATFORM_NAME
import src.constants as constants

# Set PROCESS_DEBUG=1 to log per-step timing inside _process_frame.
# Set PROCESS_DEBUG=2 for every-frame logging (very chatty).
_PROCESS_DEBUG = int(os.environ.get('PROCESS_DEBUG', '0') or '0')


class _PredictedDetection:
    """Lightweight stand-in for QRDetection used on tracker-predicted frames."""

    def __init__(self, qr_number: int, center: tuple, bbox_xywh: tuple):
        self.qr_number = qr_number
        self.value = str(qr_number)
        self.center = center
        x, y, w, h = bbox_xywh
        import numpy as _np
        self.bbox = _np.array([
            [x, y], [x + w, y], [x + w, y + h], [x, y + h]
        ], dtype=_np.int32)
        self.area = w * h
        self.confidence = 0.0   # flag as predicted


class _TrackEntry:
    """Per-QR predictor state with EMA-smoothed X-velocity (horizontal tracking)."""

    __slots__ = ('cx', 'cy', 'vx', 'bw', 'bh', 'age', 'frame_w', 'frame_h')

    # EMA smoothing factor for velocity (0 = never update, 1 = no smoothing)
    _ALPHA = 0.4

    def __init__(self, cx: float, cy: float, bw: int, bh: int,
                 frame_w: int, frame_h: int):
        self.cx = float(cx)
        self.cy = float(cy)
        self.vx = 0.0
        self.bw = bw
        self.bh = bh
        self.age = 0          # frames since last real detection
        self.frame_w = frame_w
        self.frame_h = frame_h

    def update(self, cx: float, cy: float, vx_raw: float,
               bw: int, bh: int) -> None:
        """Absorb a new real detection; smooth X-velocity with EMA.

        Y is tracked passively (updated to last observed value) but not
        extrapolated since container motion is horizontal.
        """
        self.vx = self._ALPHA * vx_raw + (1 - self._ALPHA) * self.vx
        self.cx = cx
        self.cy = cy
        self.bw = bw
        self.bh = bh
        self.age = 0

    def advance(self, step: float) -> Optional[tuple]:
        """
        Project X position `step` frames ahead (Y held constant).

        Returns (cx, cy, bbox_xywh) clamped to frame bounds,
        or None if the projected position is outside the frame.
        """
        px = self.cx + self.vx * step
        py = self.cy

        # Suppress prediction if projected centre leaves the frame horizontally
        margin = self.bw // 2
        if px < -margin or px > self.frame_w + margin:
            return None

        # Clamp to frame
        px = max(0, min(self.frame_w, px))
        x0 = int(px - self.bw // 2)
        y0 = int(py - self.bh // 2)
        return (int(px), int(py), (x0, y0, self.bw, self.bh))


class _LinearPredictor:
    """
    Production-grade per-QR position predictor.

    Uses EMA-smoothed velocity, enforces a maximum prediction age, and
    suppresses predictions whose projected centre leaves the frame.

    Design goals:
    - No jumps: velocity is exponentially smoothed over detections
    - No ghosts: predictions expire after `max_pred_frames` missed frames
    - No out-of-frame predictions: projected centre is bounds-checked
    - Minimal memory: O(N) where N = number of distinct QR values (≤ 5)
    """

    # Max consecutive predict() calls without a real detection before dropping.
    # At detect_interval=3 this gives 4 calls ≈ 2 missed detect cycles ≈ 0.3 s @20 fps.
    # Kept small deliberately: a real exit means the QR leaves the decoder's view
    # immediately; generous prediction budgets only produce ghost overlays.
    MAX_PRED_FRAMES = 4

    def __init__(self, detect_interval: int = 5,
                 frame_w: int = 1280, frame_h: int = 720):
        self._interval = detect_interval
        self._frame_w = frame_w
        self._frame_h = frame_h
        self._entries: Dict[int, _TrackEntry] = {}

    def set_frame_size(self, w: int, h: int) -> None:
        """Notify predictor of frame size (call when first frame arrives)."""
        self._frame_w = w
        self._frame_h = h
        for e in self._entries.values():
            e.frame_w = w
            e.frame_h = h

    def update_from_detection(self, qr_val: int, center: tuple, bbox_wh: tuple) -> None:
        """
        Feed a confirmed real detection.

        Computes per-frame velocity from position delta since last detection,
        then smooths it with EMA to suppress noise spikes.
        """
        cx, cy = float(center[0]), float(center[1])
        bw, bh = bbox_wh

        existing = self._entries.get(qr_val)
        if existing is None:
            entry = _TrackEntry(cx, cy, bw, bh, self._frame_w, self._frame_h)
            self._entries[qr_val] = entry
            return

        # Raw per-frame X-velocity (displacement ÷ detect_interval frames)
        elapsed_frames = max(1, self._interval + existing.age)
        vx_raw = (cx - existing.cx) / elapsed_frames

        existing.update(cx, cy, vx_raw, bw, bh)

    def predict(self, step: int = 1) -> list:
        """
        Return predictions for all entries that are still within their age limit.

        step: how many frames since the last detection (1 .. detect_interval-1).
        Returns list of (qr_val, (cx, cy), (x0, y0, bw, bh)).
        Entries whose projected centre leaves the frame are silently skipped.
        """
        results = []
        expired = []
        for qr_val, entry in list(self._entries.items()):
            entry.age += 1
            if entry.age > self.MAX_PRED_FRAMES:
                expired.append(qr_val)
                continue
            proj = entry.advance(step)
            if proj is None:
                # Projected outside frame → container has left the field of view.
                # Remove immediately so downstream consumers see no ghost overlay.
                expired.append(qr_val)
                logger.debug(
                    f"[Predictor] QR{qr_val} projected outside frame at step {step}, removing"
                )
                continue
            cx, cy, bbox_xywh = proj
            results.append((qr_val, (cx, cy), bbox_xywh))
        for qr_val in expired:
            logger.debug(f"[Predictor] QR{qr_val} prediction expired (age > {self.MAX_PRED_FRAMES})")
            del self._entries[qr_val]
        return results

    def remove(self, qr_val: int) -> None:
        """Remove a QR entry (call on track exit to stop ghost predictions)."""
        self._entries.pop(qr_val, None)

    def clear(self) -> None:
        self._entries.clear()

    @property
    def tracked_qr_values(self) -> set:
        return set(self._entries.keys())


@dataclass
class ContainerConfig:
    """Configuration for container tracking application."""
    
    # Database
    db_path: str = "data/db/bag_events.db"
    
    # Frame source
    video_source: str = ""  # For development mode
    fps: int = 20
    
    # Tracking
    exit_zone_ratio: float = 0.15
    lost_timeout: float = 2.0
    min_displacement_ratio: float = 0.3
    detect_interval: int = 4  # Run QR detection every N-th frame
    # False-positive gate: a track must accumulate this many *real* QR
    # detections before it can emit an event.  Drops single-frame decoder
    # glitches that would otherwise count as containers.
    min_detections_for_event: int = 3

    # Snapshots
    pre_event_seconds: float = 5.0
    post_event_seconds: float = 5.0
    # Use the same absolute path that the web API reads from (CONTAINER_SNAPSHOT_DIR).
    # A hardcoded relative fallback would diverge from paths.py when DATA_DIR
    # points to a USB/SSD drive, making every event-video lookup a 404.
    snapshot_dir: str = field(
        default_factory=lambda: __import__(
            'src.config.paths', fromlist=['CONTAINER_SNAPSHOT_DIR']
        ).CONTAINER_SNAPSHOT_DIR
    )
    save_video: bool = False

    # QR-camera event video (used as primary when event_video_source="qr"
    # and as fallback when the content camera is unavailable).
    # These are **independent of** ``detect_interval`` \u2014 the event video
    # is sampled from every frame for smooth playback, not from detection
    # ticks.
    event_video_fps: int = 20            # sampling + output fps for QR video
    event_video_max_seconds: float = 10.0 # hard cap on per-track buffered history
    event_video_stationary_px: int = 0    # overwrite last frame if QR barely moved (0=disabled)
    event_video_pre_seconds: float = 4.0  # seconds of video before track entry
    event_video_post_seconds: float = 2.0 # seconds of video to capture after track exit
    
    # State publishing — uses the canonical path from paths.py so the health
    # endpoint (which also reads CONTAINER_PIPELINE_STATE_FILE) always agrees.
    state_file: str = field(
        default_factory=lambda: __import__(
            'src.config.paths', fromlist=['CONTAINER_PIPELINE_STATE_FILE']
        ).CONTAINER_PIPELINE_STATE_FILE
    )
    state_publish_interval: float = 1.0
    
    # Mismatch detection
    mismatch_threshold: int = 5  # Alert if |positive - negative| > threshold
    mismatch_window_hours: float = 1.0
    
    # Display
    enable_display: bool = False

    # Content camera (3D-angle recorder for container contents)
    content_recording_enabled: bool = False
    content_rtsp_host: str = "192.168.2.128"
    content_rtsp_port: str = "554"
    content_rtsp_username: str = ""
    content_rtsp_password: str = ""
    # 0 = main stream (typically 720p, may deliver fewer fps)
    # 1 = sub-stream  (typically 360p, more stable fps)
    content_rtsp_subtype: int = 0
    content_pre_event_seconds: float = 3.0
    content_post_event_seconds: float = 2.0
    content_buffer_seconds: float = 5.0
    content_video_fps: int = 10    # every 2nd frame from 20fps source ≈ 150 frames/15s max
    # Safety cap: auto-finalize any begin/end recording longer than this.
    content_max_recording_seconds: float = 15.0

    # Event video source:
    #   "qr"      -> always use the overhead camera's buffered frames
    #   "content" -> use the content camera recorder when available;
    #                fall back to QR frames otherwise.
    event_video_source: str = "qr"

    # ---- Automatic data retention (mirrors pipeline_core.py pattern) ----
    # QR-camera event clip directories (data/container_snapshots/).
    snapshots_retention_hours: float = 72.0      # delete dirs older than N hours
    snapshots_max_count: int = 500               # hard cap; oldest deleted first
    # Content-camera MP4 files (data/container_content_videos/).
    content_videos_retention_hours: float = 72.0
    content_videos_max_count: int = 200
    # container_events DB rows.
    db_events_retention_hours: float = 168.0     # 7 days
    # How often the purge thread wakes up.
    purge_interval_minutes: float = 60.0

    @classmethod
    def from_database(
        cls,
        db: DatabaseManager,
        base_config: Optional['ContainerConfig'] = None,
    ) -> 'ContainerConfig':
        """Load DB-backed settings while preserving runtime overrides."""
        config = replace(base_config) if base_config is not None else cls()
        
        config.exit_zone_ratio = float(
            db.get_config(constants.container_exit_zone_ratio, str(config.exit_zone_ratio))
        )
        config.lost_timeout = float(
            db.get_config(constants.container_lost_timeout, str(config.lost_timeout))
        )
        config.pre_event_seconds = float(
            db.get_config(constants.container_pre_event_seconds, str(config.pre_event_seconds))
        )
        config.post_event_seconds = float(
            db.get_config(constants.container_post_event_seconds, str(config.post_event_seconds))
        )
        try:
            config.detect_interval = max(1, int(
                db.get_config(constants.container_detect_interval, str(config.detect_interval))
            ))
        except ValueError:
            pass

        try:
            config.min_detections_for_event = max(1, int(
                db.get_config(
                    constants.container_min_detections_for_event,
                    str(config.min_detections_for_event),
                )
            ))
        except ValueError:
            pass

        # ---- Event video (QR-camera side) settings ----
        try:
            config.event_video_fps = max(1, int(
                db.get_config(constants.container_event_video_fps,
                              str(config.event_video_fps))
            ))
            config.event_video_max_seconds = max(0.5, float(
                db.get_config(constants.container_event_video_max_seconds,
                              str(config.event_video_max_seconds))
            ))
            config.event_video_stationary_px = max(0, int(
                db.get_config(constants.container_event_video_stationary_px,
                              str(config.event_video_stationary_px))
            ))
            config.event_video_pre_seconds = max(0.0, float(
                db.get_config(constants.container_event_video_pre_seconds,
                              str(config.event_video_pre_seconds))
            ))
            config.event_video_post_seconds = max(0.0, float(
                db.get_config(constants.container_event_video_post_seconds,
                              str(config.event_video_post_seconds))
            ))
        except ValueError:
            pass

        # ---- Content camera settings ----
        config.content_recording_enabled = (
            db.get_config(constants.content_recording_enabled_key, '0') == '1'
        )
        config.content_rtsp_host = db.get_config(
            constants.content_rtsp_host, config.content_rtsp_host
        )
        config.content_rtsp_port = db.get_config(
            constants.content_rtsp_port, config.content_rtsp_port
        )
        config.content_rtsp_username = db.get_config(
            constants.content_rtsp_username, config.content_rtsp_username
        )
        config.content_rtsp_password = db.get_config(
            constants.content_rtsp_password, config.content_rtsp_password
        )
        try:
            config.content_pre_event_seconds = float(
                db.get_config(constants.content_pre_event_seconds,
                              str(config.content_pre_event_seconds))
            )
            config.content_post_event_seconds = float(
                db.get_config(constants.content_post_event_seconds,
                              str(config.content_post_event_seconds))
            )
            config.content_buffer_seconds = float(
                db.get_config(constants.content_buffer_seconds,
                              str(config.content_buffer_seconds))
            )
            config.content_video_fps = max(1, int(
                db.get_config(constants.content_video_fps,
                              str(config.content_video_fps))
            ))
            config.content_rtsp_subtype = max(0, int(
                db.get_config(constants.content_rtsp_subtype,
                              str(config.content_rtsp_subtype))
            ))
            config.content_max_recording_seconds = float(
                db.get_config(constants.content_max_recording_seconds,
                              str(config.content_max_recording_seconds))
            )
        except ValueError:
            pass

        # Event video source preference ("qr" | "content").
        source = db.get_config(
            constants.container_event_video_source, config.event_video_source
        ).strip().lower()
        config.event_video_source = source if source in ("qr", "content") else "qr"
        
        # Display flag — only read from DB when no base_config was given.
        # When base_config is provided the caller's value is an explicit
        # runtime override and must be preserved.
        if base_config is None:
            config.enable_display = (
                db.get_config(constants.container_enable_display_key, '0') == '1'
            )

        # ---- Retention settings ----
        try:
            config.snapshots_retention_hours = max(0.0, float(
                db.get_config(constants.container_snapshots_retention_hours,
                              str(config.snapshots_retention_hours))
            ))
            config.snapshots_max_count = max(0, int(
                db.get_config(constants.container_snapshots_max_count,
                              str(config.snapshots_max_count))
            ))
            config.content_videos_retention_hours = max(0.0, float(
                db.get_config(constants.container_content_videos_retention_hours,
                              str(config.content_videos_retention_hours))
            ))
            config.content_videos_max_count = max(0, int(
                db.get_config(constants.container_content_videos_max_count,
                              str(config.content_videos_max_count))
            ))
            config.db_events_retention_hours = max(0.0, float(
                db.get_config(constants.container_db_events_retention_hours,
                              str(config.db_events_retention_hours))
            ))
            config.purge_interval_minutes = max(1.0, float(
                db.get_config(constants.container_purge_interval_minutes,
                              str(config.purge_interval_minutes))
            ))
        except ValueError:
            pass

        return config


@dataclass
class ContainerState:
    """Thread-safe state for container tracking."""
    
    # Counts
    total_positive: int = 0
    total_negative: int = 0
    total_lost: int = 0
    
    # Per-QR counts
    qr_positive: Dict[int, int] = field(default_factory=lambda: {i: 0 for i in range(1, 6)})
    qr_negative: Dict[int, int] = field(default_factory=lambda: {i: 0 for i in range(1, 6)})
    
    # Processing metrics
    fps: float = 0.0
    frame_count: int = 0
    qr_detections: int = 0
    processing_time_ms: float = 0.0
    
    # Recent events (last 10)
    recent_events: List[Dict] = field(default_factory=list)
    
    # Status
    active_tracks: int = 0
    pending_snapshots: int = 0
    last_event_time: Optional[str] = None
    
    # Lock for thread safety
    _lock: threading.Lock = field(default_factory=threading.Lock)
    
    def increment_positive(self, qr_value: int) -> None:
        """Thread-safe positive count increment."""
        with self._lock:
            self.total_positive += 1
            self.qr_positive[qr_value] = self.qr_positive.get(qr_value, 0) + 1
    
    def increment_negative(self, qr_value: int) -> None:
        """Thread-safe negative count increment."""
        with self._lock:
            self.total_negative += 1
            self.qr_negative[qr_value] = self.qr_negative.get(qr_value, 0) + 1
    
    def add_event(self, event: Dict) -> None:
        """Add a recent event (keeps last 10)."""
        with self._lock:
            self.recent_events.append(event)
            if len(self.recent_events) > 10:
                self.recent_events.pop(0)
            self.last_event_time = event.get('timestamp')
    
    def to_dict(self) -> Dict:
        """Convert state to dictionary for JSON serialization."""
        with self._lock:
            return {
                'total_positive': self.total_positive,
                'total_negative': self.total_negative,
                'total_lost': self.total_lost,
                'mismatch': self.total_positive - self.total_negative,
                'qr_positive': self.qr_positive.copy(),
                'qr_negative': self.qr_negative.copy(),
                'fps': round(self.fps, 1),
                'frame_count': self.frame_count,
                'qr_detections': self.qr_detections,
                'processing_time_ms': round(self.processing_time_ms, 2),
                'active_tracks': self.active_tracks,
                'pending_snapshots': self.pending_snapshots,
                'recent_events': self.recent_events.copy(),
                'last_event_time': self.last_event_time,
            }


class ContainerCounterApp:
    """
    Main application for container QR tracking.
    
    Orchestrates:
    - Frame acquisition (ROS2 or OpenCV)
    - QR code detection
    - Container tracking with direction detection
    - Snapshot capture on events
    - Database recording
    - State publishing for health monitoring
    """
    
    def __init__(self, config: Optional[ContainerConfig] = None):
        """
        Initialize the container counter application.
        
        Args:
            config: Application configuration (uses defaults if None)
        """
        self.config = config or ContainerConfig()
        self.state = ContainerState()
        
        # Components (initialized in _init_components)
        self.db: Optional[DatabaseManager] = None
        self.qr_detector: Optional[QRCodeDetector] = None
        self.tracker: Optional[ContainerTracker] = None
        self.snapshotter: Optional[RingBufferSnapshotter] = None
        self.frame_server = None
        self._visualizer: Optional[ContainerVisualizer] = None
        self._snapshot_writer: Optional[SnapshotWriter] = None
        self._content_recorder = None  # ContentCameraRecorder (optional)
        self._event_video = None       # EventVideoCoordinator (always set in _init_components)
        self._qr_engine_requested: str = 'auto'
        self._qr_engine_resolved: str = 'unknown'
        
        # Display
        self.enable_display = self.config.enable_display
        
        # Control flags
        self._running = False
        self._stop_event = threading.Event()
        
        # State publishing
        self._last_state_publish = 0.0
        
        # FPS calculation
        self._fps_frames = 0
        self._fps_start_time = time.time()
        
        # Snapshot save thread
        self._snapshot_thread: Optional[threading.Thread] = None
        self._snapshot_queue: queue.Queue = queue.Queue(maxsize=50)
        
        # Last detection/track data for on-demand snapshot annotation
        self._last_detection: Optional[QRDetection] = None
        self._frame_detections: list = []  # per-frame viz list; reset in _process_frame
        # Detection runs every N-th frame; intermediate frames use linear prediction.
        # Will be overwritten with the config value in _init_components.
        self._detect_interval: int = self.config.detect_interval
        # Linear predictor for inter-detection frames (much lighter than cv2.Tracker)
        self._linear_predictor = _LinearPredictor(detect_interval=self._detect_interval)
        self._frame_size: tuple = (1280, 720)  # updated on first frame

        # ── Motion gate ──
        # On detect-eligible frames, skip the expensive QR CNN when the
        # scene is static (no container moving through).  A small grayscale
        # ROI diff between the current frame and the previous detect frame
        # is compared against ``_MOTION_THRESHOLD``.  When the score is
        # below the threshold the frame is treated as a predict frame.
        self._prev_detect_gray = None  # numpy array or None
        # Mean absolute pixel difference threshold (0-255 scale).
        # 3.0 is conservative — typical static-scene noise is <1.5,
        # a moving container produces >8.
        _MOTION_THRESHOLD_DEFAULT = 3.0
        self._motion_threshold: float = float(
            os.environ.get('QR_MOTION_THRESHOLD', _MOTION_THRESHOLD_DEFAULT)
        )
        
        # Per-track rolling buffer for event-video frames.  Decoupled
        # from ``detect_interval`` so the final clip can run at full fps.
        # The buffer automatically throttles to ``event_video_fps`` and
        # overwrites stationary frames to handle "parked container" edge
        # cases without overflowing.  See
        # :class:`src.container.content.EventFrameBuffer`.
        from src.container.content import EventFrameBuffer, EventFrameBufferConfig  # noqa: F401
        self._EventFrameBuffer = EventFrameBuffer
        self._EventFrameBufferConfig = EventFrameBufferConfig
        self._track_buffers: Dict[int, EventFrameBuffer] = {}
        self._TRACK_SNAPSHOT_QUALITY: int = 60        # JPEG quality (passed to coordinator)

        # Global frame ring buffer.  Stores the last N seconds of
        # half-res frames *regardless* of whether a track exists.
        # At finalization time the video is sliced from this single
        # continuous buffer — no per-track buffers or stitching needed.
        _preroll_seconds = (
            float(self.config.event_video_pre_seconds)
            + float(self.config.event_video_max_seconds)
            + float(self.config.event_video_post_seconds)
            + 2.0  # safety margin
        )
        self._qr_preroll = EventFrameBuffer(EventFrameBufferConfig(
            target_fps=float(self.config.event_video_fps),
            max_seconds=_preroll_seconds,
            stationary_px=0,   # no positional dedup
        ))

        # Deferred event-video writes.  When a track exits, the DB event
        # is recorded immediately but the video write is deferred by
        # ``event_video_post_seconds`` so that post-exit frames are
        # captured from the global pre-roll buffer.
        self._pending_video_writes: List[dict] = []

        # Maps qr_value → (event_id, begin_monotonic) for content-camera
        # recordings started at QR-detection-threshold time (begin/end model).
        # begin_monotonic is used to detect whether the safety cap fired.
        self._active_content_events: Dict[int, Tuple[str, float]] = {}
        
        # Async I/O for snapshot saving (never blocks the main loop)
        from concurrent.futures import ThreadPoolExecutor
        self._snapshot_io = ThreadPoolExecutor(max_workers=2, thread_name_prefix="snap")
        
        # Only handle SIGTERM (supervisor stop). Ctrl+C (SIGINT) is left to
        # Python's default so it raises KeyboardInterrupt, which immediately
        # interrupts time.sleep / cap.read / cv2.waitKey rather than waiting
        # for the current sleep to finish (PEP 475 behaviour).
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Background purge thread for old event clips and DB rows.
        # Started after _init_components() so config values are loaded from DB.
        self._purge_thread: Optional[threading.Thread] = None
        self._purge_stop_event = threading.Event()

        # Cached active-tracks snapshot from the last _process_frame call.
        # Reused by _annotate_frame to avoid a third get_active_tracks() dict copy.
        self._current_tracks: Dict[int, object] = {}

        # Frame processing mode for UI overlay: "detect", "gate", or "predict"
        self._frame_mode: str = "detect"

        # Time of the last DB poll for on-demand snapshot requests.
        # Throttled to 1 Hz so the hot frame loop is not blocked by SQLite reads.
        self._last_snapshot_check: float = 0.0

        logger.info(f"[ContainerCounterApp] Initialized on {PLATFORM_NAME}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"[ContainerCounterApp] Received signal {signum}, stopping...")
        self.stop()
    
    def _init_components(self) -> None:
        """Initialize all pipeline components."""
        logger.info("[ContainerCounterApp] Initializing components...")
        
        # Database
        self.db = DatabaseManager(self.config.db_path)
        
        # Load DB-backed settings without losing runtime overrides such as video_source.
        self.config = ContainerConfig.from_database(self.db, base_config=self.config)
        self.enable_display = self.config.enable_display
        # Sync detect_interval from config and rebuild predictor if it changed.
        if self._detect_interval != self.config.detect_interval:
            self._detect_interval = max(1, self.config.detect_interval)
            self._linear_predictor = _LinearPredictor(
                detect_interval=self._detect_interval,
                frame_w=self._frame_size[0],
                frame_h=self._frame_size[1],
            )
        logger.info(
            f"[ContainerCounterApp] Display enabled: {self.enable_display}, "
            f"detect_interval={self._detect_interval} (from DB config)"
        )
        
        # QR Detector
        qr_engine = self.db.get_config(constants.container_qr_engine, 'auto')
        # Legacy engine was removed — silently upgrade stale DB values.
        if qr_engine == 'legacy':
            qr_engine = 'auto'
            logger.warning(
                "[ContainerCounterApp] container_qr_engine='legacy' is no longer "
                "supported — falling back to 'auto' (WeChatQRCode)."
            )
        self.qr_detector = QRCodeDetector(engine=qr_engine)
        self._qr_engine_requested = qr_engine
        self._qr_engine_resolved = self.qr_detector.engine_name
        logger.info(
            f"[ContainerCounterApp] QR engine requested={qr_engine} "
            f"resolved={self.qr_detector.engine_name}"
        )
        
        # Container Tracker
        # Frame width will be updated when first frame arrives
        self.tracker = ContainerTracker(
            frame_width=1280,  # Default, updated on first frame
            exit_zone_ratio=self.config.exit_zone_ratio,
            lost_timeout=self.config.lost_timeout,
            min_displacement_ratio=self.config.min_displacement_ratio,
            min_detections_for_event=self.config.min_detections_for_event,
        )
        
        # Ring Buffer Snapshotter
        self.snapshotter = RingBufferSnapshotter(
            fps=self.config.fps,
            pre_event_seconds=self.config.pre_event_seconds,
            post_event_seconds=self.config.post_event_seconds,
            output_dir=self.config.snapshot_dir,
        )
        
        # Visualizer (always created — used for snapshot annotation even headless)
        self._visualizer = ContainerVisualizer(
            window_name="Container Tracker",
        )
        
        # On-demand snapshot writer (for /snapshot?camera=container)
        from src.config.paths import SNAPSHOT_DIR
        self._snapshot_writer = SnapshotWriter(
            snapshot_dir=SNAPSHOT_DIR,
            prefix="container_",
        )
        
        # Frame Server (platform-dependent)
        self._init_frame_server()

        # Content camera recorder (optional — second IP camera at 3D angle)
        self._init_content_recorder()

        # Event-video coordinator (chooses qr vs content per event, handles fallback)
        self._init_event_video_coordinator()

        logger.info("[ContainerCounterApp] Components initialized")
    
    def _init_frame_server(self) -> None:
        """Initialize the appropriate frame server for the platform."""
        is_development = self.db.get_config(constants.is_development_key, '0') == '1'
        
        if is_development or not IS_RDK:
            # Development mode: use OpenCV
            from src.container.frame_source.ContainerFrameServer import ContainerFrameServer
            
            video_source = self.config.video_source or os.getenv('CONTAINER_VIDEO_PATH', '')
            
            if video_source:
                logger.info(
                    f"[ContainerCounterApp] Development mode: using video {video_source}"
                )
            else:
                logger.warning(
                    "[ContainerCounterApp] Development mode: no video source, "
                    "will return empty frames"
                )
            
            self.frame_server = ContainerFrameServer(source=video_source)
            
        else:
            # Production RDK mode: use ROS2
            logger.info("[ContainerCounterApp] Production mode: using ROS2 frame server")
            
            # Initialize ROS2 context
            from src.ros2.IPC import init_ros2_context
            init_ros2_context()
            
            from src.container.frame_source.ContainerFrameServer import ContainerFrameServer
            self.frame_server = ContainerFrameServer(target_fps=float(self.config.fps))

    def _init_content_recorder(self) -> None:
        """Initialize the optional content camera recorder.

        The content camera is a second IP camera mounted at a 3D angle
        (default 192.168.2.128) that records the contents of each container
        as it passes by.  Enabled via the ``content_recording_enabled``
        config key (DB).
        """
        if not self.config.content_recording_enabled:
            logger.info(
                "[ContainerCounterApp] Content recording disabled (set "
                "content_recording_enabled=1 in config to enable)"
            )
            return

        try:
            from src.container.content import (
                ContentCameraRecorder,
                ContentRecorderConfig,
            )
            from src.config.paths import CONTAINER_CONTENT_VIDEOS_DIR

            user = self.config.content_rtsp_username or "admin"
            pwd = self.config.content_rtsp_password or ""
            host = self.config.content_rtsp_host or "192.168.2.128"
            port = self.config.content_rtsp_port or "554"
            subtype = self.config.content_rtsp_subtype  # 0=main, 1=sub-stream
            rtsp_url = (
                f"rtsp://{user}:{pwd}@{host}:{port}"
                f"/cam/realmonitor?channel=1&subtype={subtype}"
            )

            rc_cfg = ContentRecorderConfig(
                rtsp_url=rtsp_url,
                output_dir=CONTAINER_CONTENT_VIDEOS_DIR,
                buffer_seconds=self.config.content_buffer_seconds,
                pre_event_seconds=self.config.content_pre_event_seconds,
                post_event_seconds=self.config.content_post_event_seconds,
                target_fps=self.config.content_video_fps,
                max_recording_seconds=self.config.content_max_recording_seconds,
            )
            self._content_recorder = ContentCameraRecorder(rc_cfg)
            self._content_recorder.start()
            logger.info(
                f"[ContainerCounterApp] Content recorder started: "
                f"{rc_cfg.rtsp_url_masked} -> {CONTAINER_CONTENT_VIDEOS_DIR}"
            )
        except Exception as e:
            logger.error(
                f"[ContainerCounterApp] Failed to init content recorder: {e}",
                exc_info=True,
            )
            self._content_recorder = None

    def _init_event_video_coordinator(self) -> None:
        """Initialise the camera-source coordinator used per event.

        Called after :meth:`_init_content_recorder` so the coordinator
        gets the (possibly ``None``) content recorder handle.  If the
        user prefers ``content`` but never enabled the content recorder,
        the coordinator will log a warning and fall back to QR on the
        first event.
        """
        from src.container.content import EventVideoCoordinator
        from src.config.paths import CONTAINER_CONTENT_VIDEOS_DIR

        qr_dir = self.config.snapshot_dir
        qr_relroot = qr_dir[5:] if qr_dir.startswith("data/") else qr_dir
        content_dir = CONTAINER_CONTENT_VIDEOS_DIR
        content_relroot = (
            content_dir[5:] if content_dir.startswith("data/") else
            os.path.basename(content_dir.rstrip("/"))
        )

        effective_fps = float(self.config.event_video_fps)
        self._event_video = EventVideoCoordinator(
            source_preference=self.config.event_video_source,
            content_recorder=self._content_recorder,
            qr_output_dir=qr_dir,
            qr_output_relroot=qr_relroot,
            content_output_relroot=content_relroot,
            qr_fps=effective_fps,
            executor=self._snapshot_io,
            jpeg_quality=self._TRACK_SNAPSHOT_QUALITY,
        )
        logger.info(
            f"[ContainerCounterApp] Event video coordinator ready "
            f"source={self.config.event_video_source} "
            f"qr_fps={effective_fps:.1f} "
            f"buffer={self.config.event_video_max_seconds:.1f}s "
            f"stationary_px={self.config.event_video_stationary_px} "
            f"min_detections={self.config.min_detections_for_event} "
            f"content={'yes' if self._content_recorder else 'no'}"
        )
    
    def run(self, max_frames: Optional[int] = None) -> None:
        """
        Main processing loop.
        
        Args:
            max_frames: Maximum frames to process (None for infinite)
        """
        logger.info("[ContainerCounterApp] Starting main loop...")
        
        try:
            self._init_components()
            self._running = True

            # Start snapshot save thread
            self._start_snapshot_thread()

            # Start background purge thread (uses DB-loaded retention config).
            self._start_purge_thread()
            
            fps = self.config.fps or 25
            frame_interval = 1.0 / fps
            frame_count = 0
            _next_frame_time = time.time()  # budget-aware pacing
            
            if self.enable_display:
                logger.info(
                    "[ContainerCounterApp] Display window 'Container Tracker' opening. "
                    "Press 'q' to quit."
                )
            else:
                logger.info("[ContainerCounterApp] Headless mode (no display window).")
            
            for frame, latency_ms in self.frame_server:
                if self._stop_event.is_set():
                    break
                
                if max_frames and frame_count >= max_frames:
                    break

                # Defence-in-depth: skip degenerate frames (e.g. codec init)
                if frame.shape[0] < 4 or frame.shape[1] < 4:
                    continue
                
                # Reset the budget deadline to *now* each time we get a
                # frame.  The frame source (RTSP/ROS2) already paces
                # delivery at the camera's native FPS, so any blocking
                # time inside frame_server.__next__() counts as pacing.
                # The budget pacer only needs to cover the *processing*
                # tail (annotate + display) — not the full frame interval.
                _next_frame_time = time.time()
                frame_start = _next_frame_time
                
                # Process frame
                self._process_frame(frame)
                frame_count += 1
                
                t_process = time.time() - frame_start
                
                if frame_count % 100 == 0:
                    logger.info(
                        f"[ContainerCounterApp] {frame_count} frames processed "
                        f"| FPS={self.state.fps:.1f} "
                        f"| QR detections={self.state.qr_detections}"
                    )
                
                # Annotate only when needed: display is on, OR this is a periodic
                # check frame.  Snapshot capture always re-annotates on demand
                # inside _maybe_capture_snapshot, so no DB read needed here.
                need_annotated = self.enable_display or (self.state.frame_count % 5 == 0)
                t_ann_start = time.time()
                annotated = self._annotate_frame(frame) if need_annotated else frame
                t_annotate = time.time() - t_ann_start

                # Check on-demand snapshot request
                self._maybe_capture_snapshot(frame)
                
                # ── Budget-aware frame pacing ──
                # Instead of sleeping (frame_interval - elapsed) per frame,
                # maintain a running deadline.  If a detect frame overruns
                # (e.g. 300 ms) the next 3 prediction frames skip the sleep
                # entirely until the budget catches up.
                _next_frame_time += frame_interval
                now = time.time()
                remaining = _next_frame_time - now
                
                # If we're more than 1 second behind, reset the deadline
                # to avoid a burst catch-up after a long stall.
                if remaining < -1.0:
                    _next_frame_time = now
                    remaining = 0.0
                
                remaining_ms = max(1, int(remaining * 1000))
                
                if self.enable_display and self._visualizer:
                    # waitKey handles both event-loop pumping and frame pacing
                    should_continue = self._visualizer.show(annotated, delay_ms=remaining_ms)
                    if not should_continue:
                        self._running = False
                        break
                elif remaining > 0.001:
                    # Headless: only sleep if we're ahead of schedule
                    if self._stop_event.wait(timeout=remaining):
                        break
                
                # Log per-frame timing every 50 frames
                if frame_count % 50 == 0:
                    is_det = ((self.state.frame_count - 1) % self._detect_interval == 0)
                    logger.info(
                        f"[Timing] frame={frame_count} "
                        f"{'DETECT' if is_det else 'predict'} "
                        f"process={t_process*1000:.0f}ms "
                        f"annotate={t_annotate*1000:.1f}ms "
                        f"sleep={remaining_ms}ms"
                    )
                
                # Update FPS
                self._update_fps()
                
                # Publish state periodically
                self._maybe_publish_state()
            
            logger.info(f"[ContainerCounterApp] Processed {frame_count} frames")
        
        except KeyboardInterrupt:
            logger.info("[ContainerCounterApp] Interrupted by user (Ctrl+C)")
        except Exception as e:
            logger.error(f"[ContainerCounterApp] Error in main loop: {e}", exc_info=True)
            raise
        finally:
            self._cleanup()
    
    def _process_frame(self, frame) -> None:
        """
        Process a single frame through the pipeline.

        Pipeline:
        1. Add frame to ring buffer
        2. Every N-th frame: run full QR detection and re-init bbox trackers
        3. Intermediate frames: use OpenCV bbox trackers for position prediction
        4. Feed positions (detected or predicted) to ContainerTracker
        5. Check for lost tracks
        6. Handle completed captures
        """
        start_time = time.time()
        _dbg = _PROCESS_DEBUG and (
            _PROCESS_DEBUG >= 2 or (self.state.frame_count % 20 == 0)
        )
        _dbg_t = {}

        if frame is None or frame.size == 0:
            return

        h0, w0 = frame.shape[:2]
        if w0 < 4 or h0 < 4:
            return

        self.state.frame_count += 1

        # Update tracker frame width if needed
        if self.tracker and frame.shape[1] != self.tracker.frame_width:
            self.tracker.update_frame_width(frame.shape[1])

        # Notify predictor of frame size (done once; ignored if unchanged)
        h, w = frame.shape[:2]
        if (w, h) != self._frame_size:
            self._frame_size = (w, h)
            self._linear_predictor.set_frame_size(w, h)

        # Add frame to ring buffer for snapshots
        _t = time.time()
        self.snapshotter.add_frame(frame)
        if _dbg:
            _dbg_t['snapshotter'] = (time.time() - _t) * 1000

        # Always compute half-res frame for the global pre-roll and
        # per-track event-video buffers.  cv2.resize at 720p→360p is
        # <1 ms so the overhead is negligible.
        _t = time.time()
        half = cv2.resize(frame, (max(1, w // 2), max(1, h // 2)))
        self._qr_preroll.add(half, center_x=0)
        if _dbg:
            _dbg_t['half+preroll'] = (time.time() - _t) * 1000

        is_detect_frame = ((self.state.frame_count - 1) % self._detect_interval == 0)
        self._frame_detections = []   # reset per-frame viz list
        # _frame_mode set after we know what actually ran this frame

        # ── Motion gate ──
        # On detect-eligible frames with no active tracks, skip the expensive
        # QR CNN when the scene hasn't changed.  If tracks exist we always
        # run detection so the tracker stays fed.
        if is_detect_frame and not self._current_tracks:
            _t = time.time()
            small = cv2.resize(frame, (160, 90))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            if self._prev_detect_gray is not None:
                diff = cv2.absdiff(gray, self._prev_detect_gray).mean()
                if diff < self._motion_threshold:
                    is_detect_frame = False
                    self._frame_mode = "gate"
                    if _dbg:
                        _dbg_t['motion_gate'] = f'skip(diff={diff:.1f})'
                elif _dbg:
                    _dbg_t['motion_gate'] = f'pass(diff={diff:.1f})'
            elif _dbg:
                _dbg_t['motion_gate'] = 'first'
            self._prev_detect_gray = gray
            if _dbg and 'motion_gate' not in _dbg_t:
                _dbg_t['motion_gate_ms'] = (time.time() - _t) * 1000

        if is_detect_frame:
            # ── Full QR detection ──
            self._frame_mode = "detect"
            t_qr_start = time.time()
            detections = self.qr_detector.detect_all(frame)
            t_qr = time.time() - t_qr_start
            if _dbg:
                _dbg_t['qr_detect_all'] = t_qr * 1000
            
            # Log detection timing periodically (every 10th detect frame)
            if (self.state.frame_count // self._detect_interval) % 10 == 0:
                logger.info(
                    f"[QR-Timing] detect_all={t_qr*1000:.0f}ms "
                    f"found={len(detections)} frame#{self.state.frame_count}"
                )
            self._last_detection = detections[0] if detections else None

            for detection in detections:
                self.state.qr_detections += 1
                self._frame_detections.append((detection, False))  # False = not predicted

                # Feed to container tracker
                event = self.tracker.update(
                    qr_value=detection.qr_number,
                    center=detection.center,
                )
                if event:
                    self._handle_container_event(event)
                else:
                    # Check if this detection just reached the QR-threshold.
                    # If so, start content-camera recording immediately so
                    # the full transit is captured (begin/end model).
                    self._maybe_begin_content_recording(detection.qr_number)

                # Update linear predictor with real detection
                bbox_rect = cv2.boundingRect(detection.bbox)  # (x, y, w, h)
                self._linear_predictor.update_from_detection(
                    detection.qr_number,
                    detection.center,
                    (bbox_rect[2], bbox_rect[3]),
                )
        else:
            # ── Prediction frame: linear extrapolation ──
            self._last_detection = None
            step = (self.state.frame_count % self._detect_interval)
            if step == 0:
                step = self._detect_interval

            current_tracks = self.tracker.get_active_tracks()  # single call; reused below
            if current_tracks:
                self._frame_mode = "predict"
            else:
                self._frame_mode = "gate"
            for qr_val, center, bbox_xywh in self._linear_predictor.predict(step):
                # Only predict for QR values that have an active track.
                # If the track was silently dropped (FP gate, etc.) remove the
                # orphaned predictor entry so it doesn't linger until MAX_PRED_FRAMES.
                if qr_val not in current_tracks:
                    self._linear_predictor.remove(qr_val)
                    continue

                pred_det = _PredictedDetection(qr_val, center, bbox_xywh)
                self._frame_detections.append((pred_det, True))  # True = predicted

                # Feed predicted centre to container tracker.
                # is_prediction=True so the tracker does NOT count this
                # toward its false-positive min-detections gate.
                event = self.tracker.update(
                    qr_value=qr_val,
                    center=center,
                    is_prediction=True,
                )
                if event:
                    self._handle_container_event(event)

        # Check for lost tracks
        lost_events = self.tracker.check_lost_tracks()
        for event in lost_events:
            actually_lost, reason = self._classify_lost_track(event)
            event.lost_reason = reason if actually_lost else ""
            logger.info(
                f"[ContainerCounterApp] Lost-track resolved: "
                f"QR={event.qr_value} track=#{event.track_id} "
                f"dir={event.direction.value} "
                f"is_lost={actually_lost} reason={reason!r}"
            )
            self._handle_container_event(event, is_lost=actually_lost)

        # Handle completed snapshot captures
        completed = self.snapshotter.get_completed_captures()
        for capture in completed:
            try:
                self._snapshot_queue.put_nowait(capture)
            except queue.Full:
                logger.warning("[ContainerCounterApp] Snapshot queue full, dropping capture")

        # Refresh current_tracks on detect frames (tracks may have been
        # added/removed by the tracker).
        if is_detect_frame:
            current_tracks = self.tracker.get_active_tracks()

        # Clean up orphaned content-camera recordings whose tracks were
        # silently dropped by the FP gate (detection_count < min) before
        # an exit event could fire.  Without this, begin_event_recording
        # would never receive a matching end_event_recording call.
        if self._active_content_events:
            stale_content = [
                q for q in self._active_content_events
                if q not in current_tracks
            ]
            for q in stale_content:
                entry = self._active_content_events.pop(q, None)
                if entry and self._content_recorder is not None:
                    eid, _begin = entry
                    self._content_recorder.end_event_recording(eid)
                    logger.debug(
                        f"[ContainerCounterApp] Cleaned up orphan content "
                        f"recording QR={q} event_id={eid}"
                    )

        # Finalize any deferred event-video writes whose post-exit
        # buffering period has elapsed.
        if self._pending_video_writes:
            self._finalize_pending_videos()

        # Cache for _annotate_frame (avoids a third dict copy per annotated frame).
        self._current_tracks = current_tracks

        # Update state
        self.state.active_tracks = len(current_tracks)
        self.state.pending_snapshots = self._snapshot_queue.qsize()
        self.state.processing_time_ms = (time.time() - start_time) * 1000

        if _dbg:
            total_ms = self.state.processing_time_ms
            parts = ' '.join(f"{k}={v:.1f}ms" for k, v in _dbg_t.items())
            logger.info(
                f"[Process-DBG] frame#{self.state.frame_count} "
                f"detect={is_detect_frame} total={total_ms:.1f}ms {parts}"
            )

        # Warn when a single frame takes longer than its time budget (× 1.2 headroom).
        # This is the earliest signal that the pipeline is falling behind real-time
        # and frames are beginning to queue up in the frame server.
        _budget_ms = 1000.0 / max(1, self.config.fps)
        if self.state.processing_time_ms > _budget_ms * 1.2:
            logger.warning(
                f"[ContainerCounterApp] Frame over budget: "
                f"{self.state.processing_time_ms:.1f} ms > {_budget_ms:.0f} ms "
                f"(frame #{self.state.frame_count})"
            )
    
    def _annotate_frame(self, frame):
        """
        Create annotated frame with QR detections and tracking overlays.
        
        Always creates annotations (needed for on-demand snapshots even in
        headless mode). Only displayed when enable_display is True.
        """
        if self._visualizer is None or frame is None or frame.size == 0:
            return frame
        
        annotated = frame.copy()
        
        with self.state._lock:
            qr_pos = self.state.qr_positive.copy()
            qr_neg = self.state.qr_negative.copy()
            events = self.state.recent_events.copy()
        
        annotated = self._visualizer.annotate_frame(
            frame=annotated,
            detection=self._last_detection,
            active_tracks=self._current_tracks,
            fps=self.state.fps,
            total_positive=self.state.total_positive,
            total_negative=self.state.total_negative,
            total_lost=self.state.total_lost,
            qr_positive=qr_pos,
            qr_negative=qr_neg,
            recent_events=events,
            exit_zone_ratio=self.config.exit_zone_ratio,
            frame_detections=self._frame_detections,
            frame_mode=self._frame_mode,
        )
        
        return annotated
    
    def _maybe_capture_snapshot(self, frame) -> None:
        """
        Check if on-demand snapshot is requested via DB flag and capture.

        This enables /snapshot?camera=container in the web UI.
        Throttled to 1 Hz (time-based) to avoid excessive DB reads.
        Always annotates the frame at capture time so the overlay is
        guaranteed regardless of the main-loop annotation cadence.
        """
        if self.db is None or self._snapshot_writer is None:
            return

        now = time.time()
        if now - self._last_snapshot_check < 1.0:
            return
        self._last_snapshot_check = now
        
        try:
            requested = self.db.get_config(constants.container_snapshot_requested_key, "0")
            if requested != "1":
                return
            
            # Always produce a fresh overlay for the snapshot
            annotated = self._annotate_frame(frame)

            # Write snapshot (raw + annotated overlay)
            success = self._snapshot_writer.write_snapshot(
                frame=frame,
                frame_with_overlay=annotated,
                frame_number=self.state.frame_count,
            )
            
            # Clear the flag immediately
            self.db.set_config(constants.container_snapshot_requested_key, "0")
            
            if success:
                logger.debug(
                    f"[ContainerCounterApp] Snapshot captured at frame {self.state.frame_count}"
                )
        except Exception as e:
            logger.error(f"[ContainerCounterApp] Snapshot error: {e}")
            try:
                self.db.set_config(constants.container_snapshot_requested_key, "0")
            except Exception:
                pass
    
    def _maybe_begin_content_recording(self, qr_value: int) -> None:
        """Start content-camera recording when QR detection threshold is first met.

        Called on every real detection that does NOT fire an exit event.
        Only triggers once per track: when ``detection_count`` first
        reaches ``min_detections_for_event``.
        """
        if qr_value in self._active_content_events:
            return  # already started for this track
        if self._content_recorder is None or not self._content_recorder.is_available():
            return
        track = self.tracker.get_track(qr_value)
        if track is None:
            return
        if track.detection_count != self.tracker.min_detections_for_event:
            return
        event_id = f"qr{qr_value}_{datetime.now():%Y%m%d_%H%M%S_%f}"
        self._content_recorder.begin_event_recording(
            event_id, track.entry_time_monotonic,
        )
        self._active_content_events[qr_value] = (event_id, time.monotonic())
        logger.info(
            f"[ContainerCounterApp] Content recording started: "
            f"QR={qr_value} event_id={event_id}"
        )

    def _handle_container_event(
        self,
        event: ContainerEvent,
        is_lost: bool = False
    ) -> None:
        """
        Handle a container tracking event.

        The DB row and dashboard event are recorded immediately.
        The QR-camera video write is **deferred** by
        ``event_video_post_seconds`` so that post-exit frames are
        captured from the global pre-roll buffer.

        Args:
            event: The container event
            is_lost: Whether this was a lost track
        """
        # Update state counts
        if event.direction == Direction.POSITIVE:
            self.state.increment_positive(event.qr_value)
        elif event.direction == Direction.NEGATIVE:
            self.state.increment_negative(event.qr_value)

        if is_lost:
            self.state.total_lost += 1

        # ── End content-camera recording if one was started at threshold ──
        content_entry = self._active_content_events.pop(event.qr_value, None)
        content_event_id: Optional[str] = None
        content_already_started = False
        was_capped = False
        elapsed = 0.0  # time from begin_event_recording → end call (content cam only)
        if content_entry and self._content_recorder is not None:
            content_event_id, begin_mono = content_entry
            self._content_recorder.end_event_recording(content_event_id)
            content_already_started = True
            elapsed = time.monotonic() - begin_mono
            was_capped = elapsed >= (self.config.content_max_recording_seconds - 0.5)

        # ── Pop per-track buffer (cleanup only — not used for video) ──
        buf = self._track_buffers.pop(event.qr_value, None)

        # Use the threshold-time event_id for content recordings so the
        # DB record, content video file, and QR fallback all share the
        # same identifier.
        event_id = content_event_id if content_event_id else self._make_event_id(event)

        # Clear linear predictor for this QR so it doesn't ghost-predict.
        self._linear_predictor.remove(event.qr_value)

        exit_mono = time.monotonic()
        pre_seconds = float(self.config.event_video_pre_seconds)
        post_seconds = float(self.config.event_video_post_seconds)

        # ── Defer the QR video write so post-exit frames are captured ──
        # The global preroll buffer has every frame continuously.
        # At finalization time we slice it by timestamp range:
        #   [entry_mono - pre_seconds, exit_mono + post_seconds]
        # This gives a seamless clip with no boundary gaps.
        entry_mono = event.entry_time_monotonic if event.entry_time_monotonic > 0.0 else exit_mono

        pending = {
            'event': event,
            'event_id': event_id,
            'is_lost': is_lost,
            'entry_mono': entry_mono,
            'exit_mono': exit_mono,
            'pre_seconds': pre_seconds,
            'deadline': exit_mono + post_seconds,
            'content_already_started': content_already_started,
            'was_capped': was_capped,
            'elapsed': elapsed,
            'buf_stats': buf.stats() if buf is not None else None,
        }
        self._pending_video_writes.append(pending)

        if buf is not None:
            logger.debug(
                f"[ContainerCounterApp] Track buffer QR={event.qr_value}: "
                f"{buf.stats()} pre={pre_seconds:.1f}s "
                f"post={post_seconds:.1f}s"
            )

        # ── Record DB + dashboard event immediately (video path TBD) ──
        # We write a preliminary DB row now so the event appears in the
        # dashboard in real-time.  The snapshot_path and clip_duration are
        # re-computed when the video is finalized.
        preliminary_result = None
        if content_already_started:
            # Content video path is known immediately.
            from src.container.content.EventVideoCoordinator import EventVideoResult
            rel = f"{self._event_video._content_output_relroot}/{event_id}.mp4"
            preliminary_result = EventVideoResult(
                camera="content", fallback=False, video_relpath=rel
            )

        # Compute preliminary clip_duration for the DB row
        if content_already_started and not (preliminary_result and preliminary_result.fallback):
            if was_capped:
                clip_duration_seconds: float = self.config.content_max_recording_seconds
            else:
                clip_duration_seconds = min(
                    elapsed + self.config.content_post_event_seconds,
                    self.config.content_max_recording_seconds,
                )
        else:
            clip_duration_seconds = pre_seconds + event.duration_seconds + post_seconds

        recording_status = "pending"
        if content_already_started:
            recording_status = "capped" if was_capped else "ok"

        self._record_event(
            event,
            event_id=event_id,
            is_lost=is_lost,
            video_result=preliminary_result,
            recording_status=recording_status,
            clip_duration_seconds=clip_duration_seconds,
        )

        # Add to recent events (for live dashboard).
        event_dict = {
            'timestamp': event.timestamp.isoformat(),
            'qr_value': event.qr_value,
            'direction': event.direction.value,
            'track_id': event.track_id,
            'duration': round(event.duration_seconds, 2),
            'is_lost': is_lost,
            'lost_reason': event.lost_reason if is_lost else "",
            'camera': preliminary_result.camera if preliminary_result else 'qr',
            'fallback': bool(preliminary_result.fallback) if preliminary_result else False,
            'recording_status': recording_status,
            'clip_duration_seconds': round(clip_duration_seconds, 1),
        }
        self.state.add_event(event_dict)

        logger.info(
            f"[ContainerCounterApp] EVENT "
            f"QR={event.qr_value} track=#{event.track_id} "
            f"dir={event.direction.value} "
            f"lost={'YES ⚠' if is_lost else 'NO ✓'} "
            f"entry_x={event.entry_x} exit_x={event.exit_x} "
            f"disp={event.exit_x - event.entry_x:+}px "
            f"dur={event.duration_seconds:.2f}s "
            f"video_deferred={post_seconds:.1f}s "
            f"positions={len(event.positions)}"
        )

    # ------------------------------------------------------------------
    # Deferred video finalization
    # ------------------------------------------------------------------

    def _finalize_pending_videos(self) -> None:
        """Finalize deferred event-video writes whose post-exit period has elapsed.

        Called every frame from ``_process_frame``.  For each matured entry:
        1. Grab post-exit frames from the global pre-roll buffer.
        2. Combine pre-roll (pre-entry) + track buffer (transit) + post-roll (post-exit).
        3. Submit to ``EventVideoCoordinator``.
        """
        now = time.monotonic()
        still_pending = []
        for pw in self._pending_video_writes:
            if now < pw['deadline']:
                still_pending.append(pw)
                continue
            self._finalize_one_video(pw)
        self._pending_video_writes = still_pending

    def _finalize_one_video(self, pw: dict) -> None:
        """Finalize a single deferred event-video write."""
        event = pw['event']
        event_id = pw['event_id']
        is_lost = pw['is_lost']
        entry_mono = pw['entry_mono']
        exit_mono = pw['exit_mono']
        pre_seconds = pw['pre_seconds']
        post_deadline = pw['deadline']
        content_already_started = pw['content_already_started']

        # Content-camera events were written immediately — nothing to finalize.
        if content_already_started:
            return

        # Single continuous slice from the global preroll buffer.
        # No boundary stitching — one source of truth, zero gaps.
        clip_start = entry_mono - pre_seconds
        clip_end = post_deadline
        all_entries = [
            e for e in self._qr_preroll.snapshot_frames()
            if clip_start <= e[0] <= clip_end
        ]
        track_frames = [frame for _, frame, _ in all_entries]

        # Compute measured fps from timestamps for real-time playback.
        qr_measured_fps: Optional[float] = None
        if len(all_entries) >= 2:
            span = all_entries[-1][0] - all_entries[0][0]
            if span > 0:
                mfps = len(all_entries) / span
                qr_measured_fps = max(1.0, min(
                    float(self.config.event_video_fps), mfps
                ))

        # Actual clip duration from timestamps.
        if len(all_entries) >= 2:
            clip_duration_seconds = all_entries[-1][0] - all_entries[0][0]
        else:
            clip_duration_seconds = event.duration_seconds

        logger.info(
            f"[ContainerCounterApp] Finalizing video "
            f"event={event_id} QR={event.qr_value} "
            f"frames={len(all_entries)} "
            f"fps={f'{qr_measured_fps:.1f}' if qr_measured_fps else 'n/a'} "
            f"clip={clip_duration_seconds:.1f}s"
        )

        # Dispatch to coordinator.
        video_result = self._capture_event_video(
            event_id=event_id,
            event=event,
            track_frames=track_frames,
            qr_measured_fps=qr_measured_fps,
            is_lost=is_lost,
            content_already_started=False,
        )

        # Update DB row with final video path and recording status.
        if video_result is not None and video_result.video_relpath is not None:
            recording_status = "ok"
        else:
            recording_status = "no_video"

        try:
            new_metadata = json.dumps({
                'recording_status': recording_status,
                'camera': video_result.camera if video_result else None,
                'fallback': bool(video_result.fallback) if video_result else False,
                'video_relpath': video_result.video_relpath if video_result else None,
                'clip_duration_seconds': round(clip_duration_seconds, 1),
                'frame_count': len(track_frames),
            })
            self.db.enqueue_write(
                """UPDATE container_events
                   SET snapshot_path = ?, metadata = ?
                   WHERE timestamp = ? AND qr_code_value = ? AND track_id = ?""",
                (
                    video_result.video_relpath if video_result else None,
                    new_metadata,
                    event.timestamp.isoformat(),
                    event.qr_value,
                    event.track_id,
                ),
            )
        except Exception as e:
            logger.error(f"[ContainerCounterApp] DB update for video failed: {e}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_event_id(event: ContainerEvent) -> str:
        """Format: ``qr{N}_{direction}_{YYYYMMDD_HHMMSS_ffffff}``."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"qr{event.qr_value}_{event.direction.value}_{ts}"

    def _capture_event_video(
        self,
        *,
        event_id: str,
        event: ContainerEvent,
        track_frames: list,
        qr_measured_fps: Optional[float] = None,
        is_lost: bool,
        content_already_started: bool = False,
    ):
        """Delegate event-video capture to the coordinator.

        Builds the metadata payload that gets written alongside the QR
        video (and echoed into the DB ``metadata`` column).  Returns the
        :class:`EventVideoResult` or ``None`` if no coordinator exists
        (e.g. during unit tests with a partial init).
        """
        if self._event_video is None:
            return None

        metadata = {
            'event_id': event_id,
            'qr_value': event.qr_value,
            'direction': event.direction.value,
            'is_lost': bool(is_lost),
            'trigger_time': datetime.now().isoformat(),
            'duration_seconds': event.duration_seconds,
            'track_id': event.track_id,
            'entry_x': event.entry_x,
            'exit_x': event.exit_x,
            'frame_count': len(track_frames),
            'detection_count': event.detection_count,
        }

        # Anchor the content-camera pre-roll at the track's *entry*
        # moment (when the container first became visible) rather than
        # "now" (event completion).  This is what delivers the 3-second
        # lead-in the operator sees in the final clip.  If the event
        # carries no monotonic anchor (legacy data), fall back to a
        # wall-clock-derived estimate so we never crash.
        if event.entry_time_monotonic > 0.0:
            trigger_mono = event.entry_time_monotonic
        else:
            # Best-effort fallback: shift "now" back by the track's
            # duration so the ring buffer still captures the lead-in.
            trigger_mono = time.monotonic() - max(0.0, event.duration_seconds)

        try:
            return self._event_video.capture(
                event_id=event_id,
                trigger_monotonic_time=trigger_mono,
                qr_frames=track_frames,
                qr_fps_override=qr_measured_fps,
                metadata=metadata,
                content_already_started=content_already_started,
            )
        except Exception as e:
            logger.error(
                f"[ContainerCounterApp] EventVideoCoordinator.capture failed: {e}",
                exc_info=True,
            )
            return None

    def _record_event(
        self,
        event: ContainerEvent,
        *,
        event_id: Optional[str],
        is_lost: bool = False,
        video_result=None,
        recording_status: str = "ok",
        clip_duration_seconds: float = 0.0,
    ) -> None:
        """Record event to database."""
        try:
            # ``snapshot_path`` is the legacy column name; it now points to
            # the directory (QR camera) or file (content camera) that holds
            # the event clip — whichever the coordinator chose.
            snapshot_path: Optional[str] = None
            camera = video_result.camera if video_result else None
            fallback = bool(video_result.fallback) if video_result else False
            video_relpath = video_result.video_relpath if video_result else None

            if video_relpath:
                snapshot_path = video_relpath
            elif event_id:
                # No video produced, but we still point at the metadata dir
                # (QR path) so the UI can render "no video" gracefully.
                qr_root = self.config.snapshot_dir
                if qr_root.startswith('data/'):
                    qr_root = qr_root[5:]
                snapshot_path = os.path.join(qr_root, event_id)

            metadata = json.dumps({
                'position_count': len(event.positions),
                'entry_x': event.entry_x,
                'exit_x': event.exit_x,
                'camera': camera,
                'fallback': fallback,
                'video_relpath': video_relpath,
                'recording_status': recording_status,
                # Actual video clip length (pre-roll + transit + post-roll).
                # Distinct from duration_seconds which is QR tracking time only.
                'clip_duration_seconds': round(clip_duration_seconds, 1),
                'lost_reason': event.lost_reason if is_lost else None,
            })

            # Insert into database (entry_y/exit_y columns store X positions
            # since tracking axis is horizontal).
            self.db.enqueue_write(
                """
                INSERT INTO container_events (
                    timestamp, qr_code_value, direction, track_id,
                    entry_y, exit_y, duration_seconds,
                    snapshot_path, is_lost, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.timestamp.isoformat(),
                    event.qr_value,
                    event.direction.value,
                    event.track_id,
                    event.entry_x,
                    event.exit_x,
                    event.duration_seconds,
                    snapshot_path,
                    1 if is_lost else 0,
                    metadata,
                )
            )
            
        except Exception as e:
            logger.error(f"[ContainerCounterApp] Failed to record event: {e}")

    def _classify_lost_track(self, event: ContainerEvent) -> Tuple[bool, str]:
        """
        Determine whether a timed-out track is truly lost or just had its QR
        become unreadable near the exit.

        Returns
        -------
        (is_lost, reason)
            is_lost  – True  → record as lost (video + lost counter)
                       False → record as normal exit (QR unreadable near edge)
            reason   – human-readable string stored in DB and shown on UI
        """
        if self.tracker is None:
            return False, "no tracker"

        frame_w = self.tracker.frame_width
        x = event.exit_x
        direction = event.direction
        pct = int(x / frame_w * 100)

        # ── Generous zone: 5× exit_zone_ratio, capped at 30% ──
        generous_ratio = min(self.tracker.exit_zone_ratio * 5, 0.30)
        left_thr = int(frame_w * generous_ratio)        # e.g. 384 px
        right_thr = int(frame_w * (1.0 - generous_ratio))  # e.g. 896 px

        in_generous_zone = x <= left_thr or x >= right_thr

        # ── Direction-aware mid-point check ──
        # A container that has clearly crossed the frame midpoint *toward*
        # its expected exit was almost certainly exiting — the QR code just
        # became unreadable before the strict boundary.  Only applied when
        # direction is determined (not UNKNOWN).
        from src.container.tracking.ContainerTracker import Direction
        past_midpoint_toward_exit = (
            (direction == Direction.POSITIVE and x < frame_w // 2) or
            (direction == Direction.NEGATIVE and x > frame_w // 2)
        )

        if in_generous_zone:
            reason = f"near-exit at {pct}% (generous zone)"
            return False, reason

        if past_midpoint_toward_exit and direction != Direction.UNKNOWN:
            reason = f"past midpoint at {pct}% toward {direction.value} exit — QR unreadable"
            return False, reason

        # ── Truly lost ──
        if direction == Direction.UNKNOWN:
            disp = event.exit_x - event.entry_x
            reason = (
                f"direction unknown — only {abs(disp)}px displacement "
                f"(need ≥{int(frame_w * self.tracker.min_displacement_ratio)}px)"
            )
        elif direction == Direction.POSITIVE:
            reason = f"mid-frame at {pct}% (left exit not reached, entry={event.entry_x}px)"
        else:
            reason = f"mid-frame at {pct}% (right exit not reached, entry={event.entry_x}px)"

        logger.debug(
            f"[ContainerCounterApp] lost_classify QR={event.qr_value} "
            f"exit_x={x} dir={direction.value} → lost=True reason={reason!r}"
        )
        return True, reason

    def _update_fps(self) -> None:
        """Update FPS calculation with EMA smoothing."""
        self._fps_frames += 1
        now = time.time()
        elapsed = now - self._fps_start_time
        
        if elapsed >= 1.0:
            raw_fps = self._fps_frames / elapsed
            # Exponential moving average (α=0.3) for stable display
            if self.state.fps > 0:
                self.state.fps = 0.3 * raw_fps + 0.7 * self.state.fps
            else:
                self.state.fps = raw_fps
            self._fps_frames = 0
            self._fps_start_time = now
    
    def _maybe_publish_state(self) -> None:
        """Publish state to JSON file if interval has elapsed."""
        now = time.time()
        
        if now - self._last_state_publish >= self.config.state_publish_interval:
            self._publish_state()
            self._last_state_publish = now
    
    def _publish_state(self) -> None:
        """Write current state to JSON file for health monitoring.
        
        State fields are at the TOP level (not nested under 'state')
        so that health.py and server.py can read them as
        container_data.get('fps'), container_data.get('total_positive'), etc.
        """
        try:
            state_dict = self.state.to_dict()
            
            now = time.time()
            cfg = self.config
            state_data = {
                'timestamp': now,
                'updated_at': now,
                # Flatten state fields to top level for health monitoring
                **state_dict,
                # Sub-component stats
                'tracker': self.tracker.get_stats() if self.tracker else {},
                'snapshotter': self.snapshotter.get_stats() if self.snapshotter else {},
                'qr_detector': self.qr_detector.get_stats() if self.qr_detector else {},
                # Config / deployment info for the health page
                'config_info': {
                    'qr_rtsp_source': cfg.video_source or 'rtsp',
                    'qr_engine_requested': self._qr_engine_requested,
                    'qr_engine_resolved': self._qr_engine_resolved,
                    'content_recording_enabled': cfg.content_recording_enabled,
                    'content_rtsp_host': cfg.content_rtsp_host if cfg.content_recording_enabled else None,
                    'content_rtsp_port': cfg.content_rtsp_port if cfg.content_recording_enabled else None,
                    'event_video_source': cfg.event_video_source,
                    'camera_mode': 'dual' if cfg.content_recording_enabled else 'single',
                    'detect_interval': cfg.detect_interval,
                    'min_detections_for_event': cfg.min_detections_for_event,
                },
            }
            
            # Atomic write
            tmp_path = self.config.state_file + '.tmp'
            with open(tmp_path, 'w') as f:
                json.dump(state_data, f, indent=2)
            os.rename(tmp_path, self.config.state_file)
            
        except Exception as e:
            logger.error(f"[ContainerCounterApp] Failed to publish state: {e}")
    
    def _start_snapshot_thread(self) -> None:
        """Start background thread for saving snapshots."""
        self._snapshot_thread = threading.Thread(
            target=self._snapshot_save_loop,
            daemon=True,
            name="ContainerSnapshotSaver"
        )
        self._snapshot_thread.start()
    
    def _snapshot_save_loop(self) -> None:
        """Background thread that saves completed captures."""
        while not self._stop_event.is_set():
            try:
                capture = self._snapshot_queue.get(timeout=0.5)
                self.snapshotter.save_capture(
                    capture,
                    save_video=self.config.save_video
                )
            except queue.Empty:
                continue

    # ------------------------------------------------------------------
    # Automatic data-retention / purge
    # ------------------------------------------------------------------

    def _start_purge_thread(self) -> None:
        """Start background data-retention thread.

        Mirrors the ``_start_purge_thread`` pattern in ``pipeline_core.py``.
        Safe to call multiple times — does nothing if thread already running.
        """
        if self._purge_thread is not None and self._purge_thread.is_alive():
            return
        self._purge_stop_event.clear()
        self._purge_thread = threading.Thread(
            target=self._purge_loop,
            name="ContainerPurger",
            daemon=True,
        )
        self._purge_thread.start()
        logger.info(
            f"[ContainerPurger] Started "
            f"(snap_ret={self.config.snapshots_retention_hours}h "
            f"snap_max={self.config.snapshots_max_count} "
            f"vid_ret={self.config.content_videos_retention_hours}h "
            f"vid_max={self.config.content_videos_max_count} "
            f"db_ret={self.config.db_events_retention_hours}h "
            f"interval={self.config.purge_interval_minutes}min)"
        )

    def _purge_loop(self) -> None:
        """Periodic purge loop — runs in its own daemon thread."""
        interval = self.config.purge_interval_minutes * 60.0
        while not self._purge_stop_event.is_set():
            try:
                if self._purge_stop_event.wait(timeout=interval):
                    break
                self._purge_container_snapshots()
                self._purge_content_videos()
                self._purge_db_events()
            except Exception as e:
                logger.error(f"[ContainerPurger] Unhandled error: {e}", exc_info=True)
        logger.info("[ContainerPurger] Stopped")

    def _purge_container_snapshots(self) -> None:
        """Delete old QR-camera event-clip directories (data/container_snapshots/).

        Two-phase:
        1. Time-based  — remove dirs older than ``snapshots_retention_hours``.
        2. Count-based — if still over ``snapshots_max_count``, delete oldest first.
        """
        import glob
        import shutil
        snap_dir = self.config.snapshot_dir
        if not os.path.isdir(snap_dir):
            return
        try:
            # Each event lives in its own sub-directory named after event_id.
            entries = [
                (p, os.path.getmtime(p))
                for p in (os.path.join(snap_dir, d) for d in os.listdir(snap_dir))
                if os.path.isdir(p)
            ]
        except OSError as e:
            logger.warning(f"[ContainerPurger] Cannot list {snap_dir}: {e}")
            return
        if not entries:
            return

        initial = len(entries)
        deleted_time = deleted_count = 0

        # Phase 1 – time-based
        ret_h = self.config.snapshots_retention_hours
        if ret_h > 0:
            cutoff = time.time() - ret_h * 3600
            keep = []
            for path, mtime in entries:
                if mtime < cutoff:
                    try:
                        shutil.rmtree(path, ignore_errors=True)
                        deleted_time += 1
                    except Exception as exc:
                        logger.warning(f"[ContainerPurger] rmtree {path}: {exc}")
                else:
                    keep.append((path, mtime))
            entries = keep

        # Phase 2 – count-based
        max_c = self.config.snapshots_max_count
        if max_c > 0 and len(entries) > max_c:
            entries.sort(key=lambda x: x[1])   # oldest first
            for path, _ in entries[:len(entries) - max_c]:
                try:
                    shutil.rmtree(path, ignore_errors=True)
                    deleted_count += 1
                except Exception as exc:
                    logger.warning(f"[ContainerPurger] rmtree {path}: {exc}")

        total = deleted_time + deleted_count
        if total:
            logger.info(
                f"[ContainerPurger] Snapshots: "
                f"time={deleted_time} count={deleted_count} "
                f"remaining={initial - total}"
            )

    def _purge_content_videos(self) -> None:
        """Delete old content-camera MP4 files (data/container_content_videos/).

        Two-phase time + count purge, same algorithm as ``_purge_container_snapshots``.
        """
        try:
            from src.config.paths import CONTAINER_CONTENT_VIDEOS_DIR
        except ImportError:
            return
        vid_dir = CONTAINER_CONTENT_VIDEOS_DIR
        if not os.path.isdir(vid_dir):
            return
        try:
            entries = [
                (p, os.path.getmtime(p))
                for p in (
                    os.path.join(vid_dir, f)
                    for f in os.listdir(vid_dir)
                    if f.endswith('.mp4')
                )
                if os.path.isfile(p)
            ]
        except OSError as e:
            logger.warning(f"[ContainerPurger] Cannot list {vid_dir}: {e}")
            return
        if not entries:
            return

        initial = len(entries)
        deleted_time = deleted_count = 0

        # Phase 1 – time-based
        ret_h = self.config.content_videos_retention_hours
        if ret_h > 0:
            cutoff = time.time() - ret_h * 3600
            keep = []
            for path, mtime in entries:
                if mtime < cutoff:
                    try:
                        os.remove(path)
                        deleted_time += 1
                        # Remove companion metadata JSON if present
                        meta = path[:-4] + '.json'
                        if os.path.isfile(meta):
                            os.remove(meta)
                    except OSError as exc:
                        logger.warning(f"[ContainerPurger] unlink {path}: {exc}")
                else:
                    keep.append((path, mtime))
            entries = keep

        # Phase 2 – count-based
        max_c = self.config.content_videos_max_count
        if max_c > 0 and len(entries) > max_c:
            entries.sort(key=lambda x: x[1])
            for path, _ in entries[:len(entries) - max_c]:
                try:
                    os.remove(path)
                    deleted_count += 1
                    meta = path[:-4] + '.json'
                    if os.path.isfile(meta):
                        os.remove(meta)
                except OSError as exc:
                    logger.warning(f"[ContainerPurger] unlink {path}: {exc}")

        total = deleted_time + deleted_count
        if total:
            logger.info(
                f"[ContainerPurger] Content videos: "
                f"time={deleted_time} count={deleted_count} "
                f"remaining={initial - total}"
            )

    def _purge_db_events(self) -> None:
        """Delete old rows from the ``container_events`` table.

        Uses a time-based cutoff only (no count cap — the table is indexed
        and row overhead is small).  Runs asynchronously via ``enqueue_write``
        so the main loop is never blocked.
        """
        if self.db is None:
            return
        ret_h = self.config.db_events_retention_hours
        if ret_h <= 0:
            return
        try:
            from datetime import timedelta as _td, timezone as _tz
            cutoff_dt = datetime.now(_tz.utc) - _td(hours=ret_h)
            cutoff_iso = cutoff_dt.strftime("%Y-%m-%dT%H:%M:%S")
            # Count before delete (non-blocking read for stats only)
            row = self.db.fetchone(
                "SELECT COUNT(*) FROM container_events WHERE timestamp < ?",
                (cutoff_iso,),
            )
            count = row[0] if row else 0
            if count > 0:
                self.db.enqueue_write(
                    "DELETE FROM container_events WHERE timestamp < ?",
                    (cutoff_iso,),
                )
                logger.info(
                    f"[ContainerPurger] DB events: queued delete of {count} rows "
                    f"older than {ret_h:.0f}h (cutoff={cutoff_iso})"
                )
        except Exception as e:
            logger.error(f"[ContainerPurger] DB purge error: {e}", exc_info=True)

    def stop(self) -> None:
        """Stop the application gracefully."""
        logger.info("[ContainerCounterApp] Stopping...")
        self._running = False
        self._stop_event.set()

    def _cleanup(self) -> None:
        """Clean up resources."""
        logger.info("[ContainerCounterApp] Cleaning up...")

        # Flush any pending deferred video writes immediately.
        for pw in self._pending_video_writes:
            try:
                self._finalize_one_video(pw)
            except Exception as e:
                logger.error(f"[ContainerCounterApp] Flush pending video failed: {e}")
        self._pending_video_writes.clear()

        # Publish final state
        self._publish_state()

        # Stop background purge thread first so it doesn't re-open the DB
        # after we close it below.
        if self._purge_thread is not None and self._purge_thread.is_alive():
            self._purge_stop_event.set()
            self._purge_thread.join(timeout=5.0)
            self._purge_thread = None

        # Wait for snapshot thread
        if self._snapshot_thread and self._snapshot_thread.is_alive():
            self._snapshot_thread.join(timeout=5.0)

        # Drain pending snapshot I/O futures (non-blocking cancel for tasks not
        # yet started; allow already-running writes to finish).
        self._snapshot_io.shutdown(wait=False, cancel_futures=False)

        # Clean up visualizer
        if self._visualizer:
            self._visualizer.cleanup()

        # Close display windows
        if self.enable_display:
            cv2.destroyAllWindows()

        # Clean up frame server
        if self.frame_server and hasattr(self.frame_server, 'destroy_node'):
            self.frame_server.destroy_node()

        # Stop content recorder
        if self._content_recorder is not None:
            try:
                self._content_recorder.stop()
            except Exception as e:
                logger.error(f"[ContainerCounterApp] Content recorder stop failed: {e}")
            self._content_recorder = None

        # Clean up ROS2 context if on RDK
        if IS_RDK:
            try:
                from src.ros2.IPC import shutdown_ros2_context
                shutdown_ros2_context()
            except Exception:
                pass

        # Close database to stop async writer thread.
        if self.db is not None:
            self.db.close()
            self.db = None

        logger.info("[ContainerCounterApp] Cleanup complete")
    
    def get_state(self) -> Dict:
        """Get current application state."""
        return self.state.to_dict()
