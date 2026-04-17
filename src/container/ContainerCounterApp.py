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

    # Max consecutive frames without a real detection before dropping prediction
    MAX_PRED_FRAMES = 10   # at detect_interval=5 → 50 frames (~2.5 s @ 20 fps)

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
                # Predicted outside frame — stop predicting but keep entry
                # (a real detection may bring it back into frame)
                logger.debug(
                    f"[Predictor] QR{qr_val} projected outside frame at step {step}, skipping"
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
    fps: int = 30
    
    # Tracking
    exit_zone_ratio: float = 0.15
    lost_timeout: float = 2.0
    min_displacement_ratio: float = 0.3
    detect_interval: int = 3  # Run QR detection every N-th frame
    # False-positive gate: a track must accumulate this many *real* QR
    # detections before it can emit an event.  Drops single-frame decoder
    # glitches that would otherwise count as containers.
    min_detections_for_event: int = 3

    # Snapshots
    pre_event_seconds: float = 5.0
    post_event_seconds: float = 5.0
    snapshot_dir: str = "data/container_snapshots"
    save_video: bool = False

    # QR-camera event video (used as primary when event_video_source="qr"
    # and as fallback when the content camera is unavailable).
    # These are **independent of** ``detect_interval`` \u2014 the event video
    # is sampled from every frame for smooth playback, not from detection
    # ticks.
    event_video_fps: int = 20            # sampling + output fps for QR video
    event_video_max_seconds: float = 5.0  # hard cap on per-track buffered history
    event_video_stationary_px: int = 5    # overwrite last frame if QR barely moved
    
    # State publishing
    state_file: str = "data/container_pipeline_state.json"
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
        # Detection runs every N-th frame; intermediate frames use linear prediction.
        # Will be overwritten with the config value in _init_components.
        self._detect_interval: int = self.config.detect_interval
        # Linear predictor for inter-detection frames (much lighter than cv2.Tracker)
        self._linear_predictor = _LinearPredictor(detect_interval=self._detect_interval)
        self._frame_size: tuple = (1280, 720)  # updated on first frame
        
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

        # Global pre-roll ring buffer for the QR camera.  Stores the last
        # few seconds of half-res frames *regardless* of whether a track
        # exists.  When the QR camera is the fallback video source, these
        # frames are prepended to give a ~3 s lead-in that the per-track
        # EventFrameBuffer alone cannot provide (it only starts recording
        # when the track is created).
        self._qr_preroll = EventFrameBuffer(EventFrameBufferConfig(
            target_fps=float(self.config.event_video_fps),
            max_seconds=3.0,
            stationary_px=0,   # global buffer — no positional dedup
        ))

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
        self.qr_detector = QRCodeDetector()
        
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
            self.frame_server = ContainerFrameServer()

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
            
            fps = self.config.fps or 25
            frame_interval = 1.0 / fps
            frame_count = 0
            
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
                
                frame_start = time.time()
                
                # Process frame
                self._process_frame(frame)
                frame_count += 1
                
                if frame_count % 100 == 0:
                    logger.info(
                        f"[ContainerCounterApp] {frame_count} frames processed "
                        f"| FPS={self.state.fps:.1f} "
                        f"| QR detections={self.state.qr_detections}"
                    )
                
                # Create annotated frame (for display or on-demand snapshot)
                annotated = self._annotate_frame(frame)
                
                # Check on-demand snapshot request
                self._maybe_capture_snapshot(frame, annotated)
                
                # Frame pacing + display
                elapsed = time.time() - frame_start
                remaining_ms = max(1, int((frame_interval - elapsed) * 1000))
                
                if self.enable_display and self._visualizer:
                    # waitKey handles both event-loop pumping and frame pacing
                    should_continue = self._visualizer.show(annotated, delay_ms=remaining_ms)
                    if not should_continue:
                        self._running = False
                        break
                else:
                    # Headless: use stop-event wait so SIGTERM exits without full sleep
                    if self._stop_event.wait(timeout=remaining_ms / 1000.0):
                        break
                
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

        if frame is None or frame.size == 0:
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
        self.snapshotter.add_frame(frame)

        # Always compute half-res frame for the global pre-roll and
        # per-track event-video buffers.  cv2.resize at 720p→360p is
        # <1 ms so the overhead is negligible.
        half = cv2.resize(frame, (w // 2, h // 2))
        self._qr_preroll.add(half, center_x=0)

        is_detect_frame = ((self.state.frame_count - 1) % self._detect_interval == 0)
        self._frame_detections = []   # reset per-frame viz list

        if is_detect_frame:
            # ── Full QR detection ──
            detections = self.qr_detector.detect_all(frame)
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

            active = self.tracker.get_active_tracks()
            for qr_val, center, bbox_xywh in self._linear_predictor.predict(step):
                # Only predict for QR values that have an active track
                if qr_val not in active:
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
            # If the last known position was already near an exit zone the
            # container did exit normally — the QR code simply became unreadable
            # before the strict exit boundary was crossed.  Only flag tracks
            # whose last position was clearly mid-frame.
            actually_lost = not self._is_exit_direction(event)
            reason = (
                f"mid-frame (exit_x={event.exit_x}, "
                f"zone=[left<={self.tracker.left_exit_x} / right>={self.tracker.right_exit_x}])"
                if actually_lost
                else f"near-exit (exit_x={event.exit_x})"
            )
            logger.info(
                f"[ContainerCounterApp] Lost-track resolved: "
                f"QR={event.qr_value} track=#{event.track_id} "
                f"dir={event.direction.value} "
                f"is_lost={actually_lost} reason={reason}"
            )
            self._handle_container_event(event, is_lost=actually_lost)

        # Handle completed snapshot captures
        completed = self.snapshotter.get_completed_captures()
        for capture in completed:
            try:
                self._snapshot_queue.put_nowait(capture)
            except queue.Full:
                logger.warning("[ContainerCounterApp] Snapshot queue full, dropping capture")

        # Per-track event-video buffering — every frame, after tracker has been
        # updated for this frame so ``track.last_x`` reflects the current
        # detection or prediction (not the previous frame's stale value).
        # The :class:`EventFrameBuffer` throttles to ``event_video_fps`` and
        # overwrites frames with minimal positional change, so memory is bounded.
        # ``half`` was already computed above for the pre-roll buffer.
        current_tracks = self.tracker.get_active_tracks()
        if current_tracks:
            for qr_val, track in current_tracks.items():
                buf = self._track_buffers.get(qr_val)
                if buf is None:
                    buf = self._EventFrameBuffer(self._EventFrameBufferConfig(
                        target_fps=float(self.config.event_video_fps),
                        max_seconds=float(self.config.event_video_max_seconds),
                        stationary_px=int(self.config.event_video_stationary_px),
                    ))
                    self._track_buffers[qr_val] = buf
                buf.add(half, center_x=track.last_x)

        # Reclaim buffers whose track ended (FP gate drop, normal exit, or lost).
        # ``current_tracks`` was just fetched so it reflects this frame's state.
        if self._track_buffers:
            stale = [q for q in self._track_buffers if q not in current_tracks]
            for q in stale:
                dead = self._track_buffers.pop(q, None)
                if dead is not None and len(dead) > 0:
                    logger.debug(
                        f"[ContainerCounterApp] Discarded buffer for QR={q} "
                        f"(track ended, frames={len(dead)})"
                    )

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

        # Update state
        self.state.active_tracks = len(current_tracks)
        self.state.pending_snapshots = self._snapshot_queue.qsize()
        self.state.processing_time_ms = (time.time() - start_time) * 1000
    
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
            active_tracks=self.tracker.get_active_tracks(),
            fps=self.state.fps,
            total_positive=self.state.total_positive,
            total_negative=self.state.total_negative,
            total_lost=self.state.total_lost,
            qr_positive=qr_pos,
            qr_negative=qr_neg,
            recent_events=events,
            exit_zone_ratio=self.config.exit_zone_ratio,
            frame_detections=self._frame_detections,
        )
        
        return annotated
    
    def _maybe_capture_snapshot(self, frame, annotated_frame) -> None:
        """
        Check if on-demand snapshot is requested via DB flag and capture.
        
        This enables /snapshot?camera=container in the web UI.
        Throttled to every 5th frame to avoid excessive DB reads.
        """
        if self.db is None or self._snapshot_writer is None:
            return
        
        if self.state.frame_count % 5 != 0:
            return
        
        try:
            requested = self.db.get_config(constants.container_snapshot_requested_key, "0")
            if requested != "1":
                return
            
            # Write snapshot (raw + annotated overlay)
            success = self._snapshot_writer.write_snapshot(
                frame=frame,
                frame_with_overlay=annotated_frame,
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
        track = self.tracker._tracks.get(qr_value)
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
        if content_entry and self._content_recorder is not None:
            content_event_id, begin_mono = content_entry
            self._content_recorder.end_event_recording(content_event_id)
            content_already_started = True
            # If elapsed time hit the safety cap the recording was truncated
            # before the container exited — flag it for operator monitoring.
            elapsed = time.monotonic() - begin_mono
            was_capped = elapsed >= (self.config.content_max_recording_seconds - 0.5)

        # ── Build QR fallback frame list (pre-roll + track buffer) ──
        buf = self._track_buffers.pop(event.qr_value, None)
        track_entries = buf.snapshot_frames() if buf is not None else []

        # Prepend frames from the global QR pre-roll ring buffer that
        # predate this track's entry time.  This gives the QR fallback
        # clip a ~3 s lead-in (matching what the content camera provides).
        if event.entry_time_monotonic > 0.0:
            preroll = [
                e for e in self._qr_preroll.snapshot_frames()
                if e[0] < event.entry_time_monotonic
            ]
        else:
            preroll = []

        all_entries = preroll + track_entries
        track_frames = [frame for _, frame, _ in all_entries]

        # Compute measured sampling rate from the combined timestamps so
        # the video writer uses the correct playback fps.
        qr_measured_fps: Optional[float] = None
        if len(all_entries) >= 2:
            span = all_entries[-1][0] - all_entries[0][0]
            if span > 0:
                mfps = len(all_entries) / span
                qr_measured_fps = max(1.0, min(
                    float(self.config.event_video_fps), mfps
                ))

        if buf is not None:
            logger.debug(
                f"[ContainerCounterApp] Track buffer QR={event.qr_value}: "
                f"{buf.stats()} preroll_frames={len(preroll)} "
                f"total={len(all_entries)} measured_fps="
                f"{f'{qr_measured_fps:.1f}' if qr_measured_fps else 'n/a'}"
            )

        # Use the threshold-time event_id for content recordings so the
        # DB record, content video file, and QR fallback all share the
        # same identifier.  For pure-QR events (no content started),
        # generate the event_id at exit time (original behaviour).
        event_id = content_event_id if content_event_id else self._make_event_id(event)

        # Dispatch video capture to the coordinator.
        video_result = self._capture_event_video(
            event_id=event_id,
            event=event,
            track_frames=track_frames,
            qr_measured_fps=qr_measured_fps,
            is_lost=is_lost,
            content_already_started=content_already_started,
        )

        # Determine recording status for operator monitoring:
        #   "ok"       — full transit successfully captured
        #   "capped"   — container dwelled > max_recording_seconds (video truncated)
        #   "fallback" — content camera unavailable; QR camera used instead
        #   "no_video" — no clip produced
        if video_result is None or video_result.video_relpath is None:
            recording_status = "no_video"
        elif video_result.fallback:
            recording_status = "fallback"
        elif content_already_started and was_capped:
            recording_status = "capped"
        else:
            recording_status = "ok"

        # Record the event (DB).
        self._record_event(
            event,
            event_id=event_id,
            is_lost=is_lost,
            video_result=video_result,
            recording_status=recording_status,
        )

        # Clear linear predictor for this QR so it doesn't ghost-predict.
        self._linear_predictor.remove(event.qr_value)

        # Add to recent events (for live dashboard).
        event_dict = {
            'timestamp': event.timestamp.isoformat(),
            'qr_value': event.qr_value,
            'direction': event.direction.value,
            'track_id': event.track_id,
            'duration': round(event.duration_seconds, 2),
            'is_lost': is_lost,
            'camera': video_result.camera if video_result else None,
            'fallback': bool(video_result.fallback) if video_result else False,
            'recording_status': recording_status,
        }
        self.state.add_event(event_dict)

        logger.info(
            f"[ContainerCounterApp] EVENT "
            f"QR={event.qr_value} track=#{event.track_id} "
            f"dir={event.direction.value} "
            f"lost={'YES ⚠' if is_lost else 'NO ✓'} "
            f"cam={video_result.camera if video_result else '—'}"
            f"{' (fallback)' if video_result and video_result.fallback else ''} "
            f"status={recording_status} "
            f"entry_x={event.entry_x} exit_x={event.exit_x} "
            f"disp={event.exit_x - event.entry_x:+}px "
            f"dur={event.duration_seconds:.2f}s "
            f"positions={len(event.positions)}"
        )

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

    def _is_exit_direction(self, event: ContainerEvent) -> bool:
        """
        Check if event was a proper exit (reached near the exit zone) vs truly lost.

        Uses a generous 5x multiplier on exit_zone_ratio so that containers
        whose QR becomes unreadable just before the strict exit boundary are
        still classified as exited rather than lost.

        Caps the generous ratio at 0.30 to avoid false-positives for very wide
        exit zones.
        """
        if self.tracker is None:
            return True
        frame_w = self.tracker.frame_width
        generous_ratio = min(self.tracker.exit_zone_ratio * 5, 0.30)
        left_threshold = int(frame_w * generous_ratio)
        right_threshold = int(frame_w * (1.0 - generous_ratio))
        exited = event.exit_x <= left_threshold or event.exit_x >= right_threshold
        logger.debug(
            f"[ContainerCounterApp] exit_check QR={event.qr_value} "
            f"track=#{event.track_id} "
            f"exit_x={event.exit_x} "
            f"strict_zone=[left<={self.tracker.left_exit_x} / right>={self.tracker.right_exit_x}] "
            f"generous_zone=[left<={left_threshold} / right>={right_threshold}] (ratio={generous_ratio:.0%}) "
            f"→ {'EXITED' if exited else 'MID-FRAME'}"
        )
        return exited

    def _update_fps(self) -> None:
        """Update FPS calculation."""
        self._fps_frames += 1
        elapsed = time.time() - self._fps_start_time
        
        if elapsed >= 1.0:
            self.state.fps = self._fps_frames / elapsed
            self._fps_frames = 0
            self._fps_start_time = time.time()
    
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
            
            state_data = {
                'timestamp': time.time(),
                'updated_at': time.time(),
                # Flatten state fields to top level for health monitoring
                **state_dict,
                # Sub-component stats
                'tracker': self.tracker.get_stats() if self.tracker else {},
                'snapshotter': self.snapshotter.get_stats() if self.snapshotter else {},
                'qr_detector': self.qr_detector.get_stats() if self.qr_detector else {},
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
    
    def stop(self) -> None:
        """Stop the application gracefully."""
        logger.info("[ContainerCounterApp] Stopping...")
        self._running = False
        self._stop_event.set()
    
    def _cleanup(self) -> None:
        """Clean up resources."""
        logger.info("[ContainerCounterApp] Cleaning up...")
        
        # Publish final state
        self._publish_state()
        
        # Wait for snapshot thread
        if self._snapshot_thread and self._snapshot_thread.is_alive():
            self._snapshot_thread.join(timeout=5.0)
        
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
