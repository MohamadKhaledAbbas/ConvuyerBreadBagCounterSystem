"""
Main Conveyor Bread Bag Counter Application.

This is the central orchestrator that:
1. Reads frames from source (OpenCV/ROS2)
2. Runs detection to find bread bags
3. Tracks objects through the frame
4. Classifies completed tracks
5. Applies bidirectional smoothing
6. Records video and logs events

Production Notes:
- Graceful shutdown handling
- Robust error recovery
- Thread-safe state management
- Memory-efficient frame processing
"""

import os
import signal
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Callable, List

import cv2
import numpy as np

from src.classifier.EnhancedROICollectorService import EnhancedROICollectorService, EnhancedROIQualityConfig

# Optional memory monitoring (gracefully degraded if psutil not available)
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    psutil = None
    HAS_PSUTIL = False

# Import modular pipeline components
from src.app.pipeline_core import PipelineCore
from src.app.pipeline_visualizer import PipelineVisualizer
from src.classifier.BaseClassifier import BaseClassifier
from src.classifier.ClassificationWorker import ClassificationWorker
from src.classifier.ClassifierFactory import ClassifierFactory
from src.classifier.ROICollectorService import ROICollectorService, ROIQualityConfig
from src.config.settings import AppConfig
from src.config.tracking_config import TrackingConfig
from src.constants import snapshot_requested_key
from src.detection.BaseDetection import BaseDetector
from src.detection.DetectorFactory import DetectorFactory
from src.endpoint.routes.snapshot import get_snapshot_writer, SnapshotWriter
from src.frame_source.FrameSource import FrameSource
from src.frame_source.FrameSourceFactory import FrameSourceFactory
from src.logging.Database import DatabaseManager
from src.endpoint.pipeline_state import write_state as write_pipeline_state
from src.tracking.BidirectionalSmoother import BidirectionalSmoother, ClassificationRecord
from src.tracking.RunLengthStateMachine import RunLengthStateMachine
from src.tracking.ConveyorTracker import ConveyorTracker
from src.utils.AppLogging import logger


@dataclass
class CounterState:
    """Real-time state of the counter (thread-safe) with two-tier counting."""
    # Confirmed counts (after smoothing, persisted to DB)
    total_counted: int = 0
    counts_by_class: Dict[str, int] = field(default_factory=dict)

    # Tentative counts (immediate, before smoothing)
    tentative_total: int = 0
    tentative_counts: Dict[str, int] = field(default_factory=dict)

    # Active tracking
    active_tracks: int = 0
    fps: float = 0.0
    processing_time_ms: float = 0.0
    last_count_time: float = 0.0

    # Debug state for visualization
    pending_classifications: int = 0  # Tracks waiting for classification
    pending_smoothing: int = 0  # Items in smoother batch
    last_classification: Optional[str] = None  # Last classified item info
    recent_events: list = field(default_factory=list)  # Recent events log (max 10)
    rejected_count: int = 0  # Total rejected classifications
    lost_track_count: int = 0  # Total lost track events

    # Smoothing statistics
    total_smoothed: int = 0  # Total items smoothed (changed)

    def __post_init__(self):
        if self.counts_by_class is None:
            self.counts_by_class = {}
        if self.tentative_counts is None:
            self.tentative_counts = {}
        if self.recent_events is None:
            self.recent_events = []
        self._lock = threading.Lock()

    def add_tentative(self, class_name: str) -> int:
        """Add tentative count (immediate, before smoothing)."""
        with self._lock:
            self.tentative_total += 1
            if class_name not in self.tentative_counts:
                self.tentative_counts[class_name] = 0
            self.tentative_counts[class_name] += 1
            return self.tentative_total

    def remove_tentative(self, class_name: str) -> int:
        """Remove tentative count when item is confirmed (after smoothing)."""
        with self._lock:
            if self.tentative_total > 0:
                self.tentative_total -= 1
            if class_name in self.tentative_counts and self.tentative_counts[class_name] > 0:
                self.tentative_counts[class_name] -= 1
            return self.tentative_total

    def increment_count(self, class_name: str) -> int:
        """Thread-safe count increment (confirmed, after smoothing)."""
        with self._lock:
            self.total_counted += 1
            if class_name not in self.counts_by_class:
                self.counts_by_class[class_name] = 0
            self.counts_by_class[class_name] += 1
            self.last_count_time = time.time()
            return self.total_counted

    def add_event(self, event: str):
        """Add event to recent events log (thread-safe)."""
        with self._lock:
            self.recent_events.append((time.time(), event))
            # Keep only last 10 events
            if len(self.recent_events) > 10:
                self.recent_events = self.recent_events[-10:]

    def get_counts_snapshot(self) -> Dict[str, int]:
        """Get thread-safe copy of confirmed counts."""
        with self._lock:
            return self.counts_by_class.copy()

    def get_tentative_snapshot(self) -> Dict[str, int]:
        """Get thread-safe copy of tentative counts."""
        with self._lock:
            return self.tentative_counts.copy()

    def get_recent_events(self) -> list:
        """Get thread-safe copy of recent events."""
        with self._lock:
            return self.recent_events.copy()


class ConveyorCounterApp:
    """
    Main application for bread bag counting on conveyor belt.
    
    Pipeline:
    Frame → Detect → Track → Classify → Smooth → Count
    
    Key difference from v1:
    - Simplified tracking (no state machine)
    - Classification happens AFTER track completes
    - Linear movement assumption
    """
    
    def __init__(
        self,
        app_config: Optional[AppConfig] = None,
        tracking_config: Optional[TrackingConfig] = None,
        video_source: Optional[str] = None,
        frame_source: Optional[FrameSource] = None,
        detector: Optional[BaseDetector] = None,
        classifier: Optional[BaseClassifier] = None,
        enable_ros2_publish: bool = False,
        testing_mode: bool = False
    ):
        """
        Initialize counter application.
        
        Args:
            app_config: Application configuration
            tracking_config: Tracking configuration
            video_source: Video source (file path, camera index, or RTSP URL)
            frame_source: Optional pre-configured frame source
            detector: Optional pre-configured detector
            classifier: Optional pre-configured classifier
            enable_ros2_publish: Enable ROS2 count publishing
            testing_mode: Enable testing mode (no frame drops)

        Note:
            enable_display is read from database config table (key: 'enable_display')
            Recording should be done separately via rtsp_h264_recorder.py
        """
        self.app_config = app_config or AppConfig()
        self.tracking_config = tracking_config or TrackingConfig()
        self.video_source = video_source
        self.testing_mode = testing_mode
        self.enable_display = False  # Will be loaded from DB
        self.enable_ros2 = enable_ros2_publish
        
        # Pipeline components (modular architecture)
        self._frame_source = frame_source
        self._detector = detector
        self._classifier = classifier
        self._tracker: Optional[ConveyorTracker] = None
        self._roi_collector: Optional[ROICollectorService] = None
        self._classification_worker: Optional[ClassificationWorker] = None

        # Modular pipeline components
        self._pipeline_core: Optional[PipelineCore] = None
        self._pipeline_visualizer: Optional[PipelineVisualizer] = None

        # Smoothing and database
        self._smoother: Optional[BidirectionalSmoother] = None
        self._db: Optional[DatabaseManager] = None
        
        # ROS2 executor (only used on RDK platform)
        self._ros_executor = None

        # ROI cache for saving by class (track_id -> best_roi)
        self._roi_cache: Dict[int, np.ndarray] = {}

        # State
        self.state = CounterState()
        self._running = False
        self._frame_count = 0
        self._start_time: Optional[float] = None
        
        # Last frame data for on-demand snapshot annotation (headless mode)
        self._last_detections = []
        self._last_tracks = []
        self._last_debug_info = {}

        # Current batch type tracking (stable: based on rolling window majority)
        self._current_batch_type: Optional[str] = None
        self._previous_batch_type: Optional[str] = None
        self._batch_type_window: List[str] = []
        self._batch_type_window_size: int = 7
        # Last individual classification (may flicker; used for fine-grained display)
        self._last_classified_type: Optional[str] = None

        # Memory monitoring
        self._last_memory_log_time: float = 0.0
        self._memory_log_interval: float = 60.0  # Log memory every 60 seconds

        # Callbacks
        self._on_count_callback: Optional[Callable] = None
        
        # Signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("[ConveyorCounterApp] Initialized")
    
    def _signal_handler(self, signum, _frame):
        """Handle shutdown signals."""
        logger.info(f"[ConveyorCounterApp] Received signal {signum}, shutting down...")
        self._running = False
    
    def _maybe_log_memory_usage(self):
        """Log memory usage periodically for diagnostics."""
        current_time = time.perf_counter()
        if current_time - self._last_memory_log_time < self._memory_log_interval:
            return

        self._last_memory_log_time = current_time

        if HAS_PSUTIL:
            try:
                process = psutil.Process()
                mem_info = process.memory_info()
                mem_mb = mem_info.rss / (1024 * 1024)

                # Get queue size if available
                queue_size = 0
                if hasattr(self._frame_source, 'queue') and self._frame_source.queue is not None:
                    queue_size = self._frame_source.queue.qsize()

                # Get pending classifications
                pending_classify = 0
                if self._classification_worker:
                    pending_classify = self._classification_worker.get_queue_size()

                logger.info(
                    f"[MEMORY] RSS={mem_mb:.1f}MB | "
                    f"frame_queue={queue_size} | "
                    f"pending_classify={pending_classify} | "
                    f"active_tracks={self.state.active_tracks} | "
                    f"frame={self._frame_count}"
                )
            except Exception as e:
                logger.debug(f"[MEMORY] Failed to get memory info: {e}")

    def _init_components(self):
        """Initialize pipeline components with new modular architecture."""
        # Initialize database first to read config
        self._db = DatabaseManager(self.app_config.db_path)

        # Load enable_display from database config
        from src.constants import enable_display_key, is_development_key
        enable_display_str = self._db.get_config(enable_display_key, default='0')
        self.enable_display = enable_display_str == '1'
        logger.info(f"[ConveyorCounterApp] Display enabled: {self.enable_display} (from DB config)")

        # Check if running in development mode (from database config)
        is_development_str = self._db.get_config(is_development_key, default='0')
        is_development = is_development_str == '1'
        logger.info(f"[ConveyorCounterApp] Development mode: {is_development} (from DB config)")

        # Frame source
        if self._frame_source is None:
            from src.utils.platform import IS_RDK

            if is_development:
                # Development mode: use OpenCV with video file
                source_type = 'opencv'
                source = self.video_source or self.app_config.video_path
                self._frame_source = FrameSourceFactory.create(
                    source_type,
                    source=source,
                    testing_mode=self.testing_mode,
                    queue_size=self.app_config.frame_queue_size,
                    target_fps=self.app_config.frame_target_fps
                )
                logger.info(f"[ConveyorCounterApp] Development mode: using OpenCV with {source}")
            elif IS_RDK:
                # Production on RDK: use ROS2 frame source
                # Initialize ROS2 context BEFORE creating the frame source
                import os
                os.environ["HOME"] = "/home/sunrise"
                from src.ros2.IPC import init_ros2_context
                self._ros_executor = init_ros2_context()

                source_type = 'ros2'
                self._frame_source = FrameSourceFactory.create(source_type)
                logger.info(f"[ConveyorCounterApp] Production mode: using ROS2 frame source")
            else:
                # Non-RDK (Windows/Linux): use OpenCV
                source_type = 'opencv'
                source = self.video_source or self.app_config.video_path
                self._frame_source = FrameSourceFactory.create(
                    source_type,
                    source=source,
                    testing_mode=self.testing_mode,
                    queue_size=self.app_config.frame_queue_size,
                    target_fps=self.app_config.frame_target_fps
                )
                logger.info(f"[ConveyorCounterApp] Windows/Linux mode: using OpenCV with {source}")

        # Detector
        if self._detector is None:
            self._detector = DetectorFactory.create(
                config=self.app_config,
                confidence_threshold=self.tracking_config.min_detection_confidence
            )
        
        # Classifier
        if self._classifier is None:
            self._classifier = ClassifierFactory.create(config=self.app_config)
        
        # Tracker
        self._tracker = ConveyorTracker(config=self.tracking_config)

        # ROI Collector
        quality_config = ROIQualityConfig(
            min_sharpness=self.tracking_config.min_sharpness,
            min_brightness=self.tracking_config.min_mean_brightness,
            max_brightness=self.tracking_config.max_mean_brightness,
            # Brightness quality factor (shadow protection)
            optimal_brightness=self.tracking_config.optimal_brightness,
            brightness_penalty_weight=self.tracking_config.brightness_penalty_weight,
            # Diversity controls
            min_frame_spacing=self.tracking_config.roi_min_frame_spacing,
            min_position_change=self.tracking_config.roi_min_position_change,
            # Gradual position penalty
            enable_gradual_position_penalty=self.tracking_config.enable_gradual_position_penalty,
            position_penalty_start_ratio=self.tracking_config.position_penalty_start_ratio,
            position_penalty_max_ratio=self.tracking_config.position_penalty_max_ratio,
            position_penalty_min_multiplier=self.tracking_config.position_penalty_min_multiplier
        )
        self._roi_collector = ROICollectorService(
            quality_config=quality_config,
            max_rois_per_track=10,
            save_roi_candidates=self.tracking_config.save_roi_candidates,
            save_all_rois=self.tracking_config.save_all_rois,
            roi_candidates_dir=self.tracking_config.roi_candidates_dir,
            enable_temporal_weighting=self.tracking_config.enable_temporal_weighting,
            temporal_decay_rate=self.tracking_config.temporal_decay_rate
        )

        # Classification Worker
        self._classification_worker = ClassificationWorker(
            classifier=self._classifier,
            max_queue_size=100,
            name="ClassificationWorker",
            db=self._db
        )
        self._classification_worker.start()

        # === NEW MODULAR COMPONENTS ===

        # Core Pipeline (handles detection, tracking, classification)
        self._pipeline_core = PipelineCore(
            detector=self._detector,
            tracker=self._tracker,
            roi_collector=self._roi_collector,
            classification_worker=self._classification_worker,
            db=self._db,
            tracking_config=self.tracking_config
        )
        # Set callback for classification completion
        self._pipeline_core.on_track_completed = self._on_classification_completed
        # Set callback for track events (for UI debugging)
        self._pipeline_core.on_track_event = self._on_track_event

        # Visualizer - always create for snapshot support (even in headless mode)
        # In headless mode, it's only used for on-demand snapshot annotation
        self._pipeline_visualizer = PipelineVisualizer(
            tracking_config=self.tracking_config,
            window_name="Conveyor Counter",
            display_size=(1280, 720)
        )

        # Bidirectional smoother (uses sliding window approach)
        algorithm = getattr(self.tracking_config, 'smoothing_algorithm', 'run_length')
        if algorithm == 'window':
            self._smoother = BidirectionalSmoother(
                confidence_threshold=self.tracking_config.bidirectional_confidence_threshold,
                vote_ratio_threshold=self.tracking_config.evidence_ratio_threshold,
                window_size=self.tracking_config.bidirectional_buffer_size,
                window_timeout_seconds=self.tracking_config.bidirectional_inactivity_timeout_ms / 1000.0
            )
            logger.info("[ConveyorCounterApp] Smoother: BidirectionalSmoother (window algorithm)")
        else:
            self._smoother = RunLengthStateMachine(
                min_run_length=self.tracking_config.run_length_min_run,
                max_blip=self.tracking_config.run_length_max_blip,
                transition_confirm_count=self.tracking_config.run_length_transition_confirm_count,
                window_timeout_seconds=self.tracking_config.bidirectional_inactivity_timeout_ms / 1000.0
            )
            logger.info("[ConveyorCounterApp] Smoother: RunLengthStateMachine (run_length algorithm)")

        logger.info("[ConveyorCounterApp] Components initialized with modular architecture")

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame through the modular pipeline.

        Args:
            frame: Input BGR frame
            
        Returns:
            Annotated frame for display
        """
        frame_start = time.perf_counter()
        
        # Resize to standard resolution (done in consumer, not producer thread)
        # This matches v1 behavior and avoids GIL contention
        if frame.shape[:2] != (720, 1280):
            frame = cv2.resize(frame, (1280, 720))

        # Get NV12 data from frame source for native BPU detection (avoids
        # NV12→BGR→NV12 round-trip). Only Ros2FrameServer provides this.
        nv12_data = None
        frame_size = None
        if hasattr(self._frame_source, 'get_last_nv12_data'):
            nv12_data, frame_size = self._frame_source.get_last_nv12_data()

        # Use PipelineCore for all processing
        detections, active_tracks, rois_collected = self._pipeline_core.process_frame(
            frame, nv12_data=nv12_data, frame_size=frame_size
        )

        # Update state
        self.state.active_tracks = len(active_tracks)
        total_time = (time.perf_counter() - frame_start) * 1000
        self.state.processing_time_ms = total_time
        
        # Update pending classification count from worker queue
        if self._classification_worker:
            self.state.pending_classifications = self._classification_worker.get_queue_size()

        # Calculate FPS
        if self._start_time is not None:
            elapsed = time.perf_counter() - self._start_time
            self.state.fps = self._frame_count / elapsed if elapsed > 0 else 0
        
        # Build debug info for visualization (used for both display and snapshot)
        debug_info = {
            'pending_classify': self.state.pending_classifications,
            'pending_smooth': self.state.pending_smoothing,
            'rejected': self.state.rejected_count,
            'last_class': self.state.last_classification,
            'recent_events': self.state.get_recent_events(),
            'rois_collected': rois_collected,
            'processing_ms': self.state.processing_time_ms,
            'tentative_total': self.state.tentative_total,
            'tentative_counts': self.state.get_tentative_snapshot(),
            'lost_track_count': self.state.lost_track_count
        }

        # Add tracker statistics if available
        if hasattr(self._pipeline_core, 'tracker') and self._pipeline_core.tracker:
            tracker_stats = self._pipeline_core.tracker.get_statistics()
            debug_info['tracks_created'] = tracker_stats.get('tracks_created', 0)
            debug_info['duplicates_prevented'] = tracker_stats.get('duplicates_prevented', 0)
            debug_info['ghost_tracks'] = tracker_stats.get('ghost_tracks', 0)

        # Add state-machine info for visualizer overlay
        if isinstance(self._smoother, RunLengthStateMachine):
            sm = self._smoother
            in_transition = sm.state == RunLengthStateMachine.TRANSITION
            debug_info['sm_info'] = {
                'algorithm':        'run_length',
                'state':            sm.state,
                'batch_class':      sm.confirmed_batch_class,
                'run_class':        sm.current_run_class,
                'run_length':       sm.current_run_length,
                'run_target':       (sm.transition_confirm_count if in_transition
                                     else sm.min_run_length),
                'last_decision':    sm.last_decision_reason,
                'transition_count': len(sm.transition_history),
                'total_smoothed':   sm.smoothed_records,
                'total_records':    sm.total_records,
                'max_blip':         sm.max_blip,
            }

        # Store detection data for on-demand snapshot annotation
        self._last_detections = detections
        self._last_tracks = active_tracks
        self._last_debug_info = debug_info

        # Visualization (if display enabled)
        if self.enable_display and self._pipeline_visualizer:
            # Get ghost tracks for visualization
            ghost_tracks = None
            if hasattr(self._pipeline_core, 'tracker') and self._pipeline_core.tracker:
                ghost_tracks = self._pipeline_core.tracker.get_ghost_tracks_for_visualization()

            frame = self._pipeline_visualizer.annotate_frame(
                frame=frame,
                detections=detections,
                tracks=active_tracks,
                fps=self.state.fps,
                active_tracks=self.state.active_tracks,
                total_counted=self.state.total_counted,
                counts_by_class=self.state.get_counts_snapshot(),
                debug_info=debug_info,
                ghost_tracks=ghost_tracks
            )

        return frame
    
    def _on_track_event(self, event: str):
        """
        Callback for track-related events (for UI debugging).

        Args:
            event: Event description string
        """
        self.state.add_event(event)

        # Count lost track events
        if "lost before exit" in event:
            with self.state._lock:
                self.state.lost_track_count += 1

    def _on_classification_completed(self, track_id: int, class_name: str, confidence: float, non_rejected_rois: int = 0, best_roi: Optional[np.ndarray] = None):
        """
        Callback when classification completes (called by PipelineCore from worker thread).

        Args:
            track_id: Track identifier
            class_name: Classified class
            confidence: Classification confidence
            non_rejected_rois: Number of non-rejected ROIs (for trustworthiness)
            best_roi: Best quality ROI for saving by class
        """
        # Update debug state
        self.state.last_classification = f"T{track_id}:{class_name}({confidence:.2f})"
        self.state.add_event(f"CLASSIFY T{track_id}->{class_name} ({confidence:.2f}) rois={non_rejected_rois}")

        # Track last individual classification
        self._last_classified_type = class_name

        # Stable batch type from rolling window majority
        # Only reliable, non-Rejected classifications inform the batch type
        # so a single misclassification doesn't cause the display to flicker.
        min_trusted_rois = 3
        if non_rejected_rois >= min_trusted_rois and class_name != 'Rejected':
            self._batch_type_window.append(class_name)
            if len(self._batch_type_window) > self._batch_type_window_size:
                self._batch_type_window.pop(0)
            counter = Counter(self._batch_type_window)
            dominant, _ = counter.most_common(1)[0]
            # Track previous batch on transition
            if self._current_batch_type and dominant != self._current_batch_type:
                self._previous_batch_type = self._current_batch_type
            self._current_batch_type = dominant

        # Check if classification is reliable (needs >= 3 non-rejected ROIs)
        original_class = class_name

        if non_rejected_rois < min_trusted_rois and class_name != 'Rejected':
            # Too few good ROIs - override to 'Rejected' (unreliable classification)
            logger.warning(
                f"[CLASSIFICATION] T{track_id} UNRELIABLE | "
                f"original={class_name} non_rejected_rois={non_rejected_rois} < {min_trusted_rois} "
                f"action=override_to_Rejected"
            )
            class_name = 'Rejected'
            confidence = 0.0  # Mark as very low confidence
            self.state.add_event(f"UNRELIABLE T{track_id}:{original_class}->Rejected (rois={non_rejected_rois})")

        # IMMEDIATE: Add tentative count (shown in UI immediately)
        # Note: We add ALL classifications (including 'Rejected') to tentative
        # because they go through smoothing and might be overridden
        tentative_total = self.state.add_tentative(class_name)
        self.state.add_event(f"TENTATIVE T{track_id}:{class_name} (pending smoothing)")
        logger.info(
            f"[TENTATIVE_COUNT] T{track_id} {class_name} | "
            f"conf={confidence:.3f} non_rejected_rois={non_rejected_rois} "
            f"tentative_total={tentative_total} status=awaiting_batch_smoothing"
        )

        # Add to smoother (thread-safe) - uses sliding window approach
        # Returns a single confirmed record when window is full, None otherwise
        # Note: Low confidence items (including 'Rejected') will still be smoothed by window context
        confirmed_record = self._smoother.add_classification(
            track_id=track_id,
            class_name=class_name,
            confidence=confidence,
            vote_ratio=1.0,  # Single classification, full confidence
            non_rejected_rois=non_rejected_rois
        )

        # For RunLengthStateMachine, derive current_batch_type from confirmed_batch_class
        if isinstance(self._smoother, RunLengthStateMachine):
            rlsm_batch = self._smoother.confirmed_batch_class
            if rlsm_batch and rlsm_batch != self._current_batch_type:
                self._previous_batch_type = self._current_batch_type
                self._current_batch_type = rlsm_batch

        # Update pending smoothing count
        self.state.pending_smoothing = len(self._smoother.get_pending_records())

        # Process if a record was confirmed (sliding window filled)
        if confirmed_record:
            self.state.add_event(
                f"CONFIRMED (window full): T{confirmed_record.track_id}->"
                f"{confirmed_record.class_name} smoothed={confirmed_record.smoothed}"
            )
            self._record_confirmed_count(confirmed_record, best_roi)

        # Publish pipeline state for real-time visibility
        self._publish_pipeline_state()

    def _save_roi_by_class(self, roi: np.ndarray, record: ClassificationRecord):
        """
        Save ROI image organized by classification result.

        Creates subdirectories for each class type:
        - roi_candidates_dir/ClassName1/
        - roi_candidates_dir/ClassName2/
        - roi_candidates_dir/Rejected/

        Args:
            roi: ROI image
            record: Classification record with class name and metadata
        """
        try:
            # Sanitize class name for directory
            class_name_safe = record.class_name.replace(" ", "_").replace("/", "_")

            # Create class-specific subdirectory
            class_dir = os.path.join(self.tracking_config.roi_candidates_dir, class_name_safe)
            os.makedirs(class_dir, exist_ok=True)

            # Create filename with metadata
            timestamp = int(time.time() * 1000)
            smoothed_suffix = "_smoothed" if record.smoothed else ""
            filename = (
                f"track_{record.track_id}_{timestamp}_"
                f"{class_name_safe}_conf{record.confidence:.2f}"
                f"{smoothed_suffix}.jpg"
            )

            filepath = os.path.join(class_dir, filename)
            cv2.imwrite(filepath, roi)

            logger.debug(
                f"[ConveyorCounterApp] Saved classified ROI: "
                f"{class_name_safe}/{filename}"
            )

        except Exception as e:
            logger.error(f"[ConveyorCounterApp] Failed to save ROI by class: {e}")

    def _record_confirmed_count(self, record: ClassificationRecord, roi: Optional[np.ndarray]):
        """
        Record a CONFIRMED count (after smoothing, persisted to DB).

        Note: 'Rejected' class is treated as a quality indicator, not a bag type.
        However, 'Rejected' items ARE included in the sliding window and can be
        overridden by smoothing. Only items that remain 'Rejected' after smoothing
        are excluded from the count.
        """
        # Remove from tentative count (use original class if smoothed)
        tentative_class = record.original_class if record.smoothed else record.class_name
        remaining_tentative = self.state.remove_tentative(tentative_class)

        # Skip 'Rejected' class only if it wasn't overridden by smoothing
        if record.class_name == 'Rejected':
            self.state.rejected_count += 1
            self.state.add_event(f"REJECTED T{record.track_id} (excluded)")
            logger.info(
                f"[CONFIRMED_COUNT] T{record.track_id} REJECTED | "
                f"conf={record.confidence:.3f} total_rejected={self.state.rejected_count} "
                f"remaining_tentative={remaining_tentative} status=excluded_from_count"
            )
            # Save rejected ROI if enabled
            if roi is not None and self.tracking_config.save_rois_by_class:
                self._save_roi_by_class(roi, record)
            return

        # Update CONFIRMED counts (thread-safe)
        new_total = self.state.increment_count(record.class_name)

        # Get current class count
        class_count = self.state.get_counts_snapshot().get(record.class_name, 0)

        # Track if smoothed
        if record.smoothed:
            self.state.total_smoothed += 1
            self.state.add_event(f"CONFIRMED T{record.track_id}:{record.class_name} (was {record.original_class})")
            smoothed_str = f"smoothed_from={record.original_class}"

            logger.info(
                f"[CONFIRMED_COUNT] T{record.track_id} SMOOTHED | "
                f"original={record.original_class} final={record.class_name} conf={record.confidence:.3f} "
                f"class_total={class_count} system_total={new_total} "
                f"status=confirmed_and_persisted"
            )
        else:
            self.state.add_event(f"CONFIRMED T{record.track_id}:{record.class_name}")
            smoothed_str = "smoothed=no"

            logger.info(
                f"[CONFIRMED_COUNT] T{record.track_id} COUNTED | "
                f"class={record.class_name} conf={record.confidence:.3f} {smoothed_str} "
                f"class_total={class_count} system_total={new_total} "
                f"status=confirmed_and_persisted"
            )

        # Save ROI by class if enabled
        if roi is not None and self.tracking_config.save_rois_by_class:
            self._save_roi_by_class(roi, record)

        # Log to database
        if self._db is not None:
            try:
                import json
                self._db.add_event(
                    timestamp=datetime.fromtimestamp(record.timestamp).isoformat(),
                    bag_type_name=record.class_name,
                    confidence=record.confidence,
                    track_id=record.track_id,
                    metadata=json.dumps({
                        'vote_ratio': record.vote_ratio,
                        'smoothed': record.smoothed,
                        'original_class': record.original_class
                    })
                )
                # Log state-machine decision detail for RunLengthStateMachine
                if isinstance(self._smoother, RunLengthStateMachine):
                    self._db.enqueue_write(
                        """INSERT INTO track_event_details
                           (track_id, timestamp, step_type, class_name, confidence,
                            is_rejected, detail)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (
                            record.track_id,
                            datetime.fromtimestamp(record.timestamp).isoformat(),
                            'smoothing_decision',
                            record.class_name,
                            record.confidence,
                            1 if record.class_name == 'Rejected' else 0,
                            json.dumps({
                                'algorithm':      'run_length',
                                'state':          self._smoother.state,
                                'run_class':      self._smoother.current_run_class,
                                'run_length':     self._smoother.current_run_length,
                                'decision':       'smoothed' if record.smoothed else 'confirmed',
                                'original_class': record.original_class,
                                'final_class':    record.class_name,
                            })
                        )
                    )
            except Exception as e:
                logger.error(f"[ConveyorCounterApp] Failed to log event to database: {e}")

        # Callback
        if self._on_count_callback is not None:
            try:
                self._on_count_callback(record)
            except Exception as e:
                logger.error(f"[ConveyorCounterApp] Count callback error: {e}")

    def _publish_pipeline_state(self):
        """
        Write current pipeline state to shared file for the FastAPI server.

        Called after each classification or confirmation to keep the real-time
        counts endpoint up to date.
        """
        try:
            smoother_stats = self._smoother.get_statistics()
            pending_summary = self._smoother.get_pending_summary()
            pending_total = smoother_stats['pending_in_window']
            window_size = self._smoother.window_size
            next_confirmation_in = max(0, window_size - pending_total)

            # Build recent events for live feed
            raw_events = self.state.get_recent_events()
            recent_events = [
                {"ts": ts, "msg": msg} for ts, msg in raw_events
            ]

            # Build state-machine insight block
            if isinstance(self._smoother, RunLengthStateMachine):
                state_machine = {
                    "algorithm":              "run_length",
                    "state":                  self._smoother.state,
                    "confirmed_batch_class":  self._smoother.confirmed_batch_class,
                    "current_run_class":      self._smoother.current_run_class,
                    "current_run_length":     self._smoother.current_run_length,
                    "run_target":             self._smoother.min_run_length,
                    "transition_confirm_count": self._smoother.transition_confirm_count,
                    "max_blip":               self._smoother.max_blip,
                    "last_decision":          self._smoother.last_decision_reason,
                    "total_smoothed":         self._smoother.smoothed_records,
                    "total_records":          self._smoother.total_records,
                }
                transition_history = self._smoother.transition_history[-5:]
            else:
                state_machine = {
                    "algorithm":              "window",
                    "state":                  "WINDOW",
                    "confirmed_batch_class":  self._current_batch_type,
                    "current_run_class":      self._smoother.get_dominant_class(),
                    "current_run_length":     len(self._smoother.window_buffer),
                    "run_target":             self._smoother.window_size,
                    "transition_confirm_count": self._smoother.window_size,
                    "max_blip":               0,
                    "last_decision":          None,
                }
                transition_history = []

            state = {
                "confirmed": self.state.get_counts_snapshot(),
                "pending": pending_summary,
                "just_classified": self.state.get_tentative_snapshot(),
                "confirmed_total": self.state.total_counted,
                "pending_total": pending_total,
                "just_classified_total": self.state.tentative_total,
                "smoothing_rate": smoother_stats['smoothing_rate'],
                "window_status": {
                    "size": window_size,
                    "current_items": pending_total,
                    "next_confirmation_in": next_confirmation_in
                },
                "recent_events": recent_events,
                "current_batch_type": self._current_batch_type,
                "previous_batch_type": self._previous_batch_type,
                "last_classified_type": self._last_classified_type,
                "state_machine": state_machine,
                "transition_history": transition_history,
            }
            write_pipeline_state(state)
        except Exception as e:
            logger.debug(f"[ConveyorCounterApp] Failed to publish pipeline state: {e}")

    def _maybe_capture_snapshot(
        self,
        snapshot_writer: SnapshotWriter,
        frame: np.ndarray,
        annotated_frame: np.ndarray
    ):
        """
        Check if snapshot is requested and capture if so.

        On-demand snapshot capture:
        1. Check snapshot_requested flag in database
        2. If "1", capture frame and write to disk
        3. Set flag back to "0" immediately

        For headless mode (enable_display=False), creates annotated frame on-demand
        using the stored detection/tracking data.

        Args:
            snapshot_writer: SnapshotWriter instance
            frame: Raw BGR frame
            annotated_frame: Frame with detection overlays (may be same as frame in headless mode)
        """
        try:
            # Check if snapshot is requested
            if self._db is None:
                return

            requested = self._db.get_config(snapshot_requested_key, "0")
            if requested != "1":
                return

            # In headless mode, annotated_frame is same as raw frame
            # Create annotated version on-demand for snapshot
            frame_with_overlay = annotated_frame
            if not self.enable_display and self._pipeline_visualizer is not None:
                # Get ghost tracks for visualization
                ghost_tracks = None
                if hasattr(self._pipeline_core, 'tracker') and self._pipeline_core.tracker:
                    ghost_tracks = self._pipeline_core.tracker.get_ghost_tracks_for_visualization()

                # Create annotated frame on-demand
                frame_with_overlay = self._pipeline_visualizer.annotate_frame(
                    frame=frame.copy(),  # Copy to avoid modifying original
                    detections=self._last_detections,
                    tracks=self._last_tracks,
                    fps=self.state.fps,
                    active_tracks=self.state.active_tracks,
                    total_counted=self.state.total_counted,
                    counts_by_class=self.state.get_counts_snapshot(),
                    debug_info=self._last_debug_info,
                    ghost_tracks=ghost_tracks
                )

            # Capture snapshot
            success = snapshot_writer.write_snapshot(
                frame=frame,
                frame_with_overlay=frame_with_overlay,
                frame_number=self._frame_count
            )

            # Clear the flag immediately (regardless of success)
            self._db.set_config(snapshot_requested_key, "0")

            if success:
                logger.debug(f"[ConveyorCounterApp] Snapshot captured at frame {self._frame_count}")
            else:
                logger.warning(f"[ConveyorCounterApp] Snapshot capture failed at frame {self._frame_count}")

        except Exception as e:
            # Don't let snapshot errors crash the main loop
            logger.error(f"[ConveyorCounterApp] Snapshot error: {e}")
            # Try to clear the flag anyway
            try:
                if self._db:
                    self._db.set_config(snapshot_requested_key, "0")
            except Exception:
                pass

    def run(self, max_frames: Optional[int] = None):
        """
        Run the counter application with modular architecture.

        Args:
            max_frames: Optional maximum frames to process (for testing)
        """
        logger.info("[ConveyorCounterApp] Starting with modular pipeline...")

        # Initialize components
        self._init_components()
        
        # Reset pipeline state file to clear old data (ensures counts page shows only today's data)
        ws = self._smoother.window_size if self._smoother else 7
        initial_state = {
            "confirmed": {},
            "pending": {},
            "just_classified": {},
            "confirmed_total": 0,
            "pending_total": 0,
            "just_classified_total": 0,
            "smoothing_rate": 0.0,
            "window_status": {
                "size": ws,
                "current_items": 0,
                "next_confirmation_in": ws
            },
            "recent_events": [],
            "current_batch_type": None,
            "previous_batch_type": None,
            "last_classified_type": None,
            "state_machine": {
                "algorithm": "run_length" if isinstance(self._smoother, RunLengthStateMachine) else "window",
                "state": "ACCUMULATING",
                "confirmed_batch_class": None,
                "current_run_class": None,
                "current_run_length": 0,
                "run_target": ws,
                "transition_confirm_count": ws,
                "max_blip": 0,
                "last_decision": None,
            },
            "transition_history": [],
        }
        write_pipeline_state(initial_state)
        logger.info("[ConveyorCounterApp] Pipeline state reset - counts page will show today's data only")

        self._running = True
        self._start_time = time.perf_counter()
        self._frame_count = 0
        
        try:
            # Get snapshot writer for on-demand browser-based viewing
            snapshot_writer = get_snapshot_writer()

            for frame, latency_ms in self._frame_source.frames():
                if not self._running:
                    break
                
                if max_frames and self._frame_count >= max_frames:
                    break
                
                self._frame_count += 1
                
                # Process frame through modular pipeline
                annotated = self._process_frame(frame)
                
                # Check if snapshot is requested (on-demand via database flag)
                self._maybe_capture_snapshot(snapshot_writer, frame, annotated)

                # Display (delegated to PipelineVisualizer)
                if self.enable_display and self._pipeline_visualizer:
                    should_continue = self._pipeline_visualizer.show(annotated)
                    if not should_continue:
                        self._running = False
                
                # Log progress periodically
                if self._frame_count % 100 == 0:
                    logger.info(
                        f"[ConveyorCounterApp] Frame {self._frame_count}: "
                        f"FPS={self.state.fps:.1f}, Counted={self.state.total_counted}"
                    )

                # Log memory usage periodically for diagnostics
                self._maybe_log_memory_usage()

                # Check for smoother timeout and flush pending items if no activity
                if self._smoother:
                    timed_out_records = self._smoother.check_timeout()
                    for record in timed_out_records:
                        self.state.add_event(
                            f"CONFIRMED (timeout): T{record.track_id}->"
                            f"{record.class_name} smoothed={record.smoothed}"
                        )
                        self._record_confirmed_count(record, None)
                    if timed_out_records:
                        self._publish_pipeline_state()

        finally:
            self._cleanup()
    
    def _cleanup(self):
        """Clean up all resources with modular architecture."""
        logger.info("[ConveyorCounterApp] Cleaning up...")
        
        # Finalize any pending smoothing
        if self._smoother is not None:
            remaining = self._smoother.finalize_batch()
            for record in remaining:
                self._record_confirmed_count(record, None)
            self._smoother.cleanup()
        
        # Clean up modular pipeline components
        if self._pipeline_core is not None:
            self._pipeline_core.cleanup()

        if self._pipeline_visualizer is not None:
            self._pipeline_visualizer.cleanup()

        # Release frame source
        if self._frame_source is not None:
            self._frame_source.cleanup()
        
        # Shutdown ROS2 context if initialized
        if self._ros_executor is not None:
            try:
                from src.ros2.IPC import shutdown_ros2_context
                shutdown_ros2_context()
                logger.info("[ConveyorCounterApp] ROS2 context shutdown complete")
            except Exception as e:
                logger.warning(f"[ConveyorCounterApp] ROS2 shutdown error (ignored): {e}")

        # Close database
        if self._db is not None:
            self._db.close()
        
        # Close display
        if self.enable_display:
            cv2.destroyAllWindows()
        
        # Log final statistics
        total_time = time.perf_counter() - self._start_time if self._start_time else 0
        
        logger.info("=" * 50)
        logger.info("[ConveyorCounterApp] Final Statistics:")
        logger.info(f"  Total frames: {self._frame_count}")
        logger.info(f"  Total time: {total_time:.1f}s")
        logger.info(f"  Average FPS: {self._frame_count / total_time:.1f}" if total_time > 0 else "  Average FPS: N/A")
        logger.info(f"  Total counted: {self.state.total_counted}")
        logger.info(f"  Counts by class: {self.state.counts_by_class}")
        
        if self._smoother is not None:
            stats = self._smoother.get_statistics()
            logger.info(f"  Smoothing rate: {stats['smoothing_rate']:.1%}")
        
        logger.info("=" * 50)
    
    def set_on_count_callback(self, callback: Callable):
        """Set callback for count events."""
        self._on_count_callback = callback
    
    def get_state(self) -> CounterState:
        """Get current counter state."""
        return self.state
    
    def stop(self):
        """Stop the counter application."""
        self._running = False
