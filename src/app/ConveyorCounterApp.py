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

import time
import cv2
import signal
import os
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Dict, Callable
import numpy as np
import threading

from src.config.settings import AppConfig
from src.config.tracking_config import TrackingConfig
from src.utils.AppLogging import logger

from src.frame_source.FrameSourceFactory import FrameSourceFactory
from src.frame_source.FrameSource import FrameSource

from src.detection.BaseDetection import BaseDetector
from src.detection.DetectorFactory import DetectorFactory

from src.classifier.BaseClassifier import BaseClassifier
from src.classifier.ROICollectorService import ROICollectorService, ROIQualityConfig
from src.classifier.ClassificationWorker import ClassificationWorker
from src.classifier.ClassifierFactory import ClassifierFactory

from src.tracking.ConveyorTracker import ConveyorTracker
from src.tracking.BidirectionalSmoother import BidirectionalSmoother, ClassificationRecord

# Import modular pipeline components
from src.app.pipeline_core import PipelineCore
from src.app.pipeline_visualizer import PipelineVisualizer

from src.logging.Database import DatabaseManager


@dataclass
class CounterState:
    """Real-time state of the counter (thread-safe)."""
    total_counted: int = 0
    counts_by_class: Dict[str, int] = field(default_factory=dict)
    active_tracks: int = 0
    fps: float = 0.0
    processing_time_ms: float = 0.0
    last_count_time: float = 0.0

    def __post_init__(self):
        if self.counts_by_class is None:
            self.counts_by_class = {}
        self._lock = threading.Lock()

    def increment_count(self, class_name: str) -> int:
        """Thread-safe count increment."""
        with self._lock:
            self.total_counted += 1
            if class_name not in self.counts_by_class:
                self.counts_by_class[class_name] = 0
            self.counts_by_class[class_name] += 1
            self.last_count_time = time.time()
            return self.total_counted

    def get_counts_snapshot(self) -> Dict[str, int]:
        """Get thread-safe copy of counts."""
        with self._lock:
            return self.counts_by_class.copy()


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
        
        # State
        self.state = CounterState()
        self._running = False
        self._frame_count = 0
        self._start_time: Optional[float] = None
        
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
    
    def _init_components(self):
        """Initialize pipeline components with new modular architecture."""
        # Initialize database first to read config
        self._db = DatabaseManager(self.app_config.db_path)

        # Load enable_display from database config
        from src.constants import enable_display_key
        enable_display_str = self._db.get_config(enable_display_key, default='0')
        self.enable_display = enable_display_str == '1'
        logger.info(f"[ConveyorCounterApp] Display enabled: {self.enable_display} (from DB config)")

        # Frame source
        if self._frame_source is None:
            source_type = 'opencv'  # Default to OpenCV
            source = self.video_source or self.app_config.video_path
            self._frame_source = FrameSourceFactory.create(
                source_type,
                source=source,
                testing_mode=self.testing_mode
            )
        
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
            max_brightness=self.tracking_config.max_mean_brightness
        )
        self._roi_collector = ROICollectorService(
            quality_config=quality_config,
            max_rois_per_track=10
        )

        # Classification Worker
        self._classification_worker = ClassificationWorker(
            classifier=self._classifier,
            max_queue_size=100,
            name="ClassificationWorker"
        )
        self._classification_worker.start()

        # === NEW MODULAR COMPONENTS ===

        # Core Pipeline (handles detection, tracking, classification)
        self._pipeline_core = PipelineCore(
            detector=self._detector,
            tracker=self._tracker,
            roi_collector=self._roi_collector,
            classification_worker=self._classification_worker
        )
        # Set callback for classification completion
        self._pipeline_core.on_track_completed = self._on_classification_completed

        # Visualizer (handles display)
        if self.enable_display:
            self._pipeline_visualizer = PipelineVisualizer(
                tracking_config=self.tracking_config,
                window_name="Conveyor Counter",
                display_size=(960, 540)
            )

        # Bidirectional smoother
        self._smoother = BidirectionalSmoother(
            confidence_threshold=self.tracking_config.bidirectional_confidence_threshold,
            vote_ratio_threshold=self.tracking_config.evidence_ratio_threshold,
            batch_size=self.tracking_config.bidirectional_buffer_size,
            batch_timeout_seconds=self.tracking_config.bidirectional_inactivity_timeout_ms / 1000.0
        )

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
        
        # Use PipelineCore for all processing
        detections, active_tracks, rois_collected = self._pipeline_core.process_frame(frame)

        # Update state
        self.state.active_tracks = len(active_tracks)
        total_time = (time.perf_counter() - frame_start) * 1000
        self.state.processing_time_ms = total_time
        
        # Calculate FPS
        if self._start_time is not None:
            elapsed = time.perf_counter() - self._start_time
            self.state.fps = self._frame_count / elapsed if elapsed > 0 else 0
        
        # Visualization (if enabled)
        if self.enable_display and self._pipeline_visualizer:
            frame = self._pipeline_visualizer.annotate_frame(
                frame=frame,
                detections=detections,
                tracks=active_tracks,
                fps=self.state.fps,
                active_tracks=self.state.active_tracks,
                total_counted=self.state.total_counted,
                counts_by_class=self.state.get_counts_snapshot()
            )

        return frame
    
    def _on_classification_completed(self, track_id: int, class_name: str, confidence: float):
        """
        Callback when classification completes (called by PipelineCore from worker thread).

        Args:
            track_id: Track identifier
            class_name: Classified class
            confidence: Classification confidence
        """
        # Add to smoother (thread-safe)
        smoothed_batch = self._smoother.add_classification(
            track_id=track_id,
            class_name=class_name,
            confidence=confidence,
            vote_ratio=1.0  # Single classification, full confidence
        )

        # Process any finalized batch
        if smoothed_batch:
            for record in smoothed_batch:
                self._record_count(record, None)  # ROI already handled
        else:
            # Record immediately
            record = ClassificationRecord(
                track_id=track_id,
                class_name=class_name,
                confidence=confidence,
                vote_ratio=1.0,
                timestamp=time.time()
            )
            self._record_count(record, None)

    def _record_count(self, record: ClassificationRecord, roi: Optional[np.ndarray]):
        """Record a counted item."""
        # Update counts (thread-safe)
        new_total = self.state.increment_count(record.class_name)

        # Log to database
        if self._db is not None:
            try:
                # Optionally save ROI image for debugging
                if roi is not None:
                    roi_dir = os.path.join(self.tracking_config.spool_dir, "rois")
                    os.makedirs(roi_dir, exist_ok=True)
                    image_filename = f"track_{record.track_id}_{int(time.time() * 1000)}.jpg"
                    image_path = os.path.join(roi_dir, image_filename)
                    cv2.imwrite(image_path, roi)
                    logger.debug(f"[ConveyorCounterApp] Saved ROI for debugging: {image_path}")

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
            except Exception as e:
                logger.error(f"[ConveyorCounterApp] Failed to log event to database: {e}")

        # Callback
        if self._on_count_callback is not None:
            try:
                self._on_count_callback(record)
            except Exception as e:
                logger.error(f"[ConveyorCounterApp] Count callback error: {e}")

        # Log
        smoothed_str = f" (smoothed from {record.original_class})" if record.smoothed else ""
        logger.info(
            f"[ConveyorCounterApp] COUNT: {record.class_name} "
            f"(conf={record.confidence:.2f}, track={record.track_id}){smoothed_str}"
        )
        logger.info(f"[ConveyorCounterApp] Total: {new_total}, By class: {self.state.get_counts_snapshot()}")

    def run(self, max_frames: Optional[int] = None):
        """
        Run the counter application with modular architecture.

        Args:
            max_frames: Optional maximum frames to process (for testing)
        """
        logger.info("[ConveyorCounterApp] Starting with modular pipeline...")

        # Initialize components
        self._init_components()
        
        self._running = True
        self._start_time = time.perf_counter()
        self._frame_count = 0
        
        try:
            for frame, latency_ms in self._frame_source.frames():
                if not self._running:
                    break
                
                if max_frames and self._frame_count >= max_frames:
                    break
                
                self._frame_count += 1
                
                # Process frame through modular pipeline
                annotated = self._process_frame(frame)
                
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
        
        finally:
            self._cleanup()
    
    def _cleanup(self):
        """Clean up all resources with modular architecture."""
        logger.info("[ConveyorCounterApp] Cleaning up...")
        
        # Finalize any pending smoothing
        if self._smoother is not None:
            remaining = self._smoother.finalize_batch()
            for record in remaining:
                self._record_count(record, None)
            self._smoother.cleanup()
        
        # Clean up modular pipeline components
        if self._pipeline_core is not None:
            self._pipeline_core.cleanup()

        if self._pipeline_visualizer is not None:
            self._pipeline_visualizer.cleanup()

        # Release frame source
        if self._frame_source is not None:
            self._frame_source.cleanup()
        
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
