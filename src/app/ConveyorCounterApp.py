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
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Callable
import numpy as np
import threading

from src.config.settings import AppConfig
from src.config.tracking_config import TrackingConfig
from src.utils.AppLogging import logger, StructuredLogger
from src.utils.PipelineMetrics import PipelineMetrics

from src.frame_source.FrameSourceFactory import FrameSourceFactory
from src.frame_source.FrameSource import FrameSource

from src.detection.BaseDetection import BaseDetector, Detection
from src.detection.DetectorFactory import DetectorFactory

from src.classifier.BaseClassifier import BaseClassifier
from src.classifier.ROICollectorService import ROICollectorService, ROIQualityConfig
from src.classifier.ClassificationWorker import ClassificationWorker
from src.classifier.ClassifierFactory import ClassifierFactory

from src.tracking.ConveyorTracker import ConveyorTracker, TrackedObject, TrackEvent
from src.tracking.BidirectionalSmoother import BidirectionalSmoother, ClassificationRecord

from src.logging.Database import DatabaseManager
from src.spool.segment_io import SegmentWriter
from src.spool.retention import RetentionPolicy, RetentionConfig


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
        enable_recording: bool = True,
        enable_display: bool = True,
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
            enable_recording: Enable video spool recording
            enable_display: Enable visualization window
            enable_ros2_publish: Enable ROS2 count publishing
            testing_mode: Enable testing mode (no frame drops)
        """
        self.app_config = app_config or AppConfig()
        self.tracking_config = tracking_config or TrackingConfig()
        self.video_source = video_source
        self.testing_mode = testing_mode
        self.enable_display = enable_display
        self.enable_recording = enable_recording
        self.enable_ros2 = enable_ros2_publish
        
        # Pipeline components (lazy init)
        self._frame_source = frame_source
        self._detector = detector
        self._classifier = classifier
        self._roi_collector: Optional[ROICollectorService] = None  # Collects ROIs only
        self._classification_worker: Optional[ClassificationWorker] = None  # Async classification
        self._tracker: Optional[ConveyorTracker] = None
        self._smoother: Optional[BidirectionalSmoother] = None
        
        # Recording components
        self._segment_writer: Optional[SegmentWriter] = None
        self._retention_policy: Optional[RetentionPolicy] = None
        
        # Database
        self._db: Optional[DatabaseManager] = None
        
        # Structured logging
        self._structured_logger = StructuredLogger(logger)
        
        # Metrics
        self._metrics = PipelineMetrics()
        
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
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"[ConveyorCounterApp] Received signal {signum}, shutting down...")
        self._running = False
    
    def _init_components(self):
        """Initialize pipeline components."""
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
        
        # ROI Collector (collects ROIs during tracking, doesn't classify)
        quality_config = ROIQualityConfig(
            min_sharpness=self.tracking_config.min_sharpness,
            min_brightness=self.tracking_config.min_mean_brightness,
            max_brightness=self.tracking_config.max_mean_brightness
        )
        
        self._roi_collector = ROICollectorService(
            quality_config=quality_config,
            max_rois_per_track=10
        )

        # Classification Worker (async classification in background thread)
        self._classification_worker = ClassificationWorker(
            classifier=self._classifier,
            max_queue_size=100,
            name="ClassificationWorker"
        )
        self._classification_worker.start()

        # Tracker
        self._tracker = ConveyorTracker(config=self.tracking_config)
        
        # Bidirectional smoother
        self._smoother = BidirectionalSmoother(
            confidence_threshold=self.tracking_config.bidirectional_confidence_threshold,
            vote_ratio_threshold=self.tracking_config.evidence_ratio_threshold,
            batch_size=self.tracking_config.bidirectional_buffer_size,
            batch_timeout_seconds=self.tracking_config.bidirectional_inactivity_timeout_ms / 1000.0
        )
        
        # Recording
        if self.enable_recording:
            self._segment_writer = SegmentWriter(
                spool_dir=self.tracking_config.spool_dir,
                segment_duration=self.tracking_config.spool_segment_duration,
                max_segment_duration=self.tracking_config.spool_max_segment_duration
            )
            
            retention_config = RetentionConfig(
                max_age_hours=self.tracking_config.spool_retention_seconds / 3600.0
            )
            self._retention_policy = RetentionPolicy(
                spool_dir=self.tracking_config.spool_dir,
                config=retention_config
            )
            self._retention_policy.start()
        
        # Database
        self._db = DatabaseManager(self.app_config.db_path)

        logger.info("[ConveyorCounterApp] Components initialized")
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame through the pipeline.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Annotated frame for display
        """
        frame_start = time.perf_counter()
        
        # 1. Detection
        detect_start = time.perf_counter()
        detections = self._detector.detect(frame)
        detect_time = (time.perf_counter() - detect_start) * 1000
        
        # 2. Tracking
        track_start = time.perf_counter()
        active_tracks = self._tracker.update(
            detections,
            frame_shape=frame.shape[:2]
        )
        track_time = (time.perf_counter() - track_start) * 1000
        
        # 3. Collect ROIs for confirmed tracks (don't classify yet!)
        collect_start = time.perf_counter()
        rois_collected = 0

        for track in self._tracker.get_confirmed_tracks():
            # Only collect ROI (quality check + storage)
            # NO classification here - that happens after track completes
            if self._roi_collector.collect_roi(
                track_id=track.track_id,
                frame=frame,
                bbox=track.bbox
            ):
                rois_collected += 1

        collect_time = (time.perf_counter() - collect_start) * 1000

        # 4. Handle completed tracks
        completed_events = self._tracker.get_completed_events()
        
        for event in completed_events:
            self._handle_track_completed(event, frame)


        # Update metrics
        total_time = (time.perf_counter() - frame_start) * 1000
        self._metrics.record_detection(len(detections), detect_time)
        # Note: tracking and classification metrics are recorded elsewhere
        # when tracks complete and classifications occur

        self.state.active_tracks = len(active_tracks)
        self.state.processing_time_ms = total_time
        
        # Calculate FPS
        if self._start_time is not None:
            elapsed = time.perf_counter() - self._start_time
            self.state.fps = self._frame_count / elapsed if elapsed > 0 else 0
        
        # 6. Visualization
        if self.enable_display:
            frame = self._draw_annotations(frame, active_tracks, detections)
        
        return frame
    
    def _handle_track_completed(self, event: TrackEvent, frame: np.ndarray):
        """
        Handle a completed track - submit for async classification.

        NEW LOGIC: Don't classify here! Submit best ROI to worker thread.
        """
        track_id = event.track_id
        
        # Get best ROI from collector
        best_roi_data = self._roi_collector.get_best_roi(track_id)

        if best_roi_data is None:
            logger.warning(f"[ConveyorCounterApp] Track {track_id} completed without ROIs")
            self._roi_collector.remove_track(track_id)
            return
        
        best_roi, quality = best_roi_data

        logger.info(
            f"[ConveyorCounterApp] Track {track_id} completed, submitting ROI "
            f"(quality={quality:.1f}) for async classification"
        )

        # Submit to async classification worker (non-blocking!)
        def classification_callback(track_id: int, class_name: str, confidence: float):
            """Callback when classification completes."""
            # This runs in worker thread - keep it fast!
            logger.info(
                f"[ConveyorCounterApp] Track {track_id} classified as {class_name} "
                f"({confidence:.2f})"
            )

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
                    self._record_count(record, best_roi)
            else:
                # Record immediately
                record = ClassificationRecord(
                    track_id=track_id,
                    class_name=class_name,
                    confidence=confidence,
                    vote_ratio=1.0,
                    timestamp=time.time()
                )
                self._record_count(record, best_roi)

        # Submit job to worker (returns immediately)
        submitted = self._classification_worker.submit_job(
            track_id=track_id,
            roi=best_roi,
            bbox_history=event.bbox_history,
            callback=classification_callback
        )

        if not submitted:
            logger.error(f"[ConveyorCounterApp] Failed to submit track {track_id} (queue full)")

        # Clean up ROI collection
        self._roi_collector.remove_track(track_id)

    def _record_count(self, record: ClassificationRecord, roi: Optional[np.ndarray]):
        """Record a counted item."""
        # Update counts (thread-safe)
        new_total = self.state.increment_count(record.class_name)

        # Log to database
        if self._db is not None:
            try:
                # Save ROI image if available
                image_path = None
                if roi is not None and self.enable_recording:
                    roi_dir = os.path.join(self.tracking_config.spool_dir, "rois")
                    os.makedirs(roi_dir, exist_ok=True)
                    image_path = os.path.join(
                        roi_dir,
                        f"track_{record.track_id}_{int(time.time() * 1000)}.jpg"
                    )
                    cv2.imwrite(image_path, roi)

                import json
                self._db.log_event(
                    track_id=record.track_id,
                    bag_type=record.class_name,
                    confidence=record.confidence,
                    phash="",  # Could compute if needed
                    image_path=image_path,
                    candidates_count=1,
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

    def _draw_annotations(
        self,
        frame: np.ndarray,
        tracks: List[TrackedObject],
        detections: List[Detection]
    ) -> np.ndarray:
        """Draw tracking and detection visualizations."""
        # Draw detections (blue boxes)
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
        
        # Draw tracks (green boxes with ID)
        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            color = (0, 255, 0) if track.hits >= self.tracking_config.min_track_duration_frames else (0, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Track ID and confidence
            label = f"ID:{track.track_id} ({track.confidence:.2f})"
            cv2.putText(
                frame, label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 1
            )
            
            # Draw track history (trajectory)
            if len(track.position_history) > 1:
                pts = np.array(track.position_history, dtype=np.int32)
                cv2.polylines(frame, [pts], False, color, 1)
        
        # Draw status overlay
        status_lines = [
            f"FPS: {self.state.fps:.1f}",
            f"Tracks: {self.state.active_tracks}",
            f"Counted: {self.state.total_counted}",
        ]
        
        # Add class counts
        for cls, count in sorted(self.state.counts_by_class.items()):
            status_lines.append(f"  {cls}: {count}")
        
        y = 30
        for line in status_lines:
            cv2.putText(
                frame, line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2
            )
            y += 25
        
        return frame
    
    def run(self, max_frames: Optional[int] = None):
        """
        Run the counter application.
        
        Args:
            max_frames: Optional maximum frames to process (for testing)
        """
        logger.info("[ConveyorCounterApp] Starting...")
        
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
                
                # Process frame
                annotated = self._process_frame(frame)
                
                # Display
                if self.enable_display:
                    # Resize for faster display (720p -> 960x540)
                    display_frame = cv2.resize(annotated, (960, 540))
                    cv2.imshow("Conveyor Counter", display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
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
        """Clean up all resources."""
        logger.info("[ConveyorCounterApp] Cleaning up...")
        
        # Finalize any pending classifications
        if self._smoother is not None:
            remaining = self._smoother.finalize_batch()
            for record in remaining:
                self._record_count(record, None)
            self._smoother.cleanup()
        
        # Stop recording
        if self._segment_writer is not None:
            self._segment_writer.close()
        
        if self._retention_policy is not None:
            self._retention_policy.stop()

        # Release components
        if self._frame_source is not None:
            self._frame_source.cleanup()
        
        if self._detector is not None:
            self._detector.cleanup()
        
        # Stop classification worker and wait for pending jobs
        if self._classification_worker is not None:
            logger.info("[ConveyorCounterApp] Stopping classification worker...")
            stats = self._classification_worker.get_statistics()
            logger.info(f"  Worker stats: {stats}")
            self._classification_worker.stop(timeout=10.0)

        # Clean up ROI collector
        if self._roi_collector is not None:
            self._roi_collector.cleanup()

        # Clean up classifier
        if self._classifier is not None:
            self._classifier.cleanup()

        if self._tracker is not None:
            logger.info(f"[ConveyorCounterApp] Tracked objects: {len(self._tracker.tracks)}")

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
