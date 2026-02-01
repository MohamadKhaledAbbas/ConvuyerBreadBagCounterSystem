"""
Main Conveyor Bread Bag Counter Application.

This is the central orchestrator that:
1. Reads frames from source (OpenCV/ROS2)
2. Runs detection to find bread bags
3. Tracks objects through the frame
4. Classifies completed tracks
5. Applies bidirectional smoothing
6. Records video and logs events
"""

import time
import cv2
import signal
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Callable
import numpy as np

from src.config.settings import AppConfig
from src.config.tracking_config import TrackingConfig
from src.utils.AppLogging import logger, StructuredLogger
from src.utils.PipelineMetrics import PipelineMetrics, DetectionMetrics, TrackingMetrics, ClassificationMetrics

from src.frame_source.FrameSourceFactory import FrameSourceFactory
from src.frame_source.FrameSource import FrameSource

from src.detection.BaseDetection import BaseDetector, Detection
from src.detection.DetectorFactory import DetectorFactory

from src.classifier.BaseClassifier import BaseClassifier
from src.classifier.ClassifierService import ClassifierService, ROIQualityConfig
from src.classifier.ClassifierFactory import ClassifierFactory

from src.tracking.ConveyorTracker import ConveyorTracker, TrackedObject, TrackEvent
from src.tracking.BidirectionalSmoother import BidirectionalSmoother, ClassificationRecord

from src.logging.Database import DatabaseManager
from src.spool.segment_io import SegmentWriter
from src.spool.retention import RetentionPolicy, RetentionConfig


@dataclass
class CounterState:
    """Real-time state of the counter."""
    total_counted: int = 0
    counts_by_class: Dict[str, int] = None
    active_tracks: int = 0
    fps: float = 0.0
    processing_time_ms: float = 0.0
    
    def __post_init__(self):
        if self.counts_by_class is None:
            self.counts_by_class = {}


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
        self._classifier_service: Optional[ClassifierService] = None
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
                confidence_threshold=self.tracking_config.detection_confidence
            )
        
        # Classifier
        if self._classifier is None:
            self._classifier = ClassifierFactory.create(config=self.app_config)
        
        # Classifier service
        quality_config = ROIQualityConfig(
            min_sharpness=self.tracking_config.min_sharpness,
            min_brightness=self.tracking_config.min_brightness,
            max_brightness=self.tracking_config.max_brightness
        )
        
        self._classifier_service = ClassifierService(
            classifier=self._classifier,
            quality_config=quality_config,
            min_evidence_samples=self.tracking_config.min_evidence_samples,
            min_vote_ratio=self.tracking_config.min_vote_ratio,
            min_confidence=self.tracking_config.classification_confidence
        )
        
        # Tracker
        self._tracker = ConveyorTracker(config=self.tracking_config)
        
        # Bidirectional smoother
        self._smoother = BidirectionalSmoother(
            confidence_threshold=self.tracking_config.smoothing_confidence_threshold,
            vote_ratio_threshold=self.tracking_config.smoothing_vote_threshold,
            batch_size=self.tracking_config.smoothing_batch_size,
            batch_timeout_seconds=self.tracking_config.smoothing_timeout
        )
        
        # Recording
        if self.enable_recording:
            self._segment_writer = SegmentWriter(
                output_dir=self.tracking_config.spool_dir,
                segment_duration_seconds=self.tracking_config.segment_duration,
                fps=self.tracking_config.recording_fps
            )
            
            retention_config = RetentionConfig(
                max_age_hours=self.tracking_config.retention_days * 24,
                max_storage_bytes=self.tracking_config.max_storage_gb * 1024 * 1024 * 1024
            )
            self._retention_policy = RetentionPolicy(
                spool_dir=self.tracking_config.spool_dir,
                config=retention_config
            )
            self._retention_policy.start()
        
        # Database
        self._db = DatabaseManager(self.app_config.database_path)
        
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
        
        # 3. Process detections for classification
        classify_start = time.perf_counter()
        for track in self._tracker.get_confirmed_tracks():
            self._classifier_service.process_detection(
                track_id=track.track_id,
                frame=frame,
                bbox=track.bbox
            )
        classify_time = (time.perf_counter() - classify_start) * 1000
        
        # 4. Handle completed tracks
        completed_events = self._tracker.get_completed_events()
        
        for event in completed_events:
            self._handle_track_completed(event, frame)
        
        # 5. Recording
        if self.enable_recording and self._segment_writer is not None:
            self._segment_writer.write_frame(frame)
        
        # Update metrics
        total_time = (time.perf_counter() - frame_start) * 1000
        self._metrics.detection.add_sample(detect_time, len(detections))
        self._metrics.tracking.add_sample(track_time, len(active_tracks))
        self._metrics.classification.add_sample(classify_time, 0)
        
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
        """Handle a completed track - classify and count."""
        track_id = event.track_id
        
        # Get final classification
        result = self._classifier_service.get_final_classification(track_id)
        
        if result is None:
            logger.warning(f"[ConveyorCounterApp] Track {track_id} completed without classification")
            return
        
        class_name, confidence, vote_ratio, best_roi, candidates = result
        
        # Log structured event
        self._structured_logger.log_classification_result(
            track_id=track_id,
            class_name=class_name,
            confidence=confidence,
            vote_ratio=vote_ratio,
            candidates_count=len(candidates)
        )
        
        # Add to smoother
        smoothed_batch = self._smoother.add_classification(
            track_id=track_id,
            class_name=class_name,
            confidence=confidence,
            vote_ratio=vote_ratio
        )
        
        # If batch was finalized, process smoothed results
        if smoothed_batch:
            for record in smoothed_batch:
                self._record_count(record, best_roi)
        else:
            # Record immediately for now (will be retroactively smoothed)
            record = ClassificationRecord(
                track_id=track_id,
                class_name=class_name,
                confidence=confidence,
                vote_ratio=vote_ratio,
                timestamp=time.time()
            )
            self._record_count(record, best_roi)
        
        # Clean up classifier service state
        self._classifier_service.remove_track(track_id)
    
    def _record_count(self, record: ClassificationRecord, roi: Optional[np.ndarray]):
        """Record a counted item."""
        # Update counts
        self.state.total_counted += 1
        
        if record.class_name not in self.state.counts_by_class:
            self.state.counts_by_class[record.class_name] = 0
        self.state.counts_by_class[record.class_name] += 1
        
        # Log to database
        if self._db is not None:
            # Save ROI image if available
            image_path = None
            if roi is not None and self.enable_recording:
                import os
                roi_dir = os.path.join(self.tracking_config.spool_dir, "rois")
                os.makedirs(roi_dir, exist_ok=True)
                image_path = os.path.join(
                    roi_dir,
                    f"track_{record.track_id}_{int(time.time() * 1000)}.jpg"
                )
                cv2.imwrite(image_path, roi)
            
            self._db.log_event(
                track_id=record.track_id,
                bag_type=record.class_name,
                confidence=record.confidence,
                phash="",  # Could compute if needed
                image_path=image_path,
                candidates_count=1,
                metadata={
                    'vote_ratio': record.vote_ratio,
                    'smoothed': record.smoothed,
                    'original_class': record.original_class
                }
            )
        
        # Callback
        if self._on_count_callback is not None:
            self._on_count_callback(record)
        
        # Log
        smoothed_str = f" (smoothed from {record.original_class})" if record.smoothed else ""
        logger.info(
            f"[ConveyorCounterApp] COUNT: {record.class_name} "
            f"(conf={record.confidence:.2f}, track={record.track_id}){smoothed_str}"
        )
        logger.info(f"[ConveyorCounterApp] Total: {self.state.total_counted}, By class: {self.state.counts_by_class}")
    
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
            color = (0, 255, 0) if track.hits >= self.tracking_config.min_hits else (0, 255, 255)
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
                    cv2.imshow("Conveyor Counter", annotated)
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
        
        if self._retention_manager is not None:
            self._retention_manager.stop()
        
        # Release components
        if self._frame_source is not None:
            self._frame_source.cleanup()
        
        if self._detector is not None:
            self._detector.cleanup()
        
        if self._classifier_service is not None:
            self._classifier_service.cleanup()
        
        if self._tracker is not None:
            self._tracker.cleanup()
        
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
