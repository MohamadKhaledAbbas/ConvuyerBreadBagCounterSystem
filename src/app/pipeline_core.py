"""
Core pipeline for detection, tracking, and classification.

This module handles the main processing pipeline without UI or recording concerns.
Follows Single Responsibility Principle.
"""

from typing import List, Optional, Callable, Tuple
import numpy as np

from src.detection.BaseDetection import BaseDetector, Detection
from src.tracking.ITracker import ITracker, TrackedObject, TrackEvent
from src.classifier.IClassificationComponents import IROICollector, IClassificationWorker
from src.utils.AppLogging import logger


class PipelineCore:
    """
    Core pipeline: Detection → Tracking → ROI Collection → Async Classification.

    Responsibilities:
    - Run detection on frames
    - Update tracker with detections
    - Collect ROIs for confirmed tracks
    - Submit completed tracks for classification

    Does NOT handle:
    - Visualization
    - Recording
    - Database
    - Metrics (delegates to callbacks)
    """

    def __init__(
        self,
        detector: BaseDetector,
        tracker: ITracker,
        roi_collector: IROICollector,
        classification_worker: IClassificationWorker
    ):
        """
        Initialize core pipeline.

        Args:
            detector: Object detector
            tracker: Object tracker
            roi_collector: ROI collection service
            classification_worker: Async classification worker
        """
        self.detector = detector
        self.tracker = tracker
        self.roi_collector = roi_collector
        self.classification_worker = classification_worker

        # Callbacks (set by orchestrator)
        # NOTE: Callback is invoked from worker thread - must be thread-safe!
        self.on_track_completed: Optional[Callable[[int, str, float], None]] = None

        logger.info("[PipelineCore] Initialized")

    def process_frame(
        self,
        frame: np.ndarray
    ) -> Tuple[List[Detection], List[TrackedObject], int]:
        """
        Process a single frame through the pipeline.

        Args:
            frame: Input BGR frame

        Returns:
            Tuple of (detections, active_tracks, rois_collected)
        """
        # 1. Detection
        detections = self.detector.detect(frame)

        # 2. Tracking - cast frame shape to proper type
        frame_h, frame_w = frame.shape[:2]
        active_tracks = self.tracker.update(
            detections,
            frame_shape=(int(frame_h), int(frame_w))
        )

        # 3. Collect ROIs for confirmed tracks (non-blocking)
        rois_collected = 0
        for track in self.tracker.get_confirmed_tracks():
            if self.roi_collector.collect_roi(
                track_id=track.track_id,
                frame=frame,
                bbox=track.bbox
            ):
                rois_collected += 1

        # 4. Handle completed tracks
        completed_events = self.tracker.get_completed_events()
        for event in completed_events:
            self._handle_track_completed(event)

        return detections, active_tracks, rois_collected

    def _handle_track_completed(self, event: TrackEvent):
        """
        Handle a completed track - submit for async classification.

        Args:
            event: Track completion event
        """
        track_id = event.track_id

        # Get best ROI
        best_roi_data = self.roi_collector.get_best_roi(track_id)

        if best_roi_data is None:
            logger.warning(f"[PipelineCore] Track {track_id} completed without ROIs")
            self.roi_collector.remove_track(track_id)
            return

        best_roi, quality = best_roi_data

        logger.info(
            f"[PipelineCore] Track {track_id} completed, submitting for classification "
            f"(quality={quality:.1f})"
        )

        # Submit to async worker (non-blocking)
        submitted = self.classification_worker.submit_job(
            track_id=track_id,
            roi=best_roi,
            bbox_history=event.bbox_history,
            callback=self._classification_callback
        )

        if not submitted:
            logger.error(f"[PipelineCore] Failed to submit track {track_id} (queue full)")

        # Clean up
        self.roi_collector.remove_track(track_id)

    def _classification_callback(self, track_id: int, class_name: str, confidence: float):
        """
        Called when classification completes (runs in worker thread).

        Args:
            track_id: Track identifier
            class_name: Classified class
            confidence: Classification confidence

        Warning:
            This method runs in the classification worker thread. The orchestrator's
            on_track_completed callback MUST be thread-safe. If the callback accesses
            shared state (e.g., counters, collections), it must use proper synchronization
            (locks, thread-safe data structures, etc.) to prevent race conditions.
        """
        logger.info(
            f"[PipelineCore] Track {track_id} classified: {class_name} ({confidence:.2f})"
        )

        # Delegate to orchestrator callback
        # IMPORTANT: Callback must be thread-safe!
        if self.on_track_completed is not None:
            self.on_track_completed(track_id, class_name, confidence)

    def cleanup(self):
        """Release pipeline resources."""
        logger.info("[PipelineCore] Cleaning up...")

        if self.classification_worker:
            self.classification_worker.stop(timeout=10.0)

        if self.roi_collector:
            self.roi_collector.cleanup()

        if self.tracker:
            self.tracker.cleanup()

        if self.detector:
            self.detector.cleanup()

        logger.info("[PipelineCore] Cleanup complete")
