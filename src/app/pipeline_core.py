"""
Core pipeline for detection, tracking, and classification.

This module handles the main processing pipeline without UI or recording concerns.
Follows Single Responsibility Principle.
"""

import json
from datetime import datetime
from typing import List, Optional, Callable, Tuple

import numpy as np

from src.classifier.IClassificationComponents import IROICollector, IClassificationWorker
from src.detection.BaseDetection import BaseDetector, Detection
from src.tracking.ITracker import ITracker, TrackedObject, TrackEvent
from src.utils.AppLogging import logger


class PipelineCore:
    """
    Core pipeline: Detection → Tracking → ROI Collection → Async Classification.

    Responsibilities:
    - Run detection on frames
    - Update tracker with detections
    - Collect ROIs for confirmed tracks
    - Submit completed tracks for classification
    - Log track events for analytics

    Does NOT handle:
    - Visualization
    - Recording
    - Metrics (delegates to callbacks)
    """

    def __init__(
        self,
        detector: BaseDetector,
        tracker: ITracker,
        roi_collector: IROICollector,
        classification_worker: IClassificationWorker,
        db=None
    ):
        """
        Initialize core pipeline.

        Args:
            detector: Object detector
            tracker: Object tracker
            roi_collector: ROI collection service
            classification_worker: Async classification worker
            db: Optional DatabaseManager for track event analytics
        """
        self.detector = detector
        self.tracker = tracker
        self.roi_collector = roi_collector
        self.classification_worker = classification_worker
        self._db = db

        # Callbacks (set by orchestrator)
        # NOTE: Callback is invoked from worker thread - must be thread-safe!
        self.on_track_completed: Optional[Callable[[int, str, float, int, Optional[np.ndarray]], None]] = None

        # Optional callback for track events (for UI debugging)
        self.on_track_event: Optional[Callable[[str], None]] = None

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

        Only tracks with event_type='track_completed' (valid full travel path)
        are submitted for classification. Lost and invalid tracks are skipped.

        Uses multiple ROIs with voting for more robust classification.
        'Rejected' class votes are excluded from the voting process.

        Args:
            event: Track completion event
        """
        track_id = event.track_id

        # Notify UI of track completion
        if self.on_track_event:
            self.on_track_event(f"TRACK T{track_id} {event.event_type} ({event.exit_direction})")

        # Skip classification for invalid tracks (didn't follow full travel path)
        if event.event_type == 'track_invalid':
            logger.info(
                f"[PIPELINE] T{track_id} INVALID_TRAVEL | "
                f"exit={event.exit_direction} frames={event.total_frames} "
                f"reason=did_not_follow_bottom_to_top_path"
            )
            if self.on_track_event:
                self.on_track_event(f"SKIP T{track_id} invalid travel path")
            self._log_track_event(event)
            self.roi_collector.remove_track(track_id)
            return

        # Skip classification for lost tracks (didn't reach exit zone)
        if event.event_type == 'track_lost':
            logger.info(
                f"[PIPELINE] T{track_id} LOST | "
                f"exit={event.exit_direction} frames={event.total_frames} "
                f"reason=track_lost_before_exit"
            )
            if self.on_track_event:
                self.on_track_event(f"SKIP T{track_id} lost before exit")
            self._log_track_event(event)
            self.roi_collector.remove_track(track_id)
            return

        # Get all ROIs for voting (not just the best one)
        all_rois = self.roi_collector.get_all_rois(track_id)

        if all_rois is None or len(all_rois) == 0:
            logger.warning(
                f"[PIPELINE] T{track_id} NO_ROIS | "
                f"type={event.event_type} exit={event.exit_direction} frames={event.total_frames}"
            )
            if self.on_track_event:
                self.on_track_event(f"TRACK T{track_id} NO ROIs!")
            self.roi_collector.remove_track(track_id)
            return

        # Get top-K ROIs by quality for voting
        top_k = min(5, len(all_rois))  # Use up to 5 ROIs for voting
        sorted_rois = sorted(all_rois, key=lambda x: x[1], reverse=True)
        best_rois = [roi for roi, quality in sorted_rois[:top_k]]
        best_roi_for_saving = best_rois[0].copy()  # Store best ROI for saving by class
        avg_quality = sum(q for _, q in sorted_rois[:top_k]) / top_k

        # Get quality range
        qualities = [q for _, q in sorted_rois[:top_k]]
        min_quality = min(qualities)
        max_quality = max(qualities)

        # Notify UI of classification submission
        if self.on_track_event:
            self.on_track_event(f"SUBMIT T{track_id} for classify ({len(best_rois)} ROIs)")

        logger.info(
            f"[PIPELINE] T{track_id} SUBMIT_CLASSIFY | "
            f"total_rois={len(all_rois)} using={len(best_rois)} "
            f"quality_avg={avg_quality:.1f} quality_range=[{min_quality:.1f}-{max_quality:.1f}]"
        )

        # Log track event for analytics (classification will be updated later)
        self._log_track_event(event)

        # Submit to async worker with multiple ROIs for voting
        # Pass best_roi_for_saving via callback partial
        from functools import partial
        callback_with_roi = partial(self._classification_callback, best_roi=best_roi_for_saving)

        submitted = self.classification_worker.submit_job(
            track_id=track_id,
            roi=best_rois[0],  # Primary ROI (best quality)
            bbox_history=event.bbox_history,
            callback=callback_with_roi,
            extra_rois=best_rois[1:] if len(best_rois) > 1 else None  # Additional ROIs for voting
        )

        if not submitted:
            logger.error(f"[PipelineCore] Failed to submit track {track_id} (queue full)")

        # Clean up
        self.roi_collector.remove_track(track_id)

    def _log_track_event(self, event: TrackEvent):
        """
        Log a track lifecycle event to the database for analytics.

        Records the full journey of a tracked object including entry/exit positions,
        distance traveled, duration, and event type.

        Args:
            event: Track completion event from the tracker
        """
        if self._db is None:
            return

        try:
            # Extract entry and exit positions
            entry_x, entry_y = None, None
            exit_x, exit_y = None, None

            if event.position_history and len(event.position_history) > 0:
                entry_x, entry_y = event.position_history[0]
                exit_x, exit_y = event.position_history[-1]

            # Serialize position history as JSON
            position_json = None
            if event.position_history:
                position_json = json.dumps(
                    [[int(x), int(y)] for x, y in event.position_history]
                )

            # Compute total_hits from total_frames and age
            # total_frames = age + hits in the tracker, but we use total_frames directly
            total_hits = event.total_frames  # Approximate; includes both hit and missed frames

            self._db.add_track_event(
                track_id=event.track_id,
                event_type=event.event_type,
                timestamp=datetime.fromtimestamp(event.ended_at).isoformat(),
                created_at=datetime.fromtimestamp(event.created_at).isoformat(),
                entry_x=entry_x,
                entry_y=entry_y,
                exit_x=exit_x,
                exit_y=exit_y,
                exit_direction=event.exit_direction,
                distance_pixels=event.distance_traveled,
                duration_seconds=event.duration_seconds,
                total_frames=event.total_frames,
                avg_confidence=event.avg_confidence,
                total_hits=total_hits,
                position_history=position_json
            )
        except Exception as e:
            logger.error(f"[PipelineCore] Failed to log track event T{event.track_id}: {e}")

    def _classification_callback(self, track_id: int, class_name: str, confidence: float, non_rejected_rois: int = 0, best_roi: Optional[np.ndarray] = None):
        """
        Called when classification completes (runs in worker thread).

        Args:
            track_id: Track identifier
            class_name: Classified class
            confidence: Classification confidence
            non_rejected_rois: Number of non-rejected ROIs (for trustworthiness)
            best_roi: Best quality ROI for saving by class

        Warning:
            This method runs in the classification worker thread. The orchestrator's
            on_track_completed callback MUST be thread-safe. If the callback accesses
            shared state (e.g., counters, collections), it must use proper synchronization
            (locks, thread-safe data structures, etc.) to prevent race conditions.
        """
        logger.info(
            f"[PipelineCore] Track {track_id} classified: {class_name} ({confidence:.2f}) "
            f"non_rejected_rois={non_rejected_rois}"
        )

        # Update track event with classification result
        if self._db is not None:
            try:
                self._db.update_track_event_classification(
                    track_id=track_id,
                    classification=class_name,
                    classification_confidence=confidence
                )
            except Exception as e:
                logger.error(f"[PipelineCore] Failed to update track event classification T{track_id}: {e}")

        # Delegate to orchestrator callback
        # IMPORTANT: Callback must be thread-safe!
        if self.on_track_completed is not None:
            self.on_track_completed(track_id, class_name, confidence, non_rejected_rois, best_roi)

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
