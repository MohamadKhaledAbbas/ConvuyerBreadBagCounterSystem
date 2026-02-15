"""
Core pipeline for detection, tracking, and classification.

This module handles the main processing pipeline without UI or recording concerns.
Follows Single Responsibility Principle.
"""

import glob
import json
import os
import threading
import time
from datetime import datetime
from typing import List, Optional, Callable, Tuple

import cv2
import numpy as np

from src.classifier.IClassificationComponents import IROICollector, IClassificationWorker
from src.config.tracking_config import TrackingConfig
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
        db=None,
        tracking_config: Optional[TrackingConfig] = None
    ):
        """
        Initialize core pipeline.

        Args:
            detector: Object detector
            tracker: Object tracker
            roi_collector: ROI collection service
            classification_worker: Async classification worker
            db: Optional DatabaseManager for track event analytics
            tracking_config: Optional tracking configuration for ROI saving options
        """
        self.detector = detector
        self.tracker = tracker
        self.roi_collector = roi_collector
        self.classification_worker = classification_worker
        self._db = db
        self._tracking_config = tracking_config or TrackingConfig()

        # Callbacks (set by orchestrator)
        # NOTE: Callback is invoked from worker thread - must be thread-safe!
        self.on_track_completed: Optional[Callable[[int, str, float, int, Optional[np.ndarray]], None]] = None

        # Optional callback for track events (for UI debugging)
        self.on_track_event: Optional[Callable[[str], None]] = None

        # Ensure classified ROIs directory exists if saving is enabled
        if self._tracking_config.save_classified_rois:
            os.makedirs(self._tracking_config.classified_rois_dir, exist_ok=True)
            logger.info(f"[PipelineCore] Save classified ROIs enabled: {self._tracking_config.classified_rois_dir}")

        # Background purge mechanism for classified ROIs
        self._purge_thread: Optional[threading.Thread] = None
        self._purge_stop_event = threading.Event()
        self._last_purge_time: float = 0.0

        # Start purge thread if saving is enabled and retention is configured
        if self._tracking_config.save_classified_rois:
            self._start_purge_thread()

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
            collected = self.roi_collector.collect_roi(
                track_id=track.track_id,
                frame=frame,
                bbox=track.bbox
            )
            if collected:
                rois_collected += 1
                # Log ROI collection detail to DB
                self._log_roi_collected(track.track_id, track.bbox)

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

        # Save classified ROIs if enabled (before classification)
        if self._tracking_config.save_classified_rois:
            self._save_classified_rois(track_id, sorted_rois[:top_k])

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

            self._db.enqueue_write(
                """INSERT INTO track_events (
                    track_id, event_type, timestamp, created_at,
                    entry_x, entry_y, exit_x, exit_y,
                    exit_direction, distance_pixels, duration_seconds, total_frames,
                    avg_confidence, total_hits,
                    classification, classification_confidence,
                    position_history
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    event.track_id, event.event_type,
                    datetime.fromtimestamp(event.ended_at).isoformat(),
                    datetime.fromtimestamp(event.created_at).isoformat(),
                    entry_x, entry_y, exit_x, exit_y,
                    event.exit_direction, event.distance_traveled,
                    event.duration_seconds, event.total_frames,
                    event.avg_confidence, total_hits,
                    None, None,
                    position_json
                )
            )
        except Exception as e:
            logger.error(f"[PipelineCore] Failed to log track event T{event.track_id}: {e}")

        # Also log a detail row for the event type itself (non-blocking)
        try:
            self._db.enqueue_track_event_detail(
                track_id=event.track_id,
                timestamp=datetime.fromtimestamp(event.ended_at).isoformat(),
                step_type=event.event_type,
                detail=json.dumps({
                    'exit_direction': event.exit_direction,
                    'total_frames': event.total_frames,
                    'duration_seconds': round(event.duration_seconds, 2) if event.duration_seconds else None,
                    'distance_pixels': round(event.distance_traveled, 1) if event.distance_traveled else None
                })
            )
        except Exception as e:
            logger.error(f"[PipelineCore] Failed to log track event detail T{event.track_id}: {e}")

    def _log_roi_collected(self, track_id: int, bbox: tuple):
        """Log an ROI collection event with bounding box coordinates (non-blocking)."""
        if self._db is None:
            return
        try:
            x1, y1, x2, y2 = bbox
            # Get collection stats for roi_index
            stats = self.roi_collector.get_collection_stats(track_id)
            roi_index = stats['collected'] - 1 if stats else 0
            quality = stats['best_quality'] if stats else 0.0

            self._db.enqueue_track_event_detail(
                track_id=track_id,
                timestamp=datetime.now().isoformat(),
                step_type='roi_collected',
                bbox_x1=int(x1), bbox_y1=int(y1),
                bbox_x2=int(x2), bbox_y2=int(y2),
                quality_score=quality,
                roi_index=roi_index
            )
        except Exception as e:
            logger.error(f"[PipelineCore] Failed to log ROI collection T{track_id}: {e}")

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

        # Update track event with classification result (non-blocking)
        if self._db is not None:
            try:
                self._db.enqueue_write(
                    """UPDATE track_events
                       SET classification = ?, classification_confidence = ?
                       WHERE id = (
                           SELECT id FROM track_events
                           WHERE track_id = ? AND classification IS NULL
                           ORDER BY id DESC LIMIT 1
                       )""",
                    (class_name, confidence, track_id)
                )
                # Log voting result detail (non-blocking)
                self._db.enqueue_track_event_detail(
                    track_id=track_id,
                    timestamp=datetime.now().isoformat(),
                    step_type='voting_result',
                    class_name=class_name,
                    confidence=confidence,
                    valid_votes=non_rejected_rois,
                    detail=json.dumps({
                        'final_class': class_name,
                        'final_confidence': round(confidence, 4),
                        'non_rejected_rois': non_rejected_rois
                    })
                )
            except Exception as e:
                logger.error(f"[PipelineCore] Failed to update track event classification T{track_id}: {e}")

        # Delegate to orchestrator callback
        # IMPORTANT: Callback must be thread-safe!
        if self.on_track_completed is not None:
            self.on_track_completed(track_id, class_name, confidence, non_rejected_rois, best_roi)

    def _save_classified_rois(self, track_id: int, rois_with_quality: List[Tuple[np.ndarray, float]]):
        """
        Save ROIs that are used for classification (voting).

        This saves only the ROIs actually sent to the classifier, which is narrower than:
        - save_all_rois: saves all collected ROIs
        - save_roi_candidates: saves accepted ROI candidates

        Directory structure:
            classified_rois_dir/
                track_XXXX_roi_0_quality_YYY.jpg  (best quality)
                track_XXXX_roi_1_quality_YYY.jpg  (second best)
                ...

        Args:
            track_id: Track identifier
            rois_with_quality: List of (roi, quality_score) tuples, sorted by quality descending
        """
        try:
            timestamp = int(time.time() * 1000)
            saved_count = 0

            for idx, (roi, quality) in enumerate(rois_with_quality):
                if roi is None or roi.size == 0:
                    continue

                # Filename includes track_id, roi index (by quality rank), quality score, timestamp
                filename = f"track_{track_id}_{timestamp}_roi_{idx}_quality_{quality:.1f}.jpg"
                filepath = os.path.join(self._tracking_config.classified_rois_dir, filename)

                cv2.imwrite(filepath, roi)
                saved_count += 1

            logger.debug(
                f"[PipelineCore] Saved {saved_count} classified ROIs for T{track_id} "
                f"to {self._tracking_config.classified_rois_dir}"
            )

        except Exception as e:
            logger.error(f"[PipelineCore] Failed to save classified ROIs for T{track_id}: {e}")

    def _start_purge_thread(self):
        """Start background thread for purging old classified ROIs."""
        if self._purge_thread is not None and self._purge_thread.is_alive():
            return

        self._purge_stop_event.clear()
        self._purge_thread = threading.Thread(
            target=self._purge_loop,
            name="ClassifiedROIPurger",
            daemon=True
        )
        self._purge_thread.start()
        logger.info(
            f"[PipelineCore] Started classified ROI purge thread "
            f"(retention={self._tracking_config.classified_rois_retention_hours}h, "
            f"max_count={self._tracking_config.classified_rois_max_count}, "
            f"interval={self._tracking_config.classified_rois_purge_interval_minutes}min)"
        )

    def _purge_loop(self):
        """Background loop for periodic purge of old classified ROIs."""
        purge_interval_seconds = self._tracking_config.classified_rois_purge_interval_minutes * 60

        while not self._purge_stop_event.is_set():
            try:
                # Wait for purge interval or stop event
                if self._purge_stop_event.wait(timeout=purge_interval_seconds):
                    break  # Stop event was set

                # Perform purge
                self._purge_classified_rois()

            except Exception as e:
                logger.error(f"[PipelineCore] Error in purge loop: {e}", exc_info=True)

        logger.info("[PipelineCore] Classified ROI purge thread stopped")

    def _purge_classified_rois(self):
        """
        Purge old classified ROIs based on retention policy.

        Two-phase purge:
        1. Time-based: Delete files older than retention_hours
        2. Count-based: If still over max_count, delete oldest files first
        """
        try:
            roi_dir = self._tracking_config.classified_rois_dir
            if not os.path.exists(roi_dir):
                return

            # Get all ROI files with their modification times
            pattern = os.path.join(roi_dir, "track_*.jpg")
            files = glob.glob(pattern)

            if not files:
                return

            # Get file info: (filepath, mtime)
            file_info = []
            for filepath in files:
                try:
                    mtime = os.path.getmtime(filepath)
                    file_info.append((filepath, mtime))
                except OSError:
                    continue

            if not file_info:
                return

            initial_count = len(file_info)
            deleted_time_based = 0
            deleted_count_based = 0

            # Phase 1: Time-based deletion
            retention_hours = self._tracking_config.classified_rois_retention_hours
            if retention_hours > 0:
                retention_seconds = retention_hours * 3600
                cutoff_time = time.time() - retention_seconds

                # Filter out files to delete (older than cutoff)
                files_to_keep = []
                for filepath, mtime in file_info:
                    if mtime < cutoff_time:
                        try:
                            os.remove(filepath)
                            deleted_time_based += 1
                        except OSError as e:
                            logger.warning(f"[PipelineCore] Failed to delete old ROI {filepath}: {e}")
                    else:
                        files_to_keep.append((filepath, mtime))

                file_info = files_to_keep

            # Phase 2: Count-based deletion (delete oldest first)
            max_count = self._tracking_config.classified_rois_max_count
            if max_count > 0 and len(file_info) > max_count:
                # Sort by mtime ascending (oldest first)
                file_info.sort(key=lambda x: x[1])

                # Delete oldest files to get under max_count
                files_to_delete = len(file_info) - max_count
                for filepath, _ in file_info[:files_to_delete]:
                    try:
                        os.remove(filepath)
                        deleted_count_based += 1
                    except OSError as e:
                        logger.warning(f"[PipelineCore] Failed to delete excess ROI {filepath}: {e}")

            total_deleted = deleted_time_based + deleted_count_based
            remaining = initial_count - total_deleted

            if total_deleted > 0:
                logger.info(
                    f"[PipelineCore] Purged classified ROIs: "
                    f"time_based={deleted_time_based}, count_based={deleted_count_based}, "
                    f"remaining={remaining}"
                )
            else:
                logger.debug(
                    f"[PipelineCore] Classified ROI purge check: no files to purge "
                    f"(total={initial_count})"
                )

        except Exception as e:
            logger.error(f"[PipelineCore] Failed to purge classified ROIs: {e}", exc_info=True)

    def _stop_purge_thread(self, timeout: float = 5.0):
        """Stop the background purge thread."""
        if self._purge_thread is None:
            return

        self._purge_stop_event.set()

        if self._purge_thread.is_alive():
            self._purge_thread.join(timeout=timeout)

        self._purge_thread = None
        logger.info("[PipelineCore] Stopped classified ROI purge thread")

    def cleanup(self):
        """Release pipeline resources."""
        logger.info("[PipelineCore] Cleaning up...")

        # Stop purge thread first
        self._stop_purge_thread()

        if self.classification_worker:
            self.classification_worker.stop(timeout=10.0)

        if self.roi_collector:
            self.roi_collector.cleanup()

        if self.tracker:
            self.tracker.cleanup()

        if self.detector:
            self.detector.cleanup()

        logger.info("[PipelineCore] Cleanup complete")
