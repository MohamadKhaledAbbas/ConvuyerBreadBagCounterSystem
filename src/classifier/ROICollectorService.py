"""
ROI Collector Service - Simplified for Async Classification.

NEW ARCHITECTURE:
- During tracking: ONLY collect ROIs (no classification)
- After track completes: Submit best ROI to async worker for classification

This allows main loop to run at full speed without blocking on classification.
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np

from src.classifier.IClassificationComponents import IROICollector
from src.utils.AppLogging import logger
from src.utils.Utils import compute_sharpness, compute_brightness


@dataclass
class ROIQualityConfig:
    """Configuration for ROI quality filtering."""
    min_sharpness: float = 50.0
    min_brightness: float = 30.0
    max_brightness: float = 225.0
    min_size: int = 20


@dataclass
class TrackROICollection:
    """
    ROI collection for a tracked object.

    Collects ROIs during tracking WITHOUT classifying them.
    Classification happens after track completes.
    """
    track_id: int

    # Collected ROIs
    rois: List[np.ndarray] = field(default_factory=list)
    qualities: List[float] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)

    # Best ROI tracking
    best_roi: Optional[np.ndarray] = None
    best_roi_quality: float = 0.0
    best_roi_index: int = -1

    # Statistics
    collected_count: int = 0
    rejected_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)

    # Limits
    max_rois: int = 10

    def add_roi(self, roi: np.ndarray, quality: float):
        """
        Add a quality-passed ROI to collection.

        Args:
            roi: ROI image
            quality: Quality score
        """
        # Don't collect if we have enough
        if self.collected_count >= self.max_rois:
            return

        self.rois.append(roi.copy())
        self.qualities.append(quality)
        self.timestamps.append(time.time())
        self.collected_count += 1
        self.last_updated = time.time()

        # Track best ROI
        if quality > self.best_roi_quality:
            self.best_roi = roi.copy()
            self.best_roi_quality = quality
            self.best_roi_index = self.collected_count - 1

        # Log only periodically to reduce overhead
        if self.collected_count % 5 == 0 or self.collected_count == self.max_rois:
            logger.debug(
                f"[ROICollector] Track {self.track_id}: {self.collected_count}/{self.max_rois} ROIs collected, "
                f"best_quality={self.best_roi_quality:.1f}"
            )


class ROICollectorService(IROICollector):
    """
    Service that ONLY collects ROIs during tracking.

    Does NOT classify - that happens asynchronously after track completion.

    Benefits:
    - Main loop never blocks on classification
    - Can run at full FPS
    - Classification happens in parallel on worker thread
    """

    MAX_TRACKS = 100
    STALE_TIMEOUT_SECONDS = 60.0

    def __init__(
        self,
        quality_config: Optional[ROIQualityConfig] = None,
        max_rois_per_track: int = 10
    ):
        """
        Initialize ROI collector.

        Args:
            quality_config: Quality filtering configuration
            max_rois_per_track: Maximum ROIs to collect per track
        """
        self.quality_config = quality_config or ROIQualityConfig()
        self.max_rois_per_track = max_rois_per_track

        # Track ID -> ROI collection
        self.collections: Dict[int, TrackROICollection] = {}

        # Statistics
        self._total_collected = 0
        self._total_rejected = 0
        self._last_cleanup_time = time.time()

        logger.info(
            f"[ROICollector] Initialized: max_rois_per_track={max_rois_per_track}, "
            f"quality_thresholds=(sharpness≥{self.quality_config.min_sharpness}, "
            f"brightness={self.quality_config.min_brightness}-{self.quality_config.max_brightness})"
        )

    def collect_roi(
        self,
        track_id: int,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> bool:
        """
        Collect ROI for a track (if quality passes).

        Does NOT classify - just extracts and stores good quality ROI.

        Args:
            track_id: Track identifier
            frame: Full frame image
            bbox: Bounding box (x1, y1, x2, y2)

        Returns:
            True if ROI collected, False if rejected
        """
        # Periodic cleanup
        self._maybe_cleanup()

        # Get or create collection
        if track_id not in self.collections:
            self.collections[track_id] = TrackROICollection(
                track_id=track_id,
                max_rois=self.max_rois_per_track
            )

        collection = self.collections[track_id]

        # Stop collecting if we have enough
        if collection.collected_count >= self.max_rois_per_track:
            return False

        # Extract ROI with padding
        x1, y1, x2, y2 = bbox
        pad = 5
        h, w = frame.shape[:2]

        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            return False

        # Check quality
        quality, is_valid, reason = self._compute_quality(roi)

        if not is_valid:
            collection.rejected_count += 1
            self._total_rejected += 1
            logger.debug(f"[ROICollector] Track {track_id} ROI rejected: {reason}")
            return False

        # Collect the ROI
        collection.add_roi(roi, quality)
        self._total_collected += 1

        return True

    def get_best_roi(self, track_id: int) -> Optional[Tuple[np.ndarray, float]]:
        """
        Get best quality ROI for a track.

        Called when track completes to get ROI for classification.

        Args:
            track_id: Track identifier

        Returns:
            Tuple of (best_roi, quality) or None
        """
        if track_id not in self.collections:
            return None

        collection = self.collections[track_id]

        if collection.best_roi is None:
            return None

        return collection.best_roi, collection.best_roi_quality

    def get_all_rois(self, track_id: int) -> Optional[List[Tuple[np.ndarray, float]]]:
        """
        Get all collected ROIs for a track.

        Args:
            track_id: Track identifier

        Returns:
            List of (roi, quality) tuples or None
        """
        if track_id not in self.collections:
            return None

        collection = self.collections[track_id]
        return list(zip(collection.rois, collection.qualities))

    def remove_track(self, track_id: int) -> Optional[TrackROICollection]:
        """
        Remove track collection (cleanup after classification).

        Args:
            track_id: Track identifier

        Returns:
            Removed collection or None
        """
        return self.collections.pop(track_id, None)

    def get_collection_stats(self, track_id: int) -> Optional[Dict]:
        """Get statistics for a track's collection."""
        if track_id not in self.collections:
            return None

        collection = self.collections[track_id]
        return {
            'collected': collection.collected_count,
            'rejected': collection.rejected_count,
            'best_quality': collection.best_roi_quality,
            'has_best_roi': collection.best_roi is not None
        }

    def _compute_quality(self, roi: np.ndarray) -> Tuple[float, bool, str]:
        """
        Compute ROI quality score.

        Args:
            roi: ROI image

        Returns:
            Tuple of (quality_score, is_valid, rejection_reason)
        """
        # Check size
        if min(roi.shape[:2]) < self.quality_config.min_size:
            return 0.0, False, f"too_small ({roi.shape})"

        # Compute sharpness
        sharpness = compute_sharpness(roi)
        if sharpness < self.quality_config.min_sharpness:
            return sharpness, False, f"blurry (sharpness={sharpness:.1f})"

        # Compute brightness
        brightness = compute_brightness(roi)
        if brightness < self.quality_config.min_brightness:
            return sharpness, False, f"too_dark (brightness={brightness:.1f})"
        if brightness > self.quality_config.max_brightness:
            return sharpness, False, f"too_bright (brightness={brightness:.1f})"

        # Quality score (using sharpness as primary metric)
        quality = sharpness

        return quality, True, "ok"

    def _maybe_cleanup(self):
        """Periodic cleanup of stale collections."""
        now = time.time()

        if now - self._last_cleanup_time < 30.0:
            return

        self._last_cleanup_time = now

        # Remove stale collections
        stale_ids = [
            track_id for track_id, coll in self.collections.items()
            if now - coll.last_updated > self.STALE_TIMEOUT_SECONDS
        ]

        for track_id in stale_ids:
            logger.warning(f"[ROICollector] Removing stale track {track_id}")
            del self.collections[track_id]

        # Limit total collections
        if len(self.collections) > self.MAX_TRACKS:
            sorted_collections = sorted(
                self.collections.items(),
                key=lambda x: x[1].created_at
            )

            to_remove = len(self.collections) - self.MAX_TRACKS
            for track_id, _ in sorted_collections[:to_remove]:
                logger.warning(f"[ROICollector] Removing old track {track_id} (memory limit)")
                del self.collections[track_id]

    def get_statistics(self) -> Dict:
        """Get global statistics."""
        return {
            'active_tracks': len(self.collections),
            'total_collected': self._total_collected,
            'total_rejected': self._total_rejected,
            'reject_rate': self._total_rejected / max(1, self._total_collected + self._total_rejected)
        }

    def cleanup(self):
        """Clean up resources."""
        stats = self.get_statistics()
        logger.info(
            f"[ROICollector] Cleanup - Active tracks: {stats['active_tracks']}, "
            f"Total collected: {stats['total_collected']}, "
            f"Total rejected: {stats['total_rejected']}, "
            f"Reject rate: {stats['reject_rate']:.2%}"
        )
        self.collections.clear()
        logger.info("[ROICollector] Cleanup complete")
