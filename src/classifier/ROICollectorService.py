"""
ROI Collector Service - Simplified for Async Classification.

NEW ARCHITECTURE:
- During tracking: ONLY collect ROIs (no classification)
- After track completes: Submit best ROI to async worker for classification

This allows main loop to run at full speed without blocking on classification.
"""

import os
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import cv2
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
    upper_half_penalty: float = 0.5  # Quality multiplier for ROIs above half screen (Y axis)

    # Diversity/spacing controls
    min_frame_spacing: int = 3  # Minimum frames between ROI collection
    min_position_change: float = 20.0  # Minimum pixel movement (centroid) between ROIs

    # Enhanced position penalty (gradual from center to top)
    enable_gradual_position_penalty: bool = True  # Use gradual penalty vs binary
    position_penalty_start_ratio: float = 0.5  # Start applying penalty at this Y ratio (0.5 = center)
    position_penalty_max_ratio: float = 0.15  # Max penalty at this Y ratio (0.15 = top 15%)
    position_penalty_min_multiplier: float = 0.3  # Minimum quality multiplier at top


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
    frame_indices: List[int] = field(default_factory=list)  # Track frame index for spacing
    positions: List[Tuple[float, float]] = field(default_factory=list)  # Track bbox centroids

    # Best ROI tracking
    best_roi: Optional[np.ndarray] = None
    best_roi_quality: float = 0.0
    best_roi_index: int = -1

    # Statistics
    collected_count: int = 0
    rejected_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)

    # Diversity tracking
    last_frame_index: int = -1  # Last frame where ROI was collected
    last_position: Optional[Tuple[float, float]] = None  # Last bbox centroid

    # Limits
    max_rois: int = 10

    # Temporal weighting
    enable_temporal_weighting: bool = False
    temporal_decay_rate: float = 0.15

    def add_roi(self, roi: np.ndarray, quality: float, frame_index: int = 0,
                position: Optional[Tuple[float, float]] = None):
        """
        Add a quality-passed ROI to collection.

        Supports temporal weighting: earlier ROIs (closer to camera) get higher scores.

        Args:
            roi: ROI image
            quality: Quality score (raw, before temporal weighting)
            frame_index: Frame number for spacing tracking
            position: Bbox centroid (x, y) for position diversity
        """
        # Don't collect if we have enough
        if self.collected_count >= self.max_rois:
            return

        # Apply temporal weighting if enabled
        # Earlier ROIs are closer to camera = better quality
        weighted_quality = quality
        if self.enable_temporal_weighting and self.collected_count > 0:
            # Decay quality based on position in collection
            # First ROI (index 0) = 100% quality
            # Last ROI (index max_rois-1) = (100 - decay_rate*100)% quality
            decay_factor = 1.0 - (self.collected_count / max(self.max_rois, 1)) * self.temporal_decay_rate
            weighted_quality = quality * decay_factor

            logger.debug(
                f"[ROICollector] T{self.track_id}: Temporal weighting applied - "
                f"raw_quality={quality:.1f}, weighted={weighted_quality:.1f}, "
                f"decay_factor={decay_factor:.3f}"
            )

        self.rois.append(roi.copy())
        self.qualities.append(weighted_quality)  # Store weighted quality
        self.timestamps.append(time.time())
        self.frame_indices.append(frame_index)
        if position:
            self.positions.append(position)

        self.collected_count += 1
        self.last_updated = time.time()
        self.last_frame_index = frame_index
        self.last_position = position

        # Track best ROI (using weighted quality)
        if weighted_quality > self.best_roi_quality:
            self.best_roi = roi.copy()
            self.best_roi_quality = weighted_quality
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
        max_rois_per_track: int = 10,
        save_roi_candidates: bool = False,
        save_all_rois: bool = False,
        roi_candidates_dir: str = "data/roi_candidates",
        enable_temporal_weighting: bool = True,
        temporal_decay_rate: float = 0.15
    ):
        """
        Initialize ROI collector.

        Args:
            quality_config: Quality filtering configuration
            max_rois_per_track: Maximum ROIs to collect per track
            save_roi_candidates: Save ROIs that pass quality checks
            save_all_rois: Save all ROIs (including rejected ones)
            roi_candidates_dir: Directory to save ROI images
            enable_temporal_weighting: Enable temporal decay (earlier ROIs = better)
            temporal_decay_rate: Decay rate (0-1), 0.15 = 15% reduction from first to last
        """
        self.quality_config = quality_config or ROIQualityConfig()
        self.max_rois_per_track = max_rois_per_track

        # ROI saving configuration
        self.save_roi_candidates = save_roi_candidates
        self.save_all_rois = save_all_rois
        self.roi_candidates_dir = roi_candidates_dir

        # Temporal weighting configuration
        self.enable_temporal_weighting = enable_temporal_weighting
        self.temporal_decay_rate = temporal_decay_rate

        # Create ROI save directory if saving is enabled
        if self.save_roi_candidates or self.save_all_rois:
            os.makedirs(self.roi_candidates_dir, exist_ok=True)
            logger.info(
                f"[ROICollector] ROI saving enabled: "
                f"save_candidates={save_roi_candidates}, save_all={save_all_rois}, "
                f"dir={roi_candidates_dir}"
            )

        # Track ID -> ROI collection
        self.collections: Dict[int, TrackROICollection] = {}

        # Statistics
        self._total_collected = 0
        self._total_rejected = 0
        self._last_cleanup_time = time.time()

        # Frame counter for diversity tracking
        self._frame_counter = 0

        logger.info(
            f"[ROICollector] Initialized: max_rois_per_track={max_rois_per_track}, "
            f"quality_thresholds=(sharpness≥{self.quality_config.min_sharpness}, "
            f"brightness={self.quality_config.min_brightness}-{self.quality_config.max_brightness}), "
            f"temporal_weighting={'enabled' if enable_temporal_weighting else 'disabled'}"
            f"{f', decay_rate={temporal_decay_rate:.2f}' if enable_temporal_weighting else ''}"
        )

    def collect_roi(
        self,
        track_id: int,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> bool:
        """
        Collect ROI for a track (if quality passes).

        Enhanced with:
        - Frame spacing enforcement (avoid consecutive frames)
        - Position diversity enforcement (require movement)
        - Gradual position penalty (smoother than binary)

        Does NOT classify - just extracts and stores good quality ROI.

        Args:
            track_id: Track identifier
            frame: Full frame image
            bbox: Bounding box (x1, y1, x2, y2)

        Returns:
            True if ROI collected, False if rejected
        """
        # Increment frame counter
        self._frame_counter += 1

        # Periodic cleanup
        self._maybe_cleanup()

        # Get or create collection
        if track_id not in self.collections:
            self.collections[track_id] = TrackROICollection(
                track_id=track_id,
                max_rois=self.max_rois_per_track,
                enable_temporal_weighting=self.enable_temporal_weighting,
                temporal_decay_rate=self.temporal_decay_rate
            )
            logger.info(f"[ROI_LIFECYCLE] T{track_id} ROI_COLLECTION_START | max_rois={self.max_rois_per_track}")

        collection = self.collections[track_id]

        # Stop collecting if we have enough
        if collection.collected_count >= self.max_rois_per_track:
            return False

        # Calculate bbox centroid for position diversity
        x1, y1, x2, y2 = bbox
        centroid_x = (x1 + x2) / 2
        centroid_y = (y1 + y2) / 2
        current_position = (centroid_x, centroid_y)

        # Check frame spacing - enforce minimum gap between collections
        if collection.last_frame_index >= 0:
            frame_gap = self._frame_counter - collection.last_frame_index
            if frame_gap < self.quality_config.min_frame_spacing:
                logger.debug(
                    f"[ROI_LIFECYCLE] T{track_id} ROI_SKIPPED_FRAME_SPACING | "
                    f"frame_gap={frame_gap} < min={self.quality_config.min_frame_spacing}"
                )
                return False

        # Check position diversity - require significant movement
        if collection.last_position is not None:
            last_x, last_y = collection.last_position
            position_change = np.sqrt((centroid_x - last_x)**2 + (centroid_y - last_y)**2)

            if position_change < self.quality_config.min_position_change:
                logger.debug(
                    f"[ROI_LIFECYCLE] T{track_id} ROI_SKIPPED_POSITION | "
                    f"change={position_change:.1f}px < min={self.quality_config.min_position_change}px"
                )
                return False

        # Extract ROI with padding
        pad = 5
        h, w = frame.shape[:2]

        x1_crop = max(0, x1 - pad)
        y1_crop = max(0, y1 - pad)
        x2_crop = min(w, x2 + pad)
        y2_crop = min(h, y2 + pad)

        roi = frame[y1_crop:y2_crop, x1_crop:x2_crop]

        if roi.size == 0:
            return False

        # Check quality
        quality, is_valid, reason = self._compute_quality(roi)

        # Save ROI if configured
        if self.save_all_rois:
            # Save all ROIs (both accepted and rejected)
            self._save_roi_image(roi, track_id, quality, is_valid, reason)
        elif self.save_roi_candidates and is_valid:
            # Save only accepted ROIs
            self._save_roi_image(roi, track_id, quality, is_valid, reason)

        if not is_valid:
            collection.rejected_count += 1
            self._total_rejected += 1
            logger.debug(
                f"[ROI_LIFECYCLE] T{track_id} ROI_REJECTED | "
                f"reason={reason} total_collected={collection.collected_count} "
                f"total_rejected={collection.rejected_count}"
            )
            return False

        # Apply position penalty based on Y position
        # Lower in frame (closer to camera) = better quality
        # Upper in frame (farther from camera) = penalty
        bbox_center_y = centroid_y

        if self.quality_config.enable_gradual_position_penalty:
            # Gradual penalty from center to top
            # y_ratio: 0.0 = top of frame, 1.0 = bottom of frame
            y_ratio = bbox_center_y / h

            penalty_start = self.quality_config.position_penalty_start_ratio
            penalty_max = self.quality_config.position_penalty_max_ratio
            min_multiplier = self.quality_config.position_penalty_min_multiplier

            if y_ratio < penalty_start:
                # Calculate gradual penalty
                # At penalty_start (e.g., 0.5 center): no penalty (1.0x)
                # At penalty_max (e.g., 0.15 top): max penalty (e.g., 0.3x)
                if y_ratio <= penalty_max:
                    # At or above max penalty zone
                    penalty_multiplier = min_multiplier
                else:
                    # Gradual penalty between penalty_max and penalty_start
                    # Linear interpolation
                    penalty_range = penalty_start - penalty_max
                    position_in_range = (y_ratio - penalty_max) / penalty_range
                    penalty_multiplier = min_multiplier + (1.0 - min_multiplier) * position_in_range

                quality *= penalty_multiplier
                logger.debug(
                    f"[ROI_LIFECYCLE] T{track_id} GRADUAL_POSITION_PENALTY | "
                    f"y_ratio={y_ratio:.2f} penalty_mult={penalty_multiplier:.2f} quality={quality:.1f}"
                )
        else:
            # Binary penalty (original behavior)
            if bbox_center_y < h / 2:
                quality *= self.quality_config.upper_half_penalty
                logger.debug(
                    f"[ROI_LIFECYCLE] T{track_id} UPPER_HALF_PENALTY | "
                    f"bbox_center_y={bbox_center_y:.0f} frame_h={h} "
                    f"penalty={self.quality_config.upper_half_penalty} quality={quality:.1f}"
                )

        # Collect the ROI with frame index and position
        collection.add_roi(roi, quality, self._frame_counter, current_position)
        self._total_collected += 1

        logger.info(
            f"[ROI_LIFECYCLE] T{track_id} ROI_COLLECTED | "
            f"quality={quality:.1f} count={collection.collected_count}/{self.max_rois_per_track} "
            f"best_quality={collection.best_roi_quality:.1f} size={roi.shape[1]}x{roi.shape[0]} "
            f"y_pos={centroid_y:.0f}"
        )

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

    def _save_roi_image(
        self,
        roi: np.ndarray,
        track_id: int,
        quality: float,
        is_valid: bool,
        reason: str = ""
    ):
        """
        Save ROI image to disk for debugging/analysis.

        Args:
            roi: ROI image
            track_id: Track identifier
            quality: Quality score
            is_valid: Whether ROI passed quality checks
            reason: Rejection reason if not valid
        """
        try:
            timestamp = int(time.time() * 1000)
            status = "accepted" if is_valid else "rejected"

            # Create filename with metadata
            filename = f"track_{track_id}_{timestamp}_{status}_q{quality:.1f}"
            if not is_valid and reason:
                # Sanitize reason for filename
                reason_safe = reason.replace(" ", "_").replace("(", "").replace(")", "")[:30]
                filename += f"_{reason_safe}"
            filename += ".jpg"

            filepath = os.path.join(self.roi_candidates_dir, filename)
            cv2.imwrite(filepath, roi)

            logger.debug(f"[ROICollector] Saved ROI: {filename}")
        except Exception as e:
            logger.error(f"[ROICollector] Failed to save ROI: {e}")

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
