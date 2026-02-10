"""
Enhanced ROI Collector Service with multi-factor quality scoring.

Improvements over basic version:
- Multi-factor quality score (sharpness + size + brightness)
- Aspect ratio validation for bread bags
- Better thresholds for conveyor systems
- Configurable quality weights
"""

import os
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np

from src.classifier.IClassificationComponents import IROICollector
from src.classifier.ROICollectorService import TrackROICollection
from src.utils.AppLogging import logger
from src.utils.Utils import compute_sharpness, compute_brightness


@dataclass
class EnhancedROIQualityConfig:
    """
    Enhanced configuration for ROI quality filtering.

    Optimized for bread bag classification on conveyor systems.
    """
    # Sharpness (most important - detects blur from motion)
    min_sharpness: float = 100.0
    optimal_sharpness: float = 1000.0
    sharpness_weight: float = 0.6

    # Brightness (important for color and detail)
    min_brightness: float = 50.0
    max_brightness: float = 200.0
    optimal_brightness: float = 120.0
    brightness_weight: float = 0.2

    # Size (larger ROIs have more detail)
    min_size: int = 50
    optimal_area: float = 100000.0  # ~316x316 pixels
    size_weight: float = 0.2

    # Aspect ratio (bread bags are vertical rectangles)
    min_aspect_ratio: float = 0.4   # width/height
    max_aspect_ratio: float = 2.5
    optimal_aspect_ratio: float = 0.67  # 2:3 ratio
    check_aspect_ratio: bool = True


class EnhancedROICollectorService(IROICollector):
    """
    Enhanced ROI collector with multi-factor quality scoring.

    Key improvements:
    1. Multi-factor quality score (not just sharpness)
    2. Aspect ratio validation
    3. Better scoring for classification-ready ROIs
    4. Configurable quality weights
    """

    MAX_TRACKS = 100
    STALE_TIMEOUT_SECONDS = 60.0

    def __init__(
        self,
        quality_config: Optional[EnhancedROIQualityConfig] = None,
        max_rois_per_track: int = 10,
        save_roi_candidates: bool = False,
        save_all_rois: bool = False,
        roi_candidates_dir: str = "data/roi_candidates"
    ):
        """Initialize enhanced ROI collector."""
        self.quality_config = quality_config or EnhancedROIQualityConfig()
        self.max_rois_per_track = max_rois_per_track

        # ROI saving
        self.save_roi_candidates = save_roi_candidates
        self.save_all_rois = save_all_rois
        self.roi_candidates_dir = roi_candidates_dir

        if self.save_roi_candidates or self.save_all_rois:
            os.makedirs(self.roi_candidates_dir, exist_ok=True)
            logger.info(
                f"[EnhancedROICollector] ROI saving enabled: "
                f"candidates={save_roi_candidates}, all={save_all_rois}"
            )

        # Collections
        self.collections: Dict[int, 'TrackROICollection'] = {}

        # Statistics
        self._total_collected = 0
        self._total_rejected = 0
        self._rejection_reasons: Dict[str, int] = {}
        self._last_cleanup_time = time.time()

        logger.info(
            f"[EnhancedROICollector] Initialized with multi-factor scoring:\n"
            f"  Sharpness: ≥{self.quality_config.min_sharpness:.0f} (weight={self.quality_config.sharpness_weight})\n"
            f"  Brightness: {self.quality_config.min_brightness:.0f}-{self.quality_config.max_brightness:.0f} (weight={self.quality_config.brightness_weight})\n"
            f"  Size: ≥{self.quality_config.min_size}px (weight={self.quality_config.size_weight})\n"
            f"  Aspect ratio: {self.quality_config.min_aspect_ratio:.2f}-{self.quality_config.max_aspect_ratio:.2f}"
        )

    def _compute_quality(self, roi: np.ndarray) -> Tuple[float, bool, str]:
        """
        Compute multi-factor ROI quality score.

        Combines:
        - Sharpness (60%): Most important - detects blur
        - Size (20%): Larger ROIs have more detail
        - Brightness (20%): Mid-range is best

        Returns:
            (quality_score, is_valid, reason)
        """
        height, width = roi.shape[:2]

        # 1. Size check
        if min(height, width) < self.quality_config.min_size:
            return 0.0, False, f"too_small_{width}x{height}"

        # 2. Aspect ratio check
        if self.quality_config.check_aspect_ratio:
            aspect_ratio = width / max(height, 1)
            if not (self.quality_config.min_aspect_ratio <= aspect_ratio <= self.quality_config.max_aspect_ratio):
                return 0.0, False, f"bad_aspect_{aspect_ratio:.2f}"

        # 3. Compute sharpness
        sharpness = compute_sharpness(roi)
        if sharpness < self.quality_config.min_sharpness:
            return sharpness, False, f"blurry_sharp{sharpness:.0f}"

        # 4. Compute brightness
        brightness = compute_brightness(roi)
        if brightness < self.quality_config.min_brightness:
            return sharpness, False, f"too_dark_bright{brightness:.0f}"
        if brightness > self.quality_config.max_brightness:
            return sharpness, False, f"too_bright_bright{brightness:.0f}"

        # === MULTI-FACTOR QUALITY SCORE ===

        # Sharpness factor (0-1)
        sharpness_factor = min(sharpness / self.quality_config.optimal_sharpness, 1.0)

        # Size factor (0-1) - prefer larger ROIs
        area = float(width * height)
        size_factor = min(area / self.quality_config.optimal_area, 1.0)

        # Brightness factor (0-1) - prefer mid-range
        brightness_deviation = abs(brightness - self.quality_config.optimal_brightness)
        brightness_factor = 1.0 - min(brightness_deviation / self.quality_config.optimal_brightness, 1.0)

        # Combined weighted score (0-100)
        quality = (
            sharpness_factor * self.quality_config.sharpness_weight +
            size_factor * self.quality_config.size_weight +
            brightness_factor * self.quality_config.brightness_weight
        ) * 100.0

        # Store component scores in reason for debugging
        reason = f"s{sharpness:.0f}_sz{area:.0f}_b{brightness:.0f}_q{quality:.1f}"

        return quality, True, reason

    def _save_roi_image(self, roi: np.ndarray, track_id: int, quality: float,
                       is_valid: bool, reason: str = ""):
        """Save ROI image with metadata in filename."""
        try:
            timestamp = int(time.time() * 1000)
            status = "accepted" if is_valid else "rejected"

            filename = f"track_{track_id}_{timestamp}_{status}_q{quality:.1f}"
            if reason:
                reason_safe = reason.replace(" ", "_").replace("(", "").replace(")", "")[:40]
                filename += f"_{reason_safe}"
            filename += ".jpg"

            filepath = os.path.join(self.roi_candidates_dir, filename)
            cv2.imwrite(filepath, roi)

            logger.debug(f"[EnhancedROICollector] Saved: {filename}")
        except Exception as e:
            logger.error(f"[EnhancedROICollector] Save failed: {e}")

    def collect_roi(self, track_id: int, frame: np.ndarray,
                   bbox: Tuple[int, int, int, int]) -> bool:
        """Collect ROI with enhanced quality checks."""
        # Periodic cleanup
        self._maybe_cleanup()

        # Get or create collection
        if track_id not in self.collections:
            from src.classifier.ROICollectorService import TrackROICollection
            self.collections[track_id] = TrackROICollection(
                track_id=track_id,
                max_rois=self.max_rois_per_track
            )
            logger.info(f"[ROI_LIFECYCLE] T{track_id} ROI_COLLECTION_START | max_rois={self.max_rois_per_track}")

        collection = self.collections[track_id]

        # Stop if enough collected
        if collection.collected_count >= self.max_rois_per_track:
            return False

        # Extract ROI with padding
        x1, y1, x2, y2 = bbox
        pad = 10  # Increased from 5 for more context
        h, w = frame.shape[:2]

        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            return False

        # Enhanced quality check
        quality, is_valid, reason = self._compute_quality(roi)

        # Save if configured
        if self.save_all_rois or (self.save_roi_candidates and is_valid):
            self._save_roi_image(roi, track_id, quality, is_valid, reason)

        if not is_valid:
            collection.rejected_count += 1
            self._total_rejected += 1

            # Track rejection reasons
            reason_key = reason.split('_')[0]  # e.g., "blurry" from "blurry_sharp50"
            self._rejection_reasons[reason_key] = self._rejection_reasons.get(reason_key, 0) + 1

            logger.debug(
                f"[ROI_LIFECYCLE] T{track_id} ROI_REJECTED | "
                f"reason={reason} collected={collection.collected_count} "
                f"rejected={collection.rejected_count}"
            )
            return False

        # Collect the ROI
        collection.add_roi(roi, quality)
        self._total_collected += 1

        logger.info(
            f"[ROI_LIFECYCLE] T{track_id} ROI_COLLECTED | "
            f"quality={quality:.1f} count={collection.collected_count}/{self.max_rois_per_track} "
            f"best={collection.best_roi_quality:.1f} size={roi.shape[1]}x{roi.shape[0]}"
        )

        return True

    def get_best_roi(self, track_id: int) -> Optional[Tuple[np.ndarray, float]]:
        """Get best quality ROI."""
        if track_id not in self.collections:
            return None

        collection = self.collections[track_id]
        if collection.best_roi is None:
            return None

        return collection.best_roi, collection.best_roi_quality

    def get_all_rois(self, track_id: int) -> Optional[List[Tuple[np.ndarray, float]]]:
        """Get all collected ROIs."""
        if track_id not in self.collections:
            return None

        collection = self.collections[track_id]
        return list(zip(collection.rois, collection.qualities))

    def remove_track(self, track_id: int):
        """Remove track collection."""
        return self.collections.pop(track_id, None)

    def _maybe_cleanup(self):
        """Periodic cleanup of stale collections."""
        now = time.time()

        if now - self._last_cleanup_time < 30.0:
            return

        self._last_cleanup_time = now

        # Remove stale
        stale_ids = [
            tid for tid, coll in self.collections.items()
            if now - coll.last_updated > self.STALE_TIMEOUT_SECONDS
        ]

        for track_id in stale_ids:
            logger.warning(f"[EnhancedROICollector] Removing stale track {track_id}")
            del self.collections[track_id]

        # Limit total
        if len(self.collections) > self.MAX_TRACKS:
            sorted_collections = sorted(
                self.collections.items(),
                key=lambda x: x[1].created_at
            )
            to_remove = len(self.collections) - self.MAX_TRACKS
            for track_id, _ in sorted_collections[:to_remove]:
                logger.warning(f"[EnhancedROICollector] Removing old track {track_id}")
                del self.collections[track_id]

    def get_statistics(self) -> Dict:
        """Get comprehensive statistics."""
        return {
            'active_tracks': len(self.collections),
            'total_collected': self._total_collected,
            'total_rejected': self._total_rejected,
            'reject_rate': self._total_rejected / max(1, self._total_collected + self._total_rejected),
            'rejection_reasons': self._rejection_reasons.copy()
        }

    def cleanup(self):
        """Clean up resources."""
        stats = self.get_statistics()
        logger.info(
            f"[EnhancedROICollector] Cleanup:\n"
            f"  Active tracks: {stats['active_tracks']}\n"
            f"  Collected: {stats['total_collected']}\n"
            f"  Rejected: {stats['total_rejected']} ({stats['reject_rate']:.1%})\n"
            f"  Rejection reasons: {stats['rejection_reasons']}"
        )
        self.collections.clear()
