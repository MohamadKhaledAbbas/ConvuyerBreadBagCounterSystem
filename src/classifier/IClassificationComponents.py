"""
Abstract interfaces for classification components.

Defines contracts for ROI collection and async classification workers.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Callable, Dict

import numpy as np


class IROICollector(ABC):
    """
    Abstract interface for ROI collection during tracking.

    Collects and filters ROIs without blocking the main loop.
    """

    @abstractmethod
    def collect_roi(
        self,
        track_id: int,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> bool:
        """
        Collect ROI for a track if it passes quality checks.

        Args:
            track_id: Track identifier
            frame: Full frame image
            bbox: Bounding box (x1, y1, x2, y2)

        Returns:
            True if ROI was collected, False if rejected
        """
        pass

    @abstractmethod
    def get_best_roi(self, track_id: int) -> Optional[Tuple[np.ndarray, float]]:
        """
        Get the best quality ROI for a completed track.

        Args:
            track_id: Track identifier

        Returns:
            Tuple of (roi_image, quality_score) or None
        """
        pass

    @abstractmethod
    def remove_track(self, track_id: int):
        """
        Remove track data after classification.

        Args:
            track_id: Track identifier
        """
        pass

    @abstractmethod
    def cleanup(self):
        """Release resources."""
        pass


class IClassificationWorker(ABC):
    """
    Abstract interface for async classification workers.

    Handles classification in background thread to avoid blocking main loop.
    """

    @abstractmethod
    def start(self):
        """Start the background worker thread."""
        pass

    @abstractmethod
    def stop(self, timeout: float = 5.0):
        """
        Stop the background worker and wait for completion.

        Args:
            timeout: Maximum time to wait for pending jobs
        """
        pass

    @abstractmethod
    def submit_job(
        self,
        track_id: int,
        roi: np.ndarray,
        bbox_history: List[Tuple[int, int, int, int]],
        callback: Optional[Callable] = None,
        extra_rois: Optional[List[np.ndarray]] = None
    ) -> bool:
        """
        Submit classification job to worker queue.

        Args:
            track_id: Track identifier
            roi: Primary ROI image to classify
            bbox_history: Track history for context
            callback: Optional callback(track_id, class_name, confidence)
            extra_rois: Additional ROIs for voting-based classification

        Returns:
            True if job queued, False if queue full
        """
        pass

    @abstractmethod
    def get_statistics(self) -> Dict:
        """
        Get worker statistics.

        Returns:
            Dictionary with worker metrics
        """
        pass
