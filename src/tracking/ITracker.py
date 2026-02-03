"""
Abstract interfaces for tracking components.

Defines contracts for tracker implementations to ensure loose coupling.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from dataclasses import dataclass

from src.detection.BaseDetection import Detection


@dataclass
class TrackedObject:
    """Represents a tracked object (defined in implementation)."""
    pass


@dataclass
class TrackEvent:
    """Event emitted when a track completes (defined in implementation)."""
    pass


class ITracker(ABC):
    """
    Abstract interface for object trackers.

    Implementations handle object tracking across frames with different strategies:
    - IoU-based tracking (ConveyorTracker)
    - Centroid tracking
    - Deep SORT, etc.
    """

    @abstractmethod
    def update(
        self,
        detections: List[Detection],
        frame_shape: Optional[Tuple[int, int]] = None
    ) -> List[TrackedObject]:
        """
        Update tracker with new detections.

        Args:
            detections: List of detections from current frame
            frame_shape: Optional frame dimensions (height, width)

        Returns:
            List of active tracked objects
        """
        pass

    @abstractmethod
    def get_confirmed_tracks(self) -> List[TrackedObject]:
        """
        Get tracks that have been confirmed (met minimum duration).

        Returns:
            List of confirmed tracks suitable for classification
        """
        pass

    @abstractmethod
    def get_completed_events(self) -> List[TrackEvent]:
        """
        Get and clear completed track events.

        Returns:
            List of track completion events for processing
        """
        pass

    @abstractmethod
    def cleanup(self):
        """Release tracker resources."""
        pass
