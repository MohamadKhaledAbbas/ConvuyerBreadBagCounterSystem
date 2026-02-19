"""
Abstract interfaces for tracking components.

Defines contracts for tracker implementations to ensure loose coupling.
"""

from abc import ABC, abstractmethod
from collections import deque
from typing import Dict, List, Optional, Tuple, Protocol

from src.detection.BaseDetection import Detection


class TrackedObject(Protocol):
    """
    Protocol for tracked object implementations.

    Defines the expected interface for tracked objects across different tracker implementations.
    Implementations should provide these attributes and methods.
    """
    track_id: int
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    age: int
    hits: int
    time_since_update: int
    position_history: deque
    bbox_history: deque
    created_at: float
    last_seen_at: float
    classified: bool

    # Entry type classification (diagnostics)
    entry_type: str  # 'bottom_entry', 'thrown_entry', 'midway_entry'

    # Ghost recovery tracking
    ghost_recovery_count: int  # Number of times re-associated after occlusion

    # Shadow/merge tracking
    shadow_of: Optional[int]  # track_id this is a shadow of (if merge detected)
    shadow_tracks: Dict[int, 'TrackedObject']  # Shadows riding on this track

    @property
    def center(self) -> Tuple[int, int]:
        """Return center point of current bbox."""
        ...

    @property
    def velocity(self) -> Optional[Tuple[float, float]]:
        """Return estimated velocity (vx, vy) or None."""
        ...

    def update(self, detection: Detection) -> None:
        """Update track with new detection."""
        ...

    def predict(self) -> Tuple[int, int, int, int]:
        """Predict next position based on velocity."""
        ...

    def mark_missed(self) -> None:
        """Mark frame without detection."""
        ...


class TrackEvent(Protocol):
    """
    Protocol for track completion event implementations.

    Emitted when a track completes (exits frame or is lost).
    """
    track_id: int
    event_type: str  # 'track_completed', 'track_lost'
    bbox_history: List[Tuple[int, int, int, int]]
    position_history: List[Tuple[int, int]]
    total_frames: int
    created_at: float
    ended_at: float
    avg_confidence: float
    exit_direction: str

    # Enhanced lifecycle fields
    entry_type: str  # 'bottom_entry', 'thrown_entry', 'midway_entry'
    suspected_duplicate: bool  # True only for midway_entry
    ghost_recovery_count: int  # Times track was re-associated after occlusion
    occlusion_events: List[dict]  # [{lost_at_y, recovered_at_y, gap_seconds}]
    shadow_of: Optional[int]  # track_id this was a shadow of
    shadow_count: int  # Number of shadow tracks when this track exited
    merge_events: List[dict]  # [{merged_track_id, merge_y, unmerge_y}]

    @property
    def duration_seconds(self) -> float:
        """Track duration in seconds."""
        ...

    @property
    def distance_traveled(self) -> float:
        """Euclidean distance traveled by track."""
        ...


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
