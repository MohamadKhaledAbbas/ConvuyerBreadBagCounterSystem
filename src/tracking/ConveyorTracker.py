"""
Simplified linear tracker for conveyor belt environment.

This is the KEY DIFFERENCE from v1 (BreadBagCounterSystem):
- v1 used EventCentricTracker with complex state machines (open/closing/closed)
- v2 uses a simple IoU-based linear tracker for conveyor movement

Conveyor assumptions:
1. Objects move in a predictable linear direction
2. Objects appear on one side, move across, disappear on the other side
3. No complex worker interactions or occlusions
4. No need for open/closing/closed state detection

Production Features:
- Velocity-based prediction for better matching
- Configurable exit zones
- Minimum track duration filtering
- Memory-efficient history management
"""

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple, Set

import numpy as np

from src.config.tracking_config import TrackingConfig
from src.detection.BaseDetection import Detection
from src.tracking.ITracker import ITracker
from src.utils.AppLogging import logger
from src.utils.Utils import compute_iou

# Try to import scipy for optimal linear assignment
try:
    from scipy.optimize import linear_sum_assignment  # noqa
    HAS_SCIPY = True
except ImportError:
    linear_sum_assignment = None  # type: ignore
    HAS_SCIPY = False


class ExitDirection(Enum):
    """Direction objects exit the frame."""
    ANY = "any"
    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"


@dataclass
class TrackedObject:
    """
    Represents a tracked bread bag on the conveyor.
    
    Unlike v1's BagStateMonitor with complex state machines,
    this is a simple position-based tracker.
    """
    track_id: int
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    
    # Tracking state
    age: int = 0  # Total frames since creation
    hits: int = 1  # Consecutive detections
    time_since_update: int = 0  # Frames since last detection
    
    # Position history for motion estimation (using deque for efficient memory management)
    position_history: deque = field(default_factory=lambda: deque(maxlen=30))
    bbox_history: deque = field(default_factory=lambda: deque(maxlen=30))

    # Timestamps
    created_at: float = field(default_factory=time.time)
    last_seen_at: float = field(default_factory=time.time)
    
    # Classification tracking
    classified: bool = False
    
    @property
    def center(self) -> Tuple[int, int]:
        """Return center point of current bbox."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    @property
    def velocity(self) -> Optional[Tuple[float, float]]:
        """
        Estimate velocity from position history.
        
        Returns:
            Tuple of (vx, vy) pixels per frame or None
        """
        if len(self.position_history) < 2:
            return None
        
        # Use last few positions for smoothed velocity
        n = min(5, len(self.position_history))

        # Convert deque to list for slicing, or access elements directly
        # More efficient: access first and last of recent n elements
        if n == len(self.position_history):
            # Use all elements
            first_pos = self.position_history[0]
            last_pos = self.position_history[-1]
        else:
            # Get last n elements by converting to list
            recent = list(self.position_history)[-n:]
            first_pos = recent[0]
            last_pos = recent[-1]

        dx = last_pos[0] - first_pos[0]
        dy = last_pos[1] - first_pos[1]

        return dx / (n - 1), dy / (n - 1)

    def update(self, detection: Detection):
        """Update track with new detection."""
        self.bbox = detection.bbox
        self.confidence = detection.confidence
        self.hits += 1
        self.time_since_update = 0
        self.last_seen_at = time.time()
        
        # Update history (deque automatically maintains maxlen)
        self.position_history.append(self.center)
        self.bbox_history.append(self.bbox)

    def predict(self) -> Tuple[int, int, int, int]:
        """
        Predict next position based on velocity.
        
        Returns:
            Predicted bbox
        """
        vel = self.velocity
        if vel is None:
            return self.bbox
        
        vx, vy = vel
        x1, y1, x2, y2 = self.bbox
        
        return (
            int(x1 + vx),
            int(y1 + vy),
            int(x2 + vx),
            int(y2 + vy)
        )
    
    def mark_missed(self):
        """Mark frame without detection."""
        self.time_since_update += 1
        self.age += 1


@dataclass 
class TrackEvent:
    """Event emitted by tracker for completed tracks."""
    track_id: int
    event_type: str  # 'track_completed', 'track_lost'
    bbox_history: List[Tuple[int, int, int, int]]
    position_history: List[Tuple[int, int]]
    total_frames: int
    created_at: float
    ended_at: float
    avg_confidence: float = 0.0
    exit_direction: str = "unknown"

    @property
    def duration_seconds(self) -> float:
        """Track duration in seconds."""
        return self.ended_at - self.created_at

    @property
    def distance_traveled(self) -> float:
        """Euclidean distance traveled by track."""
        if len(self.position_history) < 2:
            return 0.0
        start = self.position_history[0]
        end = self.position_history[-1]
        return np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)


class ConveyorTracker(ITracker):
    """
    Simple IoU-based tracker optimized for conveyor belt movement.
    
    Key simplifications over v1:
    - No state machine (open/closing/closed)
    - No ByteTrack integration
    - No complex centroid association
    - Simple IoU matching with velocity prediction
    
    Workflow:
    1. Detection → IoU match to existing tracks
    2. Unmatched detections → new tracks
    3. Lost tracks → emit completion event after max_age
    4. Tracks that leave frame → emit completion event
    """
    
    def __init__(self, config: Optional[TrackingConfig] = None):
        """
        Initialize conveyor tracker.
        
        Args:
            config: Tracking configuration
        """
        self.config = config or TrackingConfig()
        
        # Active tracks
        self.tracks: Dict[int, TrackedObject] = {}
        
        # Track ID counter
        self._next_id = 1
        
        # Frame dimensions for exit detection
        self.frame_width: Optional[int] = None
        self.frame_height: Optional[int] = None
        
        # Completed tracks buffer (for batch processing)
        self.completed_tracks: List[TrackEvent] = []
        
        logger.info(
            f"[ConveyorTracker] Initialized with iou_threshold={self.config.iou_threshold}, "
            f"max_age={self.config.max_frames_without_detection}, min_hits={self.config.min_track_duration_frames}"
        )
    
    def _compute_cost_matrix(
        self,
        tracks: List[TrackedObject],
        detections: List[Detection]
    ) -> np.ndarray:
        """
        Compute IoU-based cost matrix for Hungarian matching.
        
        Args:
            tracks: List of existing tracks
            detections: List of new detections
            
        Returns:
            Cost matrix (1 - IoU)
        """
        cost_matrix = np.zeros((len(tracks), len(detections)))
        
        for i, track in enumerate(tracks):
            # Use predicted position for matching
            predicted_bbox = track.predict()
            
            for j, det in enumerate(detections):
                iou = compute_iou(predicted_bbox, det.bbox)
                cost_matrix[i, j] = 1 - iou
        
        return cost_matrix
    
    def _linear_assignment(
        self,
        cost_matrix: np.ndarray,
        threshold: float
    ) -> Tuple[List[Tuple[int, int]], Set[int], Set[int]]:
        """
        Perform linear assignment (Hungarian algorithm).
        
        Returns:
            Tuple of (matches, unmatched_tracks, unmatched_detections)
        """
        if cost_matrix.size == 0:
            return (
                [],
                set(range(cost_matrix.shape[0])),
                set(range(cost_matrix.shape[1]))
            )
        
        if HAS_SCIPY:
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
        else:
            # Greedy fallback
            row_indices = []
            col_indices = []
            used_cols = set()
            
            for i in range(cost_matrix.shape[0]):
                best_j = -1
                best_cost = float('inf')
                
                for j in range(cost_matrix.shape[1]):
                    if j not in used_cols and cost_matrix[i, j] < best_cost:
                        best_cost = cost_matrix[i, j]
                        best_j = j
                
                if best_j >= 0:
                    row_indices.append(i)
                    col_indices.append(best_j)
                    used_cols.add(best_j)
        
        # Filter by threshold
        matches = []
        unmatched_tracks = set(range(cost_matrix.shape[0]))
        unmatched_dets = set(range(cost_matrix.shape[1]))
        
        for i, j in zip(row_indices, col_indices):
            if cost_matrix[i, j] <= threshold:
                matches.append((i, j))
                unmatched_tracks.discard(i)
                unmatched_dets.discard(j)
        
        return matches, unmatched_tracks, unmatched_dets

    def _get_exit_direction(self, track: TrackedObject) -> Optional[str]:
        """
        Determine which direction track is exiting.

        Args:
            track: Tracked object

        Returns:
            Exit direction string or None if not exiting
        """
        if self.frame_width is None or self.frame_height is None:
            return None

        cx, cy = track.center
        margin = getattr(self.config, 'exit_margin_pixels', 20)

        if cx < margin:
            return "left"
        if cx > self.frame_width - margin:
            return "right"
        if cy < margin:
            return "top"
        if cy > self.frame_height - margin:
            return "bottom"

        return None

    def _is_exiting_frame(self, track: TrackedObject) -> bool:
        """
        Check if track is exiting the frame.
        
        Args:
            track: Tracked object
            
        Returns:
            True if track center is near frame edge
        """
        return self._get_exit_direction(track) is not None

    def update(
        self,
        detections: List[Detection],
        frame_shape: Optional[Tuple[int, int]] = None
    ) -> List[TrackedObject]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of new detections
            frame_shape: Optional frame dimensions (height, width)
            
        Returns:
            List of active tracks after update
        """
        # Update frame dimensions
        if frame_shape is not None:
            self.frame_height, self.frame_width = frame_shape
        
        # Get list of active tracks
        track_list = list(self.tracks.values())
        
        if not track_list:
            # No existing tracks - create new ones for all detections
            for det in detections:
                self._create_track(det)
            return list(self.tracks.values())
        
        if not detections:
            # No detections - mark all tracks as missed
            for track in track_list:
                track.mark_missed()
            
            # Check for track completion
            self._check_completed_tracks()
            return list(self.tracks.values())
        
        # Compute cost matrix
        cost_matrix = self._compute_cost_matrix(track_list, detections)
        
        # Perform assignment
        threshold = 1 - self.config.iou_threshold
        matches, unmatched_tracks, unmatched_dets = self._linear_assignment(
            cost_matrix, threshold
        )
        
        # Update matched tracks
        for track_idx, det_idx in matches:
            track_list[track_idx].update(detections[det_idx])
        
        # Mark unmatched tracks as missed
        for track_idx in unmatched_tracks:
            track_list[track_idx].mark_missed()
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            self._create_track(detections[det_idx])
        
        # Check for completed tracks
        self._check_completed_tracks()
        
        return list(self.tracks.values())
    
    def _create_track(self, detection: Detection) -> TrackedObject:
        """Create a new track from detection."""
        track = TrackedObject(
            track_id=self._next_id,
            bbox=detection.bbox,
            confidence=detection.confidence
        )
        track.position_history.append(track.center)
        track.bbox_history.append(track.bbox)
        
        self.tracks[self._next_id] = track
        self._next_id += 1
        
        logger.debug(f"[ConveyorTracker] Created track {track.track_id}")
        
        return track
    
    def _check_completed_tracks(self):
        """Check for tracks that should be marked as completed."""
        tracks_to_remove = []
        
        for track_id, track in self.tracks.items():
            should_complete = False
            event_type = 'track_completed'
            
            # Check if track exceeded max age
            if track.time_since_update > self.config.max_frames_without_detection:
                should_complete = True
                event_type = 'track_lost'
            
            # Check if track is exiting frame and has been tracked long enough
            elif (
                track.hits >= self.config.min_track_duration_frames and
                self._is_exiting_frame(track)
            ):
                should_complete = True
                event_type = 'track_completed'
            
            if should_complete:
                # Calculate average confidence
                avg_conf = track.confidence  # Use last confidence for now

                # Get exit direction
                exit_dir = self._get_exit_direction(track) or "timeout"

                # Emit completion event
                event = TrackEvent(
                    track_id=track_id,
                    event_type=event_type,
                    bbox_history=list(track.bbox_history),  # Convert deque to list
                    position_history=list(track.position_history),  # Convert deque to list
                    total_frames=track.age + track.hits,
                    created_at=track.created_at,
                    ended_at=time.time(),
                    avg_confidence=avg_conf,
                    exit_direction=exit_dir
                )
                self.completed_tracks.append(event)
                tracks_to_remove.append(track_id)
                
                logger.info(
                    f"[ConveyorTracker] Track {track_id} {event_type}: "
                    f"{track.hits} hits, {track.time_since_update} missed, "
                    f"exit={exit_dir}, duration={event.duration_seconds:.2f}s"
                )
        
        # Remove completed tracks
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
    
    def get_confirmed_tracks(self) -> List[TrackedObject]:
        """
        Get tracks that have been confirmed (min_track_duration_frames reached).

        Returns:
            List of confirmed tracks
        """
        return [
            track for track in self.tracks.values()
            if track.hits >= self.config.min_track_duration_frames
        ]
    
    def get_completed_events(self) -> List[TrackEvent]:
        """
        Get and clear completed track events.
        
        Returns:
            List of track completion events
        """
        events = self.completed_tracks.copy()
        self.completed_tracks.clear()
        return events
    
    def get_track(self, track_id: int) -> Optional[TrackedObject]:
        """Get a specific track by ID."""
        return self.tracks.get(track_id)
    
    def cleanup(self):
        """Clean up tracker resources."""
        self.tracks.clear()
        self.completed_tracks.clear()
        logger.info("[ConveyorTracker] Cleanup complete")
