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
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
import time
import numpy as np

from src.detection.BaseDetection import Detection
from src.utils.Utils import compute_iou, compute_centroid
from src.config.tracking_config import TrackingConfig
from src.utils.AppLogging import logger


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
    age: int = 0  # Frames since creation
    hits: int = 1  # Consecutive detections
    time_since_update: int = 0  # Frames since last detection
    
    # Position history for motion estimation
    position_history: List[Tuple[int, int]] = field(default_factory=list)
    bbox_history: List[Tuple[int, int, int, int]] = field(default_factory=list)
    
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
        recent = self.position_history[-n:]
        
        dx = recent[-1][0] - recent[0][0]
        dy = recent[-1][1] - recent[0][1]
        
        return (dx / (n - 1), dy / (n - 1))
    
    def update(self, detection: Detection):
        """Update track with new detection."""
        self.bbox = detection.bbox
        self.confidence = detection.confidence
        self.hits += 1
        self.time_since_update = 0
        self.last_seen_at = time.time()
        
        # Update history
        self.position_history.append(self.center)
        self.bbox_history.append(self.bbox)
        
        # Limit history size
        if len(self.position_history) > 30:
            self.position_history = self.position_history[-30:]
            self.bbox_history = self.bbox_history[-30:]
    
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


class ConveyorTracker:
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
            f"max_age={self.config.max_age}, min_hits={self.config.min_hits}"
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
        try:
            from scipy.optimize import linear_sum_assignment
            use_scipy = True
        except ImportError:
            use_scipy = False
        
        if cost_matrix.size == 0:
            return (
                [],
                set(range(cost_matrix.shape[0])),
                set(range(cost_matrix.shape[1]))
            )
        
        if use_scipy:
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
        
        return (matches, unmatched_tracks, unmatched_dets)
    
    def _is_exiting_frame(self, track: TrackedObject) -> bool:
        """
        Check if track is exiting the frame.
        
        Args:
            track: Tracked object
            
        Returns:
            True if track center is near frame edge
        """
        if self.frame_width is None or self.frame_height is None:
            return False
        
        cx, cy = track.center
        margin = self.config.exit_margin
        
        return (
            cx < margin or
            cx > self.frame_width - margin or
            cy < margin or
            cy > self.frame_height - margin
        )
    
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
            if track.time_since_update > self.config.max_age:
                should_complete = True
                event_type = 'track_lost'
            
            # Check if track is exiting frame and has been tracked long enough
            elif (
                track.hits >= self.config.min_hits and
                self._is_exiting_frame(track)
            ):
                should_complete = True
                event_type = 'track_completed'
            
            if should_complete:
                # Emit completion event
                event = TrackEvent(
                    track_id=track_id,
                    event_type=event_type,
                    bbox_history=track.bbox_history.copy(),
                    position_history=track.position_history.copy(),
                    total_frames=track.age + track.hits,
                    created_at=track.created_at,
                    ended_at=time.time()
                )
                self.completed_tracks.append(event)
                tracks_to_remove.append(track_id)
                
                logger.info(
                    f"[ConveyorTracker] Track {track_id} {event_type}: "
                    f"{track.hits} hits, {track.time_since_update} missed"
                )
        
        # Remove completed tracks
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
    
    def get_confirmed_tracks(self) -> List[TrackedObject]:
        """
        Get tracks that have been confirmed (min_hits reached).
        
        Returns:
            List of confirmed tracks
        """
        return [
            track for track in self.tracks.values()
            if track.hits >= self.config.min_hits
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
