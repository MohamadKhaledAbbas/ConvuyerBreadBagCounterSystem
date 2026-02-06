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
- Multi-criteria cost function (IoU + centroid distance + motion consistency)
- Velocity-based prediction for better matching
- Configurable exit zones
- Minimum track duration filtering
- Memory-efficient history management
"""

import math
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
from src.utils.Utils import compute_iou, compute_centroid

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
    this is a simple position-based tracker with enhanced motion estimation.
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

    # Smoothed velocity (exponential moving average)
    _smooth_velocity: Optional[Tuple[float, float]] = None
    _velocity_alpha: float = 0.3  # EMA smoothing factor (0-1, higher = more responsive)

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
        Return smoothed velocity estimate.

        Returns:
            Tuple of (vx, vy) pixels per frame or None
        """
        return self._smooth_velocity

    def _compute_instant_velocity(self) -> Optional[Tuple[float, float]]:
        """
        Compute instantaneous velocity from last two positions.

        Returns:
            Tuple of (vx, vy) pixels per frame or None
        """
        if len(self.position_history) < 2:
            return None
        
        last_pos = self.position_history[-1]
        prev_pos = self.position_history[-2]

        return (
            float(last_pos[0] - prev_pos[0]),
            float(last_pos[1] - prev_pos[1])
        )

    def _update_smooth_velocity(self):
        """Update exponentially smoothed velocity estimate."""
        instant_vel = self._compute_instant_velocity()

        if instant_vel is None:
            return

        if self._smooth_velocity is None:
            # Initialize with first velocity observation
            self._smooth_velocity = instant_vel
        else:
            # Exponential moving average
            alpha = self._velocity_alpha
            self._smooth_velocity = (
                alpha * instant_vel[0] + (1 - alpha) * self._smooth_velocity[0],
                alpha * instant_vel[1] + (1 - alpha) * self._smooth_velocity[1]
            )

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

        # Update smoothed velocity
        self._update_smooth_velocity()

    def predict(self) -> Tuple[int, int, int, int]:
        """
        Predict next position based on velocity.
        
        Uses smoothed velocity for more stable predictions.
        Prediction distance scales with time_since_update for missed frames.

        Returns:
            Predicted bbox
        """
        vel = self.velocity
        if vel is None:
            return self.bbox
        
        vx, vy = vel
        # Scale prediction by frames since last update (for tracking during missed detections)
        scale = self.time_since_update + 1
        x1, y1, x2, y2 = self.bbox
        
        return (
            int(x1 + vx * scale),
            int(y1 + vy * scale),
            int(x2 + vx * scale),
            int(y2 + vy * scale)
        )
    
    def mark_missed(self):
        """Mark frame without detection."""
        self.time_since_update += 1
        self.age += 1

        # Update predicted position in history based on velocity
        # This helps maintain motion consistency even during missed detections
        vel = self.velocity
        if vel is not None and len(self.position_history) > 0:
            last_pos = self.position_history[-1]
            predicted_pos = (
                int(last_pos[0] + vel[0]),
                int(last_pos[1] + vel[1])
            )
            # Don't add to history, but update bbox for next prediction
            x1, y1, x2, y2 = self.bbox
            self.bbox = (
                int(x1 + vel[0]),
                int(y1 + vel[1]),
                int(x2 + vel[0]),
                int(y2 + vel[1])
            )


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
    Enhanced tracker optimized for conveyor belt movement.

    Key features:
    - Multi-criteria cost function (IoU + centroid + motion + size)
    - Two-stage matching: strict first pass, relaxed second pass
    - Exponentially smoothed velocity for stable motion prediction
    - Velocity-based bbox prediction during missed detections

    Workflow:
    1. Detection → Multi-criteria match to existing tracks (IoU + distance + motion)
    2. Unmatched → Second-stage centroid-based matching (fallback)
    3. Still unmatched detections → new tracks
    4. Lost tracks → emit completion event after max_age
    5. Tracks that leave frame → emit completion event
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
        
        # Log configuration
        multi_criteria = getattr(self.config, 'use_multi_criteria_matching', True)
        second_stage = getattr(self.config, 'use_second_stage_matching', True)
        min_conf_new = getattr(self.config, 'min_confidence_new_track', 0.7)

        logger.info(
            f"[ConveyorTracker] Initialized: iou_threshold={self.config.iou_threshold}, "
            f"max_age={self.config.max_frames_without_detection}, min_hits={self.config.min_track_duration_frames}, "
            f"min_conf_new_track={min_conf_new:.2f}, "
            f"multi_criteria={multi_criteria}, second_stage={second_stage}"
        )
    
    def _compute_cost_matrix(
        self,
        tracks: List[TrackedObject],
        detections: List[Detection]
    ) -> np.ndarray:
        """
        Compute multi-criteria cost matrix for Hungarian matching.

        Uses a weighted combination of:
        - IoU (overlap between predicted and detected boxes)
        - Centroid distance (normalized by frame diagonal)
        - Motion consistency (how well detection aligns with predicted motion)
        - Size similarity (aspect ratio and area similarity)

        Args:
            tracks: List of existing tracks
            detections: List of new detections
            
        Returns:
            Cost matrix (lower = better match)
        """
        cost_matrix = np.zeros((len(tracks), len(detections)))
        
        # Weights for different criteria (must sum to 1.0)
        w_iou = 0.4       # IoU similarity
        w_centroid = 0.3  # Centroid distance
        w_motion = 0.2    # Motion consistency
        w_size = 0.1      # Size similarity

        # Frame diagonal for distance normalization
        if self.frame_width and self.frame_height:
            frame_diag = math.sqrt(self.frame_width**2 + self.frame_height**2)
        else:
            frame_diag = 1000.0  # Default fallback

        # Maximum centroid distance for normalization (as fraction of frame diagonal)
        max_centroid_dist = frame_diag * 0.25  # 25% of diagonal

        for i, track in enumerate(tracks):
            # Use predicted position for matching
            predicted_bbox = track.predict()
            predicted_center = (
                (predicted_bbox[0] + predicted_bbox[2]) // 2,
                (predicted_bbox[1] + predicted_bbox[3]) // 2
            )
            track_velocity = track.velocity
            track_area = (predicted_bbox[2] - predicted_bbox[0]) * (predicted_bbox[3] - predicted_bbox[1])
            track_aspect = (predicted_bbox[2] - predicted_bbox[0]) / max(1, predicted_bbox[3] - predicted_bbox[1])

            for j, det in enumerate(detections):
                det_center = compute_centroid(det.bbox)
                det_area = (det.bbox[2] - det.bbox[0]) * (det.bbox[3] - det.bbox[1])
                det_aspect = (det.bbox[2] - det.bbox[0]) / max(1, det.bbox[3] - det.bbox[1])

                # 1. IoU cost (1 - IoU)
                iou = compute_iou(predicted_bbox, det.bbox)
                iou_cost = 1.0 - iou

                # 2. Centroid distance cost (normalized)
                dx = det_center[0] - predicted_center[0]
                dy = det_center[1] - predicted_center[1]
                centroid_dist = math.sqrt(dx**2 + dy**2)
                centroid_cost = min(1.0, centroid_dist / max_centroid_dist)

                # 3. Motion consistency cost
                # If we have velocity, check if the detection is in the expected direction
                motion_cost = 0.5  # Default: neutral
                if track_velocity is not None and track.hits >= 3:
                    vx, vy = track_velocity
                    vel_magnitude = math.sqrt(vx**2 + vy**2)

                    if vel_magnitude > 2.0:  # Only if moving significantly
                        # Vector from current center to detection
                        current_center = track.center
                        move_dx = det_center[0] - current_center[0]
                        move_dy = det_center[1] - current_center[1]
                        move_magnitude = math.sqrt(move_dx**2 + move_dy**2)

                        if move_magnitude > 1.0:
                            # Cosine similarity between velocity and movement
                            cos_sim = (vx * move_dx + vy * move_dy) / (vel_magnitude * move_magnitude)
                            # Convert to cost (cos_sim = 1 means same direction, -1 opposite)
                            motion_cost = (1.0 - cos_sim) / 2.0  # Scale to [0, 1]

                # 4. Size similarity cost
                area_ratio = min(track_area, det_area) / max(1, max(track_area, det_area))
                aspect_diff = abs(track_aspect - det_aspect) / max(0.01, max(track_aspect, det_aspect))
                size_cost = 1.0 - (0.7 * area_ratio + 0.3 * (1.0 - min(1.0, aspect_diff)))

                # Combined weighted cost
                cost_matrix[i, j] = (
                    w_iou * iou_cost +
                    w_centroid * centroid_cost +
                    w_motion * motion_cost +
                    w_size * size_cost
                )

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

    def _second_stage_matching(
        self,
        tracks: List[TrackedObject],
        detections: List[Detection],
        track_indices: List[int],
        det_indices: List[int]
    ) -> List[Tuple[int, int]]:
        """
        Second-stage matching using relaxed centroid distance.

        This catches tracks that moved more than expected (fast motion,
        detection jitter, etc.) where IoU-based matching failed.

        Uses:
        - Centroid distance with larger threshold
        - Motion direction consistency (must be moving roughly in predicted direction)

        Args:
            tracks: Unmatched tracks
            detections: Unmatched detections
            track_indices: Original indices of tracks in track_list
            det_indices: Original indices of detections in detection list

        Returns:
            List of (track_idx, det_idx) matches using original indices
        """
        if not tracks or not detections:
            return []

        matches = []

        # Get configurable parameters
        max_dist_base = getattr(self.config, 'second_stage_max_distance', 150.0)
        threshold = getattr(self.config, 'second_stage_threshold', 0.8)

        # Build cost matrix based on centroid distance and motion consistency
        cost_matrix = np.zeros((len(tracks), len(detections)))

        for i, track in enumerate(tracks):
            predicted_bbox = track.predict()
            predicted_center = (
                (predicted_bbox[0] + predicted_bbox[2]) // 2,
                (predicted_bbox[1] + predicted_bbox[3]) // 2
            )
            track_velocity = track.velocity

            # Adjust max distance based on velocity magnitude
            max_dist = max_dist_base
            if track_velocity is not None:
                vel_mag = math.sqrt(track_velocity[0]**2 + track_velocity[1]**2)
                # Allow larger search radius for faster tracks
                max_dist = max(max_dist_base, vel_mag * 5)

            for j, det in enumerate(detections):
                det_center = compute_centroid(det.bbox)

                # Centroid distance
                dx = det_center[0] - predicted_center[0]
                dy = det_center[1] - predicted_center[1]
                dist = math.sqrt(dx**2 + dy**2)

                # If too far, mark as invalid
                if dist > max_dist:
                    cost_matrix[i, j] = 1e6
                    continue

                # Check motion consistency
                motion_penalty = 0.0
                if track_velocity is not None and track.hits >= 3:
                    vx, vy = track_velocity
                    vel_mag = math.sqrt(vx**2 + vy**2)

                    if vel_mag > 2.0:
                        # Check if detection is in roughly the right direction
                        current_center = track.center
                        move_dx = det_center[0] - current_center[0]
                        move_dy = det_center[1] - current_center[1]
                        move_mag = math.sqrt(move_dx**2 + move_dy**2)

                        if move_mag > 1.0:
                            cos_sim = (vx * move_dx + vy * move_dy) / (vel_mag * move_mag)
                            # Penalize if moving in opposite direction
                            if cos_sim < -0.5:
                                motion_penalty = 0.5
                            elif cos_sim < 0:
                                motion_penalty = 0.2

                # Normalized distance cost
                cost_matrix[i, j] = (dist / max_dist) + motion_penalty

        # Greedy matching for second stage
        used_dets = set()
        for i in range(len(tracks)):
            best_j = -1
            best_cost = threshold

            for j in range(len(detections)):
                if j not in used_dets and cost_matrix[i, j] < best_cost:
                    best_cost = cost_matrix[i, j]
                    best_j = j

            if best_j >= 0:
                matches.append((track_indices[i], det_indices[best_j]))
                used_dets.add(best_j)

                logger.debug(
                    f"[ConveyorTracker] Second-stage match: "
                    f"track {tracks[i].track_id} -> det {best_j} (cost={best_cost:.3f})"
                )

        return matches

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
        
        # Get confidence threshold for new tracks
        min_conf_new_track = getattr(self.config, 'min_confidence_new_track', 0.7)

        if not track_list:
            # No existing tracks - create new ones for high-confidence detections only
            for det in detections:
                if det.confidence >= min_conf_new_track:
                    self._create_track(det)
                else:
                    logger.debug(
                        f"[ConveyorTracker] Skipping low-confidence detection "
                        f"(conf={det.confidence:.2f} < {min_conf_new_track:.2f}) - not creating initial track"
                    )
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
        
        # First stage: Strict matching with combined criteria
        threshold = 1 - self.config.iou_threshold
        matches, unmatched_tracks, unmatched_dets = self._linear_assignment(
            cost_matrix, threshold
        )
        
        # Second stage: Relaxed matching for remaining tracks using centroid distance
        # This helps recover tracks that moved more than expected between frames
        use_second_stage = getattr(self.config, 'use_second_stage_matching', True)
        if use_second_stage and unmatched_tracks and unmatched_dets:
            second_stage_matches = self._second_stage_matching(
                [track_list[i] for i in unmatched_tracks],
                [detections[j] for j in unmatched_dets],
                list(unmatched_tracks),
                list(unmatched_dets)
            )

            # Update indices based on second stage matches
            for track_idx, det_idx in second_stage_matches:
                matches.append((track_idx, det_idx))
                unmatched_tracks.discard(track_idx)
                unmatched_dets.discard(det_idx)

        # Update matched tracks
        for track_idx, det_idx in matches:
            track_list[track_idx].update(detections[det_idx])
        
        # Mark unmatched tracks as missed
        for track_idx in unmatched_tracks:
            track_list[track_idx].mark_missed()
        
        # Create new tracks for unmatched detections (with confidence filtering)
        # Only create tracks for high-confidence detections to avoid noise
        # Note: min_conf_new_track was already retrieved at the start of update()
        for det_idx in unmatched_dets:
            detection = detections[det_idx]
            if detection.confidence >= min_conf_new_track:
                self._create_track(detection)
            else:
                logger.debug(
                    f"[ConveyorTracker] Skipping low-confidence detection "
                    f"(conf={detection.confidence:.2f} < {min_conf_new_track:.2f}) - not creating track"
                )

        # Check for completed tracks
        self._check_completed_tracks()
        
        return list(self.tracks.values())
    
    def _create_track(self, detection: Detection) -> TrackedObject:
        """Create a new track from detection."""
        # Get velocity smoothing alpha from config
        velocity_alpha = getattr(self.config, 'velocity_smoothing_alpha', 0.3)

        track = TrackedObject(
            track_id=self._next_id,
            bbox=detection.bbox,
            confidence=detection.confidence,
            _velocity_alpha=velocity_alpha
        )
        track.position_history.append(track.center)
        track.bbox_history.append(track.bbox)
        
        self.tracks[self._next_id] = track

        logger.info(
            f"[TRACK_LIFECYCLE] T{self._next_id} CREATED | "
            f"bbox={detection.bbox} center={track.center} conf={detection.confidence:.2f}"
        )

        self._next_id += 1
        
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

                # Calculate track statistics
                velocity = track.velocity
                vel_str = f"vel=({velocity[0]:.1f},{velocity[1]:.1f})" if velocity else "vel=None"
                distance = 0.0
                if len(track.position_history) >= 2:
                    start = track.position_history[0]
                    end = track.position_history[-1]
                    distance = ((end[0]-start[0])**2 + (end[1]-start[1])**2)**0.5

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
                    f"[TRACK_LIFECYCLE] T{track_id} COMPLETED | "
                    f"type={event_type} exit={exit_dir} hits={track.hits} missed={track.time_since_update} "
                    f"duration={event.duration_seconds:.2f}s distance={distance:.0f}px {vel_str} "
                    f"positions={len(track.position_history)} avg_conf={avg_conf:.2f}"
                )

        # Remove completed tracks (use set to avoid duplicates)
        for track_id in set(tracks_to_remove):
            if track_id in self.tracks:
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
