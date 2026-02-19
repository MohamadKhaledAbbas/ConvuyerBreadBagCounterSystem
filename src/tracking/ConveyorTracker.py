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
- Ghost track recovery for occlusion handling
- Shadow track / merge detection for merged bboxes near top
- Entry type classification for operator diagnostics
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

    # Travel path validation
    entry_center_y: Optional[int] = None  # Y position when track was first created

    # Entry type classification (diagnostics only)
    entry_type: str = "bottom_entry"  # 'bottom_entry', 'thrown_entry', 'midway_entry'
    _entry_classified: bool = False  # Whether entry type has been finalized

    # Ghost recovery tracking
    ghost_recovery_count: int = 0  # Number of times re-associated after occlusion
    occlusion_events: list = field(default_factory=list)  # [{lost_at_y, recovered_at_y, gap_seconds}]

    # Shadow/merge tracking
    shadow_of: Optional[int] = None  # track_id this is a shadow of (if merge detected)
    shadow_tracks: Dict = field(default_factory=dict)  # {track_id: TrackedObject} shadows riding on this track
    shadow_count: int = 0  # Number of shadow tracks when exited
    merge_events: list = field(default_factory=list)  # [{merged_track_id, merge_y, unmerge_y}]
    
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
        Note: The bbox is already shifted by velocity in mark_missed() for each
        missed frame, so we only need to add one more frame of velocity here.

        Returns:
            Predicted bbox
        """
        vel = self.velocity
        if vel is None:
            return self.bbox
        
        vx, vy = vel
        # Only add one frame of velocity since mark_missed() already updates bbox
        # for each missed frame. Using time_since_update would double-count.
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
    event_type: str  # 'track_completed', 'track_lost', 'track_invalid'
    bbox_history: List[Tuple[int, int, int, int]]
    position_history: List[Tuple[int, int]]
    total_frames: int
    created_at: float
    ended_at: float
    avg_confidence: float = 0.0
    exit_direction: str = "unknown"

    # Enhanced lifecycle fields
    entry_type: str = "bottom_entry"  # 'bottom_entry', 'thrown_entry', 'midway_entry'
    suspected_duplicate: bool = False  # True only for midway_entry
    ghost_recovery_count: int = 0  # Times track was re-associated after occlusion
    occlusion_events: list = field(default_factory=list)  # [{lost_at_y, recovered_at_y, gap_seconds}]
    shadow_of: Optional[int] = None  # track_id this was a shadow of
    shadow_count: int = 0  # Number of shadow tracks when this track exited
    merge_events: list = field(default_factory=list)  # [{merged_track_id, merge_y, unmerge_y}]

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
        
        # Ghost tracks buffer (for occlusion recovery)
        self.ghost_tracks: Dict[int, dict] = {}  # {track_id: {track, lost_at, predicted_pos}}
        
        # Track ID counter
        self._next_id = 1
        
        # Frame dimensions for exit detection
        self.frame_width: Optional[int] = None
        self.frame_height: Optional[int] = None
        
        # Completed tracks buffer (for batch processing)
        self.completed_tracks: List[TrackEvent] = []
        
        # Cache frequently accessed config values (performance optimization)
        # These are accessed every frame, so caching avoids repeated getattr() lookups
        self._min_conf_new_track = getattr(self.config, 'min_confidence_new_track', 0.7)
        self._use_second_stage = getattr(self.config, 'use_second_stage_matching', True)
        self._velocity_alpha = getattr(self.config, 'velocity_smoothing_alpha', 0.3)
        self._exit_margin_pixels = getattr(self.config, 'exit_margin_pixels', 20)
        self._bottom_exit_zone_ratio = getattr(self.config, 'bottom_exit_zone_ratio', 0.15)
        self._exit_zone_ratio = getattr(self.config, 'exit_zone_ratio', 0.15)
        self._require_full_travel = getattr(self.config, 'require_full_travel', True)
        self._min_travel_duration_seconds = getattr(self.config, 'min_travel_duration_seconds', 2.0)
        self._second_stage_max_distance = getattr(self.config, 'second_stage_max_distance', 150.0)
        self._second_stage_threshold = getattr(self.config, 'second_stage_threshold', 0.8)

        # Ghost track config
        self._ghost_max_age = getattr(self.config, 'ghost_track_max_age_seconds', 4.0)
        self._ghost_x_tolerance = getattr(self.config, 'ghost_track_x_tolerance_pixels', 50.0)
        self._ghost_max_y_gap_ratio = getattr(self.config, 'ghost_track_max_y_gap_ratio', 0.2)

        # Shadow/merge config
        self._merge_bbox_growth = getattr(self.config, 'merge_bbox_growth_threshold', 1.4)
        self._merge_spatial_tol = getattr(self.config, 'merge_spatial_tolerance_pixels', 50.0)
        self._merge_y_tol = getattr(self.config, 'merge_y_tolerance_pixels', 30.0)

        # Entry type config
        self._bottom_entry_zone_ratio = getattr(self.config, 'bottom_entry_zone_ratio', 0.4)
        self._thrown_entry_min_velocity = getattr(self.config, 'thrown_entry_min_velocity', 15.0)
        self._thrown_entry_detection_frames = getattr(self.config, 'thrown_entry_detection_frames', 5)

        # Log configuration
        multi_criteria = getattr(self.config, 'use_multi_criteria_matching', True)

        logger.info(
            f"[ConveyorTracker] Initialized: iou_threshold={self.config.iou_threshold}, "
            f"max_age={self.config.max_frames_without_detection}, min_hits={self.config.min_track_duration_frames}, "
            f"min_conf_new_track={self._min_conf_new_track:.2f}, "
            f"multi_criteria={multi_criteria}, second_stage={self._use_second_stage}, "
            f"ghost_max_age={self._ghost_max_age}s, merge_growth={self._merge_bbox_growth}"
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
            max_dist = self._second_stage_max_distance
            if track_velocity is not None:
                vel_mag = math.sqrt(track_velocity[0]**2 + track_velocity[1]**2)
                # Allow larger search radius for faster tracks (8x velocity)
                # This handles cases where bags move faster than expected
                max_dist = max(self._second_stage_max_distance, vel_mag * 8)

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
            best_cost = self._second_stage_threshold

            for j in range(len(detections)):
                if j not in used_dets and cost_matrix[i, j] < best_cost:
                    best_cost = cost_matrix[i, j]
                    best_j = j

            if best_j >= 0:
                matches.append((track_indices[i], det_indices[best_j]))
                used_dets.add(best_j)

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

        if cx < self._exit_margin_pixels:
            return "left"
        if cx > self.frame_width - self._exit_margin_pixels:
            return "right"
        if cy < self._exit_margin_pixels:
            return "top"
        if cy > self.frame_height - self._exit_margin_pixels:
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

    def _is_in_bottom_exit_zone(self, y: int) -> bool:
        """
        Check if Y coordinate is in the bottom exit zone (invalid exit).

        Tracks exiting from the bottom are moving in the wrong direction
        and should be marked as invalid.

        Args:
            y: Y coordinate to check

        Returns:
            True if in bottom exit zone (invalid exit direction)
        """
        if self.frame_height is None:
            return False  # Can't validate without frame dimensions
        bottom_y_threshold = self.frame_height * (1.0 - self._bottom_exit_zone_ratio)
        return y >= bottom_y_threshold

    def _is_in_exit_zone(self, y: int) -> bool:
        """
        Check if Y coordinate is in the exit zone (top of frame).

        Bread bags disappear at the top, so the exit zone is the
        upper portion of the frame.

        Args:
            y: Y coordinate to check

        Returns:
            True if in exit zone
        """
        if self.frame_height is None:
            return False  # Can't validate without frame dimensions
        exit_y_threshold = self.frame_height * self._exit_zone_ratio
        return y <= exit_y_threshold

    def _has_valid_travel_path(self, track: TrackedObject) -> bool:
        """
        Check if track has a valid travel path using adaptive time-based validation.

        A valid travel path means:
        1. Track has been visible for minimum required duration (adaptive based on entry position)
        2. Track is exiting through the top exit zone (valid direction)
        3. Track is NOT exiting through the bottom (invalid direction)

        This approach is more robust than fixed time thresholds because:
        - Adapts to late detections (objects appearing mid-frame get shorter duration requirements)
        - Filters out noise/transient detections (too short duration)
        - Ensures objects traveled the expected path (exit from top)

        Args:
            track: Tracked object to validate

        Returns:
            True if the track has a valid travel path
        """
        if not self._require_full_travel:
            return True

        # Get current position
        _, cy = track.center

        # Check 1: Track must NOT be exiting from the bottom (wrong direction)
        if self._is_in_bottom_exit_zone(cy):
            return False

        # Check 2: Track must be exiting from the top (correct direction)
        if not self._is_in_exit_zone(cy):
            return False

        # Check 3: Track must have been visible for minimum duration (ADAPTIVE time-based)
        # Adaptive duration based on entry position
        # If track appeared mid-frame or higher, reduce duration requirement proportionally
        if self.frame_height and track.entry_center_y is not None:
            # Calculate where track started as a ratio (0=top, 1=bottom)
            entry_ratio = track.entry_center_y / self.frame_height

            # If track appeared in the upper half (entry_ratio < 0.5), it has less distance to travel
            # Scale the minimum duration requirement proportionally
            # Bottom entry (ratio=1.0): 100% of min duration required
            # Mid entry (ratio=0.5): 50% of min duration required
            # Top entry (ratio=0.3): 30% of min duration required
            duration_scale = max(0.3, entry_ratio)  # Minimum 30% of base duration
            min_travel_seconds = self._min_travel_duration_seconds * duration_scale
        else:
            min_travel_seconds = self._min_travel_duration_seconds

        track_duration = time.time() - track.created_at

        # Return result without debug logging (called every frame)
        return track_duration >= min_travel_seconds

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
            # No existing tracks - try ghost recovery first, then create new
            unmatched_dets = set(range(len(detections)))
            if detections:
                unmatched_dets = self._try_recover_ghosts(detections, unmatched_dets)
            for det_idx in unmatched_dets:
                if detections[det_idx].confidence >= self._min_conf_new_track:
                    self._create_track(detections[det_idx])
            # Update ghosts (expire old ones)
            self._update_ghosts()
            # Classify entry types for tracks that have enough frames
            self._classify_entry_types()
            return list(self.tracks.values())
        
        if not detections:
            # No detections - mark all tracks as missed
            for track in track_list:
                track.mark_missed()
            
            # Check for track completion
            self._check_completed_tracks()
            # Update ghosts (expire old ones)
            self._update_ghosts()
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
        if self._use_second_stage and unmatched_tracks and unmatched_dets:
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

        # Check for merges: unmatched tracks that may have been absorbed by a matched track
        if unmatched_tracks:
            self._check_for_merges(track_list, unmatched_tracks, matches)
        
        # Try ghost recovery for unmatched detections before creating new tracks
        if unmatched_dets:
            remaining_dets = [detections[j] for j in unmatched_dets]
            remaining_indices = list(unmatched_dets)
            recovered_indices = self._try_recover_ghosts(detections, unmatched_dets)
            unmatched_dets = recovered_indices

        # Try to detach shadows if a matched track's bbox shrank and there are unmatched detections
        if unmatched_dets:
            self._detach_shadows(track_list, matches, detections, unmatched_dets)

        # Create new tracks for unmatched detections (with confidence filtering)
        # Only create tracks for high-confidence detections to avoid noise
        for det_idx in unmatched_dets:
            detection = detections[det_idx]
            if detection.confidence >= self._min_conf_new_track:
                self._create_track(detection)

        # Check for completed tracks
        self._check_completed_tracks()

        # Update ghosts (expire old ones)
        self._update_ghosts()

        # Classify entry types for tracks that have enough frames
        self._classify_entry_types()
        
        return list(self.tracks.values())
    
    def _create_track(self, detection: Detection) -> Optional[TrackedObject]:
        """
        Create a new track from detection.

        Returns None if the detection should not create a track (e.g., already in exit zone).
        """
        # Create temporary track to check its center position
        temp_track = TrackedObject(
            track_id=self._next_id,
            bbox=detection.bbox,
            confidence=detection.confidence,
            _velocity_alpha=self._velocity_alpha
        )

        # Check if detection is already in exit zone (top or bottom)
        # Don't create tracks for objects that are about to leave - they won't have time
        # to accumulate enough hits and will just create unnecessary "track_lost" events
        _, cy = temp_track.center

        if self._is_in_exit_zone(cy):
            return None

        if self._is_in_bottom_exit_zone(cy):
            return None

        # Determine initial entry type based on Y position
        entry_type = "bottom_entry"
        if self.frame_height is not None:
            bottom_entry_y = self.frame_height * (1.0 - self._bottom_entry_zone_ratio)
            if cy < bottom_entry_y:
                # Not in bottom zone -> midway_entry (may be reclassified as thrown_entry later)
                entry_type = "midway_entry"

        # Create the actual track
        track = TrackedObject(
            track_id=self._next_id,
            bbox=detection.bbox,
            confidence=detection.confidence,
            _velocity_alpha=self._velocity_alpha,
            entry_type=entry_type,
            _entry_classified=(entry_type == "bottom_entry")  # bottom_entry is final
        )
        track.position_history.append(track.center)
        track.bbox_history.append(track.bbox)
        track.entry_center_y = track.center[1]
        
        self.tracks[self._next_id] = track

        logger.info(
            f"[TRACK_LIFECYCLE] T{self._next_id} CREATED | "
            f"bbox={detection.bbox} center={track.center} conf={detection.confidence:.2f} "
            f"entry_type={entry_type}"
        )

        self._next_id += 1
        
        return track

    def _check_completed_tracks(self):
        """
        Check for tracks that should be marked as completed.
        
        Core counting rule: A bread bag is COUNTED if and only if its track
        exits from the TOP of the frame. Lost tracks are NEVER counted - they
        are moved to the ghost buffer for potential re-association.
        """
        tracks_to_remove = []
        tracks_to_ghost = []
        
        for track_id, track in self.tracks.items():
            should_complete = False
            event_type = 'track_completed'
            
            # Check if track exceeded max age (lost)
            if track.time_since_update > self.config.max_frames_without_detection:
                # Lost tracks are NEVER counted. Move to ghost buffer for possible recovery.
                tracks_to_ghost.append(track_id)
                continue

            # Check if track is exiting frame and has been tracked long enough
            elif (
                track.hits >= self.config.min_track_duration_frames and
                self._is_exiting_frame(track)
            ):
                should_complete = True
                # Validate full travel path (bottom → top)
                if self._has_valid_travel_path(track):
                    event_type = 'track_completed'
                else:
                    event_type = 'track_invalid'
            
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

                # Emit completion event with enriched lifecycle data
                event = TrackEvent(
                    track_id=track_id,
                    event_type=event_type,
                    bbox_history=list(track.bbox_history),
                    position_history=list(track.position_history),
                    total_frames=track.age + track.hits,
                    created_at=track.created_at,
                    ended_at=time.time(),
                    avg_confidence=avg_conf,
                    exit_direction=exit_dir,
                    entry_type=track.entry_type,
                    suspected_duplicate=(track.entry_type == "midway_entry"),
                    ghost_recovery_count=track.ghost_recovery_count,
                    occlusion_events=list(track.occlusion_events),
                    shadow_of=track.shadow_of,
                    shadow_count=len(track.shadow_tracks),
                    merge_events=list(track.merge_events)
                )
                self.completed_tracks.append(event)
                tracks_to_remove.append(track_id)

                # Also emit events for any shadow tracks riding on this track
                if event_type == 'track_completed' and track.shadow_tracks:
                    self._complete_with_shadows(track, exit_dir)

                logger.info(
                    f"[TRACK_LIFECYCLE] T{track_id} COMPLETED | "
                    f"type={event_type} exit={exit_dir} hits={track.hits} missed={track.time_since_update} "
                    f"duration={event.duration_seconds:.2f}s distance={distance:.0f}px {vel_str} "
                    f"positions={len(track.position_history)} avg_conf={avg_conf:.2f} "
                    f"entry_type={track.entry_type} ghosts={track.ghost_recovery_count} shadows={len(track.shadow_tracks)}"
                )

        # Move lost tracks to ghost buffer
        for track_id in tracks_to_ghost:
            self._move_to_ghost(track_id)

        # Remove completed tracks
        for track_id in set(tracks_to_remove):
            if track_id in self.tracks:
                del self.tracks[track_id]

    # =========================================================================
    # Ghost Track Recovery (Occlusion Handling)
    # =========================================================================

    def _move_to_ghost(self, track_id: int):
        """
        Move a lost track to the ghost buffer for potential re-association.
        
        Ghost tracks are held for up to ghost_track_max_age_seconds before
        being finalized as track_lost.
        """
        track = self.tracks.pop(track_id, None)
        if track is None:
            return

        # Predict where the ghost should be
        predicted_pos = self._predict_ghost_position(track)

        self.ghost_tracks[track_id] = {
            'track': track,
            'lost_at': time.time(),
            'predicted_pos': predicted_pos,
            'last_velocity': track.velocity
        }

        logger.info(
            f"[TRACK_LIFECYCLE] T{track_id} GHOST | "
            f"moved to ghost buffer, predicted_pos={predicted_pos}, "
            f"will expire in {self._ghost_max_age}s"
        )

    def _predict_ghost_position(self, track: TrackedObject) -> Tuple[int, int]:
        """Return the last real observed position as the ghost's reference point.

        Uses position_history[-1] (last actual detection), NOT track.center
        which has been shifted by mark_missed() velocity predictions and would
        cause massive overshoot when combined with further velocity projection
        in _try_recover_ghosts().
        """
        if track.position_history:
            return track.position_history[-1]
        return track.center

    def _try_recover_ghosts(
        self,
        detections: List[Detection],
        unmatched_dets: Set[int]
    ) -> Set[int]:
        """
        Try to re-associate unmatched detections with ghost tracks.
        
        For each unmatched detection, check if it matches a ghost:
        - X-axis within tolerance (bags don't move sideways)
        - Y-axis within max gap and moved toward top
        
        Returns the remaining set of unmatched detection indices.
        """
        if not self.ghost_tracks or not unmatched_dets:
            return unmatched_dets

        max_y_gap = int(self._ghost_max_y_gap_ratio * (self.frame_height or 1000))
        remaining_dets = set(unmatched_dets)
        ghosts_recovered = []

        for ghost_id, ghost_info in list(self.ghost_tracks.items()):
            ghost_track = ghost_info['track']
            ghost_vel = ghost_info['last_velocity']
            lost_at = ghost_info['lost_at']

            # Predict current position from last real observation + elapsed velocity
            # predicted_pos is the last real observed position (from position_history)
            target_fps = getattr(self.config, 'target_fps', 25.0)
            elapsed_seconds = time.time() - lost_at
            elapsed_frames = max(1, int(elapsed_seconds * target_fps))
            # Cap velocity projection to avoid overshoot (max 1 second of projection)
            max_projection_frames = int(target_fps)
            projection_frames = min(elapsed_frames, max_projection_frames)
            pred_x, pred_y = ghost_info['predicted_pos']
            if ghost_vel is not None:
                pred_x = int(pred_x + ghost_vel[0] * projection_frames)
                pred_y = int(pred_y + ghost_vel[1] * projection_frames)

            logger.debug(
                f"[GHOST_RECOVERY] T{ghost_id} checking | "
                f"stored_pos={ghost_info['predicted_pos']}, vel={ghost_vel}, "
                f"elapsed={elapsed_seconds:.2f}s, proj_frames={projection_frames}, "
                f"pred_pos=({pred_x}, {pred_y}), max_y_gap={max_y_gap}"
            )

            best_det_idx = None
            best_dist = float('inf')

            for det_idx in remaining_dets:
                det = detections[det_idx]
                det_cx = (det.bbox[0] + det.bbox[2]) // 2
                det_cy = (det.bbox[1] + det.bbox[3]) // 2

                # X-axis tolerance check
                x_diff = abs(det_cx - pred_x)
                if x_diff > self._ghost_x_tolerance:
                    logger.debug(
                        f"[GHOST_RECOVERY] T{ghost_id} reject det_idx={det_idx} | "
                        f"x_diff={x_diff} > x_tol={self._ghost_x_tolerance}"
                    )
                    continue

                # Y-axis check: detection should be at or above ghost's predicted Y
                y_diff = pred_y - det_cy  # positive means det moved toward top
                if y_diff < -max_y_gap:  # Detection is too far below predicted position
                    logger.debug(
                        f"[GHOST_RECOVERY] T{ghost_id} reject det_idx={det_idx} | "
                        f"det below ghost: y_diff={y_diff} < -max_y_gap={-max_y_gap}"
                    )
                    continue
                if abs(det_cy - pred_y) > max_y_gap:
                    logger.debug(
                        f"[GHOST_RECOVERY] T{ghost_id} reject det_idx={det_idx} | "
                        f"y_gap={abs(det_cy - pred_y)} > max_y_gap={max_y_gap} "
                        f"(det_cy={det_cy}, pred_y={pred_y}, vel={ghost_vel})"
                    )
                    continue

                # Score by distance
                dist = math.sqrt(x_diff**2 + (det_cy - pred_y)**2)
                if dist < best_dist:
                    best_dist = dist
                    best_det_idx = det_idx

            if best_det_idx is not None:
                # Re-associate: restore ghost as active track
                det = detections[best_det_idx]
                ghost_track.update(det)
                ghost_track.ghost_recovery_count += 1
                ghost_track.occlusion_events.append({
                    'lost_at_y': ghost_info['predicted_pos'][1],
                    'recovered_at_y': (det.bbox[1] + det.bbox[3]) // 2,
                    'gap_seconds': round(time.time() - lost_at, 2)
                })

                self.tracks[ghost_id] = ghost_track
                ghosts_recovered.append(ghost_id)
                remaining_dets.discard(best_det_idx)

                logger.info(
                    f"[TRACK_LIFECYCLE] T{ghost_id} RECOVERED | "
                    f"ghost re-associated after {time.time() - lost_at:.1f}s, "
                    f"recovery_count={ghost_track.ghost_recovery_count}"
                )

        # Remove recovered ghosts
        for gid in ghosts_recovered:
            del self.ghost_tracks[gid]

        return remaining_dets

    def _update_ghosts(self):
        """Expire ghost tracks that exceeded max age."""
        now = time.time()
        expired = []

        for ghost_id, ghost_info in self.ghost_tracks.items():
            elapsed = now - ghost_info['lost_at']
            if elapsed >= self._ghost_max_age:
                expired.append(ghost_id)

        for ghost_id in expired:
            ghost_info = self.ghost_tracks.pop(ghost_id)
            track = ghost_info['track']

            # Finalize as track_lost (not counted)
            event = TrackEvent(
                track_id=ghost_id,
                event_type='track_lost',
                bbox_history=list(track.bbox_history),
                position_history=list(track.position_history),
                total_frames=track.age + track.hits,
                created_at=track.created_at,
                ended_at=now,
                avg_confidence=track.confidence,
                exit_direction="timeout",
                entry_type=track.entry_type,
                suspected_duplicate=(track.entry_type == "midway_entry"),
                ghost_recovery_count=track.ghost_recovery_count,
                occlusion_events=list(track.occlusion_events),
                shadow_of=track.shadow_of,
                shadow_count=0,
                merge_events=list(track.merge_events)
            )
            self.completed_tracks.append(event)

            logger.info(
                f"[TRACK_LIFECYCLE] T{ghost_id} GHOST_EXPIRED | "
                f"ghost expired after {self._ghost_max_age}s, finalized as track_lost"
            )

    # =========================================================================
    # Shadow Track / Merge Detection
    # =========================================================================

    def _check_for_merges(
        self,
        track_list: List[TrackedObject],
        unmatched_track_indices: Set[int],
        matches: List[Tuple[int, int]]
    ):
        """
        Check if unmatched tracks were absorbed (merged) by a matched track.

        Merge conditions:
        1. Matched track's bbox grew significantly (>= merge_bbox_growth_threshold)
        2. The unmatched and matched tracks were spatially adjacent
        3. Both tracks were moving in the same direction (toward top)
        """
        if not matches or not unmatched_track_indices:
            return

        matched_tracks = [(track_list[ti], ti) for ti, _ in matches]

        for unmatched_idx in list(unmatched_track_indices):
            lost_track = track_list[unmatched_idx]

            # Only consider tracks with enough history
            if len(lost_track.bbox_history) < 3:
                continue

            lost_cx, lost_cy = lost_track.center

            for survivor, survivor_idx in matched_tracks:
                # Check 1: Bbox growth
                if len(survivor.bbox_history) < 4:
                    continue

                current_width = survivor.bbox[2] - survivor.bbox[0]
                # Average width over recent history (excluding current)
                recent_widths = [
                    b[2] - b[0] for b in list(survivor.bbox_history)[-5:-1]
                ]
                if not recent_widths:
                    continue
                avg_width = sum(recent_widths) / len(recent_widths)

                if avg_width <= 0:
                    continue
                growth_ratio = current_width / avg_width
                if growth_ratio < self._merge_bbox_growth:
                    continue

                # Check 2: Spatial adjacency (previous frame)
                surv_cx, surv_cy = survivor.center
                x_gap = abs(surv_cx - lost_cx)
                y_gap = abs(surv_cy - lost_cy)

                if x_gap > self._merge_spatial_tol or y_gap > self._merge_y_tol:
                    continue

                # Check 3: Same direction (both moving toward top = negative vy)
                lost_vel = lost_track.velocity
                surv_vel = survivor.velocity
                if lost_vel is not None and surv_vel is not None:
                    if lost_vel[1] > 5 or surv_vel[1] > 5:  # Moving downward significantly
                        continue

                # All checks passed → attach as shadow
                self._attach_shadow(survivor, lost_track)
                break  # One merge per lost track

    def _attach_shadow(self, survivor: TrackedObject, shadow: TrackedObject):
        """Attach a shadow track to a surviving track."""
        shadow_id = shadow.track_id
        survivor.shadow_tracks[shadow_id] = shadow
        shadow.shadow_of = survivor.track_id

        survivor.merge_events.append({
            'merged_track_id': shadow_id,
            'merge_y': shadow.center[1],
            'unmerge_y': None
        })

        # Remove shadow from active tracks
        if shadow_id in self.tracks:
            del self.tracks[shadow_id]

        logger.info(
            f"[TRACK_LIFECYCLE] T{shadow_id} SHADOW | "
            f"merged into T{survivor.track_id}, total_shadows={len(survivor.shadow_tracks)}"
        )

    def _detach_shadows(
        self,
        track_list: List[TrackedObject],
        matches: List[Tuple[int, int]],
        detections: List[Detection],
        unmatched_dets: Set[int]
    ):
        """
        Detach shadows if the survivor's bbox shrank and an unmatched detection appeared nearby.
        """
        if not unmatched_dets:
            return

        for track_idx, _ in matches:
            track = track_list[track_idx]
            if not track.shadow_tracks:
                continue

            # Check if bbox shrank back to normal
            if len(track.bbox_history) < 4:
                continue

            current_width = track.bbox[2] - track.bbox[0]
            recent_widths = [b[2] - b[0] for b in list(track.bbox_history)[-5:-1]]
            if not recent_widths:
                continue
            avg_width = sum(recent_widths) / len(recent_widths)
            if avg_width <= 0:
                continue

            # If bbox is still near its merged (grown) width, no un-merge yet
            # Below 90% of average means the bbox has shrunk back, suggesting separation
            if current_width / avg_width > 0.9:
                continue

            # Try to detach shadows
            tcx, tcy = track.center
            for det_idx in list(unmatched_dets):
                det = detections[det_idx]
                det_cx = (det.bbox[0] + det.bbox[2]) // 2
                det_cy = (det.bbox[1] + det.bbox[3]) // 2

                if abs(det_cx - tcx) < self._merge_spatial_tol and abs(det_cy - tcy) < self._merge_y_tol:
                    # Detach first shadow
                    if track.shadow_tracks:
                        shadow_id = next(iter(track.shadow_tracks))
                        shadow = track.shadow_tracks.pop(shadow_id)

                        # Restore as active track
                        shadow.update(det)
                        shadow.shadow_of = None
                        self.tracks[shadow_id] = shadow
                        unmatched_dets.discard(det_idx)

                        # Update merge events
                        for me in track.merge_events:
                            if me['merged_track_id'] == shadow_id and me['unmerge_y'] is None:
                                me['unmerge_y'] = det_cy
                                break

                        logger.info(
                            f"[TRACK_LIFECYCLE] T{shadow_id} SHADOW_DETACHED | "
                            f"detached from T{track.track_id}"
                        )
                        break

    def _complete_with_shadows(self, track: TrackedObject, exit_dir: str):
        """Emit completion events for all shadow tracks when their survivor exits top."""
        for shadow_id, shadow in track.shadow_tracks.items():
            event = TrackEvent(
                track_id=shadow_id,
                event_type='track_completed',
                bbox_history=list(shadow.bbox_history),
                position_history=list(shadow.position_history),
                total_frames=shadow.age + shadow.hits,
                created_at=shadow.created_at,
                ended_at=time.time(),
                avg_confidence=shadow.confidence,
                exit_direction=exit_dir,
                entry_type=shadow.entry_type,
                suspected_duplicate=(shadow.entry_type == "midway_entry"),
                ghost_recovery_count=shadow.ghost_recovery_count,
                occlusion_events=list(shadow.occlusion_events),
                shadow_of=track.track_id,
                shadow_count=0,
                merge_events=list(shadow.merge_events)
            )
            self.completed_tracks.append(event)

            logger.info(
                f"[TRACK_LIFECYCLE] T{shadow_id} SHADOW_COMPLETED | "
                f"shadow of T{track.track_id} counted on exit"
            )

    # =========================================================================
    # Entry Type Classification (Diagnostics Only)
    # =========================================================================

    def _classify_entry_types(self):
        """
        Classify entry type for tracks that have accumulated enough frames.
        
        Tracks created mid-frame start as 'midway_entry' and can be reclassified
        to 'thrown_entry' if their initial velocity exceeds the threshold.
        """
        for track in self.tracks.values():
            if track._entry_classified:
                continue

            if track.hits < self._thrown_entry_detection_frames:
                continue

            # Only reclassify midway_entry tracks
            if track.entry_type != "midway_entry":
                track._entry_classified = True
                continue

            # Measure initial velocity
            vel = track.velocity
            if vel is not None:
                vel_magnitude = math.sqrt(vel[0]**2 + vel[1]**2)
                if vel_magnitude >= self._thrown_entry_min_velocity:
                    track.entry_type = "thrown_entry"

            track._entry_classified = True

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
        self.ghost_tracks.clear()
        logger.info("[ConveyorTracker] Cleanup complete")
