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
    # This is used for velocity calculation and prediction - kept small for performance
    position_history: deque = field(default_factory=lambda: deque(maxlen=30))
    bbox_history: deque = field(default_factory=lambda: deque(maxlen=30))

    # Checkpoint positions for DB storage (saved every N frames)
    # This is the sparse history that gets stored in the database
    checkpoint_positions: list = field(default_factory=list)
    _last_checkpoint_frame: int = 0
    _checkpoint_interval: int = 10  # Save position every N frames

    # Smoothed velocity (exponential moving average)
    _smooth_velocity: Optional[Tuple[float, float]] = None
    _velocity_alpha: float = 0.3  # EMA smoothing factor (0-1, higher = more responsive)

    # Timestamps
    created_at: float = field(default_factory=time.time)
    last_seen_at: float = field(default_factory=time.time)
    
    # Last detected position (for ghost recovery - guaranteed to be from a detection, not drift)
    last_detected_position: Optional[Tuple[int, int]] = None

    # Classification tracking
    classified: bool = False

    # Travel path validation
    entry_center_y: Optional[int] = None  # Y position when track was first created
    entry_center: Optional[Tuple[int, int]] = None  # Full (x, y) position when track was created

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

    # Concurrent/overlapping track tracking (for ghost exit validation)
    # Records IDs of other tracks that were spatially close during this track's lifetime.
    # Used as evidence that detection loss was due to overlap (not the bag disappearing).
    concurrent_track_ids: set = field(default_factory=set)

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
        
        # Store the last detected position (guaranteed to be from actual detection)
        self.last_detected_position = self.center

        # Update history (deque automatically maintains maxlen)
        self.position_history.append(self.center)
        self.bbox_history.append(self.bbox)

        # Save checkpoint position every N frames for DB storage
        total_frames = self.age + self.hits
        if total_frames - self._last_checkpoint_frame >= self._checkpoint_interval:
            self.checkpoint_positions.append(self.center)
            self._last_checkpoint_frame = total_frames

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
        """Mark frame without detection.

        ROBUST VELOCITY HANDLING:
        - For individual track drift during active tracking, require minimum hits
        - Clamps velocity to reasonable conveyor speed limits
        - Decays velocity confidence over missed frames
        - Stops drifting after maximum missed frames

        Note: Ghost prediction uses _get_conveyor_velocity() which is more reliable
        than individual track velocity.
        """
        self.time_since_update += 1
        self.age += 1

        # Constants for robust velocity handling during active tracking
        # Lower threshold since we use conveyor velocity for ghost recovery
        MIN_HITS_FOR_VELOCITY = 4  # Need 4+ detections for individual track drift
        MAX_VEL_X = 15.0  # Max horizontal pixels/frame (conveyor mostly vertical)
        MAX_VEL_Y = 50.0  # Max vertical pixels/frame
        VELOCITY_DECAY = 0.9  # Decay velocity each missed frame
        MAX_DRIFT_FRAMES = 10  # Stop drifting after this many frames

        vel = self.velocity
        if vel is None or len(self.position_history) == 0:
            return

        # Check if we have enough history to trust velocity for drift
        if self.hits < MIN_HITS_FOR_VELOCITY:
            if self.time_since_update == 1:
                logger.info(
                    f"[TRACK_DRIFT] T{self.track_id} NOT drifting | "
                    f"hits={self.hits} < min={MIN_HITS_FOR_VELOCITY}, velocity not trusted"
                )
            return

        # Stop drifting after too many missed frames
        if self.time_since_update > MAX_DRIFT_FRAMES:
            if self.time_since_update == MAX_DRIFT_FRAMES + 1:
                logger.info(
                    f"[TRACK_DRIFT] T{self.track_id} STOPPED drifting | "
                    f"missed={self.time_since_update} > max={MAX_DRIFT_FRAMES}"
                )
            return

        # Clamp velocity to reasonable limits
        vx = max(-MAX_VEL_X, min(MAX_VEL_X, vel[0]))
        vy = max(-MAX_VEL_Y, min(MAX_VEL_Y, vel[1]))

        # Apply decay based on how many frames missed
        decay_factor = VELOCITY_DECAY ** self.time_since_update
        vx *= decay_factor
        vy *= decay_factor

        # Apply velocity to bbox
        old_bbox = self.bbox
        x1, y1, x2, y2 = self.bbox
        self.bbox = (
            int(x1 + vx),
            int(y1 + vy),
            int(x2 + vx),
            int(y2 + vy)
        )

        # Log drift for debugging
        if self.time_since_update <= 5 or self.time_since_update % 10 == 0:
            logger.info(
                f"[TRACK_DRIFT] T{self.track_id} frame #{self.time_since_update} | "
                f"raw_vel=({vel[0]:.1f}, {vel[1]:.1f}) clamped=({vx:.1f}, {vy:.1f}) | "
                f"center: {self.center}"
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

    # Ghost exit validation fields
    ghost_exit_promoted: bool = False  # True if promoted from track_lost to track_completed
    concurrent_track_count: int = 0  # Number of distinct tracks that overlapped during lifetime

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
        self._last_conveyor_velocity: Optional[Tuple[float, float]] = None  # Historical velocity estimate

        # Ghost exit validation config (promote near-top ghosts to completed)
        self._ghost_exit_validation_enabled = getattr(self.config, 'ghost_exit_validation_enabled', True)
        self._ghost_exit_near_top_ratio = getattr(self.config, 'ghost_exit_near_top_ratio', 0.35)
        self._ghost_exit_min_travel_ratio = getattr(self.config, 'ghost_exit_min_travel_ratio', 0.40)
        self._ghost_exit_min_hits = getattr(self.config, 'ghost_exit_min_hits', 5)
        self._ghost_exit_predicted_top_ratio = getattr(self.config, 'ghost_exit_predicted_top_ratio', 0.20)

        # Learned conveyor velocity (FPS-independent, stored in pixels/second)
        # This is learned from completed tracks that traveled from center to top
        self._learned_velocity_px_per_sec: Optional[float] = None  # Y velocity in pixels/second
        self._velocity_samples: List[float] = []  # Recent velocity samples (px/sec)
        self._max_velocity_samples = 20  # Keep last N samples for averaging

        # Shadow/merge config
        self._merge_bbox_growth = getattr(self.config, 'merge_bbox_growth_threshold', 1.4)
        self._merge_spatial_tol = getattr(self.config, 'merge_spatial_tolerance_pixels', 50.0)
        self._merge_y_tol = getattr(self.config, 'merge_y_tolerance_pixels', 30.0)

        # Entry type config
        self._bottom_entry_zone_ratio = getattr(self.config, 'bottom_entry_zone_ratio', 0.4)
        self._thrown_entry_min_velocity = getattr(self.config, 'thrown_entry_min_velocity', 15.0)
        self._thrown_entry_detection_frames = getattr(self.config, 'thrown_entry_detection_frames', 5)

        # Statistics counters
        self._duplicates_prevented = 0  # Count of prevented duplicate tracks
        self._tracks_created = 0  # Total tracks actually created
        self._ghost_exits_promoted = 0  # Ghosts that would have exited top (diagnostic)

        # Log configuration
        multi_criteria = getattr(self.config, 'use_multi_criteria_matching', True)

        logger.info(
            f"[ConveyorTracker] Initialized: iou_threshold={self.config.iou_threshold}, "
            f"max_age={self.config.max_frames_without_detection}, min_hits={self.config.min_track_duration_frames}, "
            f"min_conf_new_track={self._min_conf_new_track:.2f}, "
            f"multi_criteria={multi_criteria}, second_stage={self._use_second_stage}, "
            f"ghost_max_age={self._ghost_max_age}s, merge_growth={self._merge_bbox_growth}, "
            f"ghost_exit_validation={self._ghost_exit_validation_enabled}"
        )

    def get_ghost_tracks_for_visualization(self) -> List[Dict]:
        """
        Get ghost track data for visualization.

        Returns a list of ghost track info with predicted positions based on
        conveyor velocity, so the UI can show where the ghost is expected to be.

        Returns:
            List of dicts with: track_id, predicted_bbox, original_pos, elapsed_seconds, velocity
        """
        if not self.ghost_tracks:
            return []

        result = []
        conveyor_vel = self._get_conveyor_velocity()
        target_fps = getattr(self.config, 'target_fps', 17.0)
        now = time.time()

        for ghost_id, ghost_info in self.ghost_tracks.items():
            track = ghost_info['track']
            original_pos = ghost_info['predicted_pos']
            lost_at = ghost_info['lost_at']

            # Calculate elapsed time and frames
            elapsed_seconds = now - lost_at
            elapsed_frames = max(1, int(elapsed_seconds * target_fps))

            # Cap projection to avoid overshooting (max 2 seconds)
            max_projection_frames = int(target_fps * 2)
            projection_frames = min(elapsed_frames, max_projection_frames)

            # Predict current position using conveyor velocity
            pred_x = int(original_pos[0] + conveyor_vel[0] * projection_frames)
            pred_y = int(original_pos[1] + conveyor_vel[1] * projection_frames)

            # Get original bbox dimensions to create predicted bbox
            orig_bbox = track.bbox
            width = orig_bbox[2] - orig_bbox[0]
            height = orig_bbox[3] - orig_bbox[1]

            # Create predicted bbox centered at predicted position
            pred_bbox = (
                pred_x - width // 2,
                pred_y - height // 2,
                pred_x + width // 2,
                pred_y + height // 2
            )

            result.append({
                'track_id': ghost_id,
                'predicted_bbox': pred_bbox,
                'original_pos': original_pos,
                'predicted_pos': (pred_x, pred_y),
                'elapsed_seconds': elapsed_seconds,
                'elapsed_frames': elapsed_frames,
                'projection_frames': projection_frames,
                'velocity': conveyor_vel,
                'hits': track.hits,
                'learned_velocity_px_sec': self._learned_velocity_px_per_sec,
                'velocity_source': 'learned' if self._learned_velocity_px_per_sec else 'default',
                'missed_frames_before_ghost': ghost_info.get('missed_frames_before_ghost', 0)
            })

        return result

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

        # Update concurrent track proximity (for ghost exit validation evidence)
        self._update_concurrent_tracks()

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
        cx, cy = temp_track.center

        if self._is_in_exit_zone(cy):
            return None

        if self._is_in_bottom_exit_zone(cy):
            return None

        # DUPLICATE PREVENTION: Don't create new track if an existing track is very close
        # This prevents detector glitches from creating multiple tracks for the same object
        # when two bags overlap and detector reports multiple detections
        #
        # Check against:
        # 1. Active tracks
        # 2. Shadow tracks (attached to active tracks)
        # 3. Ghost tracks (recently lost, may recover)

        # Collect all positions to check against
        all_existing_positions = []

        # Active tracks
        for existing_track in self.tracks.values():
            all_existing_positions.append((existing_track.track_id, existing_track.center, "active"))
            # Also check shadows attached to this track
            for shadow_id, shadow in existing_track.shadow_tracks.items():
                # Use predicted position for shadow based on conveyor velocity
                shadow_pos = shadow.last_detected_position or shadow.center
                all_existing_positions.append((shadow_id, shadow_pos, "shadow"))

        # Ghost tracks (with predicted positions)
        conveyor_vel = self._get_conveyor_velocity()
        target_fps = getattr(self.config, 'target_fps', 17.0)
        now = time.time()
        for ghost_id, ghost_info in self.ghost_tracks.items():
            original_pos = ghost_info['predicted_pos']
            lost_at = ghost_info['lost_at']
            elapsed_seconds = now - lost_at
            elapsed_frames = int(elapsed_seconds * target_fps)
            # Predict where ghost should be now
            pred_x = int(original_pos[0] + conveyor_vel[0] * elapsed_frames)
            pred_y = int(original_pos[1] + conveyor_vel[1] * elapsed_frames)
            all_existing_positions.append((ghost_id, (pred_x, pred_y), "ghost"))

        # Check if new detection is too close to any existing position
        # Use slightly larger tolerance than merge to catch edge cases
        dup_x_tol = self._merge_spatial_tol * 1.5  # 75 pixels typically
        dup_y_tol = self._merge_y_tol * 2.0  # 60 pixels typically (bags can be 30-50px apart vertically)

        for track_id, (ex_cx, ex_cy), track_type in all_existing_positions:
            x_gap = abs(cx - ex_cx)
            y_gap = abs(cy - ex_cy)

            # If very close (within duplicate tolerance), don't create new track
            if x_gap <= dup_x_tol and y_gap <= dup_y_tol:
                self._duplicates_prevented += 1
                logger.info(
                    f"[TRACK_LIFECYCLE] DUPLICATE_PREVENTED | "
                    f"detection at ({cx}, {cy}) too close to T{track_id} ({track_type}) "
                    f"at ({ex_cx}, {ex_cy}), gap=({x_gap}, {y_gap}), "
                    f"total_prevented={self._duplicates_prevented}"
                )
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
        track.entry_center = track.center  # Store full entry position
        track.checkpoint_positions.append(track.center)  # First checkpoint = entry point
        track.last_detected_position = track.center  # Initialize last detected position

        self.tracks[self._next_id] = track
        self._tracks_created += 1

        logger.info(
            f"[TRACK_LIFECYCLE] T{self._next_id} CREATED | "
            f"bbox={detection.bbox} center={track.center} conf={detection.confidence:.2f} "
            f"entry_type={entry_type}, total_created={self._tracks_created}"
        )

        self._next_id += 1
        
        return track

    def _update_concurrent_tracks(self):
        """
        Record which active tracks are spatially close to each other.

        For each pair of active tracks, if their bounding boxes overlap or
        are nearly touching (within a small margin), record each as concurrent
        with the other.  This is lightweight — O(N^2) for N active tracks, but
        N is typically ≤ 5 on a conveyor.

        Used by ghost exit validation: a track that was never near another
        track should not be promoted (detection loss from overlap is the
        primary failure mode we are recovering from).
        """
        track_list = list(self.tracks.values())
        n = len(track_list)
        if n < 2:
            return

        # Use a generous bbox-edge margin: if the horizontal gap between
        # bbox edges is ≤ margin, they are "near". With bags on conveyor
        # this catches side-by-side and overlapping scenarios.
        bbox_edge_margin = self._merge_spatial_tol  # 50px default
        y_margin = self._merge_y_tol * 4  # 120px — generous vertical tolerance

        for i in range(n):
            t_a = track_list[i]
            ax1, ay1, ax2, ay2 = t_a.bbox
            for j in range(i + 1, n):
                t_b = track_list[j]
                bx1, by1, bx2, by2 = t_b.bbox

                # Horizontal gap between bbox edges (negative = overlap)
                x_gap = max(0, max(ax1, bx1) - min(ax2, bx2))
                # Vertical gap between bbox edges
                y_gap = max(0, max(ay1, by1) - min(ay2, by2))

                if x_gap <= bbox_edge_margin and y_gap <= y_margin:
                    t_a.concurrent_track_ids.add(t_b.track_id)
                    t_b.concurrent_track_ids.add(t_a.track_id)

    def _check_completed_tracks(self):
        """
        Check for tracks that should be marked as completed.
        
        Core counting rule: A bread bag is COUNTED if and only if its track
        exits from the TOP of the frame. Lost tracks are NEVER counted - they
        are moved to the ghost buffer for potential re-association.

        ORDERING IS IMPORTANT:
        1. First identify and move LOST tracks to ghost buffer
        2. Then process EXITING tracks (completions)

        This ordering ensures that when a track exits top, any concurrent
        lost tracks are already in the ghost buffer and can be found by
        _complete_ghost_companions().  Example: T304 exits top in the same
        frame that T305 exceeds max_frames_without_detection — T305 must
        be in the ghost buffer before T304's companion check runs.
        """
        tracks_to_ghost = []
        tracks_to_complete = []  # (track_id, track)

        for track_id, track in self.tracks.items():
            # Check if track exceeded max age (lost)
            if track.time_since_update > self.config.max_frames_without_detection:
                tracks_to_ghost.append(track_id)
            # Check if track is exiting frame and has been tracked long enough
            elif (
                track.hits >= self.config.min_track_duration_frames and
                self._is_exiting_frame(track)
            ):
                tracks_to_complete.append((track_id, track))

        # STEP 1: Move lost tracks to ghost buffer FIRST
        # This ensures they are available for companion completion in step 2.
        for track_id in tracks_to_ghost:
            self._move_to_ghost(track_id)

        # STEP 2: Process exiting tracks (completions)
        tracks_to_remove = []
        for track_id, track in tracks_to_complete:
            # Validate full travel path (bottom → top)
            if self._has_valid_travel_path(track):
                event_type = 'track_completed'
            else:
                event_type = 'track_invalid'

            # Calculate average confidence
            avg_conf = track.confidence  # Use last confidence for now

            # Get exit direction
            exit_dir = self._get_exit_direction(track) or "timeout"

            # Calculate track statistics
            velocity = track.velocity
            vel_str = f"vel=({velocity[0]:.1f},{velocity[1]:.1f})" if velocity else "vel=None"

            # Build position history from checkpoints for DB storage
            # Checkpoints are saved every N frames, plus entry and exit
            final_position_history = list(track.checkpoint_positions)

            # Always include the final/exit position if different from last checkpoint
            exit_position = track.center
            if not final_position_history or final_position_history[-1] != exit_position:
                final_position_history.append(exit_position)

            # Calculate distance using entry and exit points
            distance = 0.0
            if len(final_position_history) >= 2:
                start = final_position_history[0]
                end = final_position_history[-1]
                distance = ((end[0]-start[0])**2 + (end[1]-start[1])**2)**0.5

            # Emit completion event with enriched lifecycle data
            event = TrackEvent(
                track_id=track_id,
                event_type=event_type,
                bbox_history=list(track.bbox_history),
                position_history=final_position_history,
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
                merge_events=list(track.merge_events),
                concurrent_track_count=len(track.concurrent_track_ids),
            )
            self.completed_tracks.append(event)
            tracks_to_remove.append(track_id)

            # Also emit events for any shadow tracks riding on this track
            if event_type == 'track_completed' and track.shadow_tracks:
                self._complete_with_shadows(track, exit_dir)

            # GHOST COMPANION COMPLETION:
            # When this track exits top, check the ghost buffer for concurrent
            # companions — tracks that were traveling alongside this one but
            # lost detection due to overlap. The survivor's successful exit
            # is hard evidence that companions also reached the destination.
            if event_type == 'track_completed' and exit_dir == 'top':
                self._complete_ghost_companions(track)

            # Learn conveyor velocity from this completed track
            if event_type == 'track_completed' and exit_dir == 'top':
                self._learn_conveyor_velocity(track, event.duration_seconds)

            logger.info(
                f"[TRACK_LIFECYCLE] T{track_id} COMPLETED | "
                f"type={event_type} exit={exit_dir} hits={track.hits} missed={track.time_since_update} "
                f"duration={event.duration_seconds:.2f}s distance={distance:.0f}px {vel_str} "
                f"positions={len(track.position_history)} avg_conf={avg_conf:.2f} "
                f"entry_type={track.entry_type} ghosts={track.ghost_recovery_count} shadows={len(track.shadow_tracks)}"
            )

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

        # Predict where the ghost should be (uses last_detected_position)
        predicted_pos = self._predict_ghost_position(track)

        # Log discrepancy between detected position and drifted center for debugging
        drifted_center = track.center
        last_detected = track.last_detected_position or predicted_pos

        # IMPORTANT: Use track.last_seen_at (when last detected) NOT time.time()
        # This accounts for the max_frames_without_detection waiting period.
        # If we use time.time(), we'd miss the distance the object traveled
        # during the waiting frames before being declared lost.
        last_detection_time = track.last_seen_at
        missed_frames = track.time_since_update

        self.ghost_tracks[track_id] = {
            'track': track,
            'lost_at': last_detection_time,  # When we LAST SAW it, not now
            'predicted_pos': predicted_pos,
            'last_velocity': track.velocity,
            'missed_frames_before_ghost': missed_frames  # For debugging
        }

        logger.info(
            f"[TRACK_LIFECYCLE] T{track_id} GHOST | "
            f"last_detected={last_detected}, drifted_to={drifted_center}, "
            f"using_pos={predicted_pos}, missed_frames={missed_frames}, "
            f"will expire in {self._ghost_max_age}s"
        )

    def _learn_conveyor_velocity(self, track: TrackedObject, duration_seconds: float):
        """
        Learn conveyor velocity from a successfully completed track.

        Uses the center-to-exit portion of the track's journey for stable measurement,
        since the bottom portion may have detection instability.

        Stores velocity in PIXELS PER SECOND (FPS-independent) so it works
        correctly regardless of camera FPS changes.

        Args:
            track: The completed track
            duration_seconds: Total track duration
        """
        if not track.position_history or len(track.position_history) < 5:
            return

        if not self.frame_height or duration_seconds <= 0:
            return

        positions = list(track.position_history)

        # Find positions in the "stable zone" (center to top of frame)
        # Use positions where Y < 60% of frame height (upper portion)
        center_y_threshold = self.frame_height * 0.6  # 432 for 720p
        exit_y_threshold = self.frame_height * 0.15   # 108 for 720p (near top)

        # Filter to positions in the stable center-to-exit zone
        stable_positions = [
            pos for pos in positions
            if exit_y_threshold < pos[1] < center_y_threshold
        ]

        if len(stable_positions) < 3:
            # Not enough positions in stable zone, use full track
            stable_positions = positions

        if len(stable_positions) < 2:
            return

        # Calculate Y displacement in stable zone
        start_pos = stable_positions[0]
        end_pos = stable_positions[-1]

        y_displacement = end_pos[1] - start_pos[1]  # Should be negative (moving up)

        if y_displacement >= 0:
            # Track moved down or didn't move, invalid
            return

        # Estimate time spent in stable zone based on proportion of positions
        stable_proportion = len(stable_positions) / len(positions)
        stable_duration = duration_seconds * stable_proportion

        if stable_duration <= 0.1:
            return

        # Calculate velocity in pixels per second
        vy_px_per_sec = y_displacement / stable_duration

        # Sanity check: velocity should be reasonable
        # At 720p, typical velocity is ~140 px/sec (8.2 px/frame * 17 fps)
        # Allow range of 80-250 px/sec
        if not (-250 < vy_px_per_sec < -80):
            logger.debug(
                f"[VELOCITY_LEARN] T{track.track_id} rejected | "
                f"vy={vy_px_per_sec:.1f} px/sec out of range"
            )
            return

        # Add to samples
        self._velocity_samples.append(vy_px_per_sec)

        # Keep only last N samples
        if len(self._velocity_samples) > self._max_velocity_samples:
            self._velocity_samples = self._velocity_samples[-self._max_velocity_samples:]

        # Update learned velocity (exponential moving average)
        if self._learned_velocity_px_per_sec is None:
            self._learned_velocity_px_per_sec = vy_px_per_sec
        else:
            alpha = 0.3  # Smoothing factor
            self._learned_velocity_px_per_sec = (
                alpha * vy_px_per_sec +
                (1 - alpha) * self._learned_velocity_px_per_sec
            )

        logger.info(
            f"[VELOCITY_LEARN] T{track.track_id} | "
            f"sample={vy_px_per_sec:.1f} px/sec, "
            f"learned={self._learned_velocity_px_per_sec:.1f} px/sec, "
            f"samples={len(self._velocity_samples)}"
        )

    def _predict_ghost_position(self, track: TrackedObject) -> Tuple[int, int]:
        """Return the last real observed position as the ghost's reference point.

        Uses track.last_detected_position which is guaranteed to be set from
        an actual detection (not from drift predictions).

        Falls back to position_history[-1] or track.center if not available.
        """
        # Priority 1: Use dedicated last_detected_position (most reliable)
        if track.last_detected_position is not None:
            return track.last_detected_position

        # Priority 2: Use position_history (should contain only detected positions)
        if track.position_history:
            return track.position_history[-1]

        # Priority 3: Fallback to current center (may be drifted)
        return track.center

    def _get_conveyor_velocity(self) -> Tuple[float, float]:
        """
        Estimate conveyor velocity for ghost prediction.

        Priority order:
        1. Average of active tracks with reliable velocity
        2. Learned velocity from completed tracks (FPS-independent)
        3. Historical estimate from previous frames
        4. Frame-based calculation using learned or default values

        Returns:
            (vx, vy) in PIXELS PER FRAME where vx ≈ 0 and vy < 0 (moving toward top)
        """
        target_fps = getattr(self.config, 'target_fps', 17.0)

        # Priority 1: Use active tracks if available
        velocities = []
        for track in self.tracks.values():
            vel = track.velocity
            # Include tracks with 4+ hits that are moving upward
            if vel is not None and track.hits >= 4 and vel[1] < 0:
                velocities.append(vel)

        if velocities:
            # Average velocity across all contributing tracks
            avg_vx = sum(v[0] for v in velocities) / len(velocities)
            avg_vy = sum(v[1] for v in velocities) / len(velocities)

            # Clamp X to near zero (conveyor moves vertically)
            clamped_vx = max(-2.0, min(2.0, avg_vx))

            # Ensure Y is negative (moving toward top) and reasonable
            clamped_vy = max(-50.0, min(-3.0, avg_vy))

            # Store as historical estimate
            self._last_conveyor_velocity = (clamped_vx, clamped_vy)

            return (clamped_vx, clamped_vy)

        # Priority 2: Use LEARNED velocity from completed tracks (FPS-independent)
        if self._learned_velocity_px_per_sec is not None:
            # Convert from pixels/second to pixels/frame
            vy_per_frame = self._learned_velocity_px_per_sec / target_fps
            return (0.0, vy_per_frame)

        # Priority 3: Use historical estimate if available
        if self._last_conveyor_velocity is not None:
            return self._last_conveyor_velocity

        # Priority 4: Calculate based on frame dimensions
        # Default assumption: ~140 px/sec based on measurements
        # This is FPS-independent
        if self.frame_height:
            # Default: objects travel ~560 pixels in ~4 seconds = 140 px/sec
            default_vy_px_per_sec = -140.0
            vy_per_frame = default_vy_px_per_sec / target_fps
            return (0.0, vy_per_frame)

        # Final fallback: -140 px/sec / 17 fps ≈ -8.2 px/frame
        return (0.0, -140.0 / target_fps)

    def _try_recover_ghosts(
        self,
        detections: List[Detection],
        unmatched_dets: Set[int]
    ) -> Set[int]:
        """
        Try to re-associate unmatched detections with ghost tracks.
        
        CONVEYOR-SPECIFIC PREDICTION:
        - Uses average conveyor velocity from all active tracks (not individual track velocity)
        - X-axis is nearly locked (bread bags don't move sideways)
        - Y-axis predicted based on consistent conveyor speed

        Returns the remaining set of unmatched detection indices.
        """
        if not self.ghost_tracks or not unmatched_dets:
            return unmatched_dets

        max_y_gap = int(self._ghost_max_y_gap_ratio * (self.frame_height or 1000))
        remaining_dets = set(unmatched_dets)
        ghosts_recovered = []

        # Get conveyor velocity (stable estimate from all active tracks)
        conveyor_vel = self._get_conveyor_velocity()

        for ghost_id, ghost_info in list(self.ghost_tracks.items()):
            ghost_track = ghost_info['track']
            individual_vel = ghost_info['last_velocity']
            lost_at = ghost_info['lost_at']
            original_pos = ghost_info['predicted_pos']

            # Predict current position using CONVEYOR velocity (not individual track velocity)
            target_fps = getattr(self.config, 'target_fps', 25.0)
            elapsed_seconds = time.time() - lost_at
            elapsed_frames = max(1, int(elapsed_seconds * target_fps))

            # Cap projection to avoid overshooting (max 2 seconds)
            max_projection_frames = int(target_fps * 2)
            projection_frames = min(elapsed_frames, max_projection_frames)

            # Use conveyor velocity for prediction (X locked, Y toward top)
            pred_x = int(original_pos[0] + conveyor_vel[0] * projection_frames)
            pred_y = int(original_pos[1] + conveyor_vel[1] * projection_frames)

            logger.info(
                f"[GHOST_PREDICT] T{ghost_id} | "
                f"original={original_pos}, elapsed={elapsed_seconds:.2f}s, "
                f"conveyor_vel=({conveyor_vel[0]:.1f}, {conveyor_vel[1]:.1f}), "
                f"individual_vel={f'({individual_vel[0]:.1f}, {individual_vel[1]:.1f})' if individual_vel else 'None'}, "
                f"predicted=({pred_x}, {pred_y})"
            )

            best_det_idx = None
            best_dist = float('inf')

            for det_idx in remaining_dets:
                det = detections[det_idx]
                det_cx = (det.bbox[0] + det.bbox[2]) // 2
                det_cy = (det.bbox[1] + det.bbox[3]) // 2

                # X-axis tolerance check (bags don't move sideways)
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
                        f"(det_cy={det_cy}, pred_y={pred_y})"
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
        """Expire ghost tracks that exceeded max age.

        Note: lost_at is set to track.last_seen_at (when last detected),
        NOT when the track was moved to ghost buffer. This means the elapsed
        time already includes the max_frames_without_detection waiting period.

        GHOST NEAR-TOP DIAGNOSTICS (v2.7):
        When a ghost expires, check if the track would have reached the top
        exit zone based on conveyor velocity prediction. If so, flag it as
        ghost_would_have_exited=True on the track_lost event for monitoring.
        This does NOT change counting — ghosts are ALWAYS track_lost. The
        flag lets us measure how often the detector loses bags near the top
        so we can track detector improvement over time.
        """
        now = time.time()
        expired = []

        for ghost_id, ghost_info in self.ghost_tracks.items():
            # elapsed includes waiting frames + time in ghost buffer
            elapsed = now - ghost_info['lost_at']
            if elapsed >= self._ghost_max_age:
                expired.append(ghost_id)

        for ghost_id in expired:
            ghost_info = self.ghost_tracks.pop(ghost_id)
            track = ghost_info['track']
            original_pos = ghost_info['predicted_pos']  # Last REAL observed position

            # Build position history from checkpoints for DB storage
            final_position_history = list(track.checkpoint_positions)

            # IMPORTANT: Use original_pos (last real detection), NOT track.center
            # track.center may have drifted due to mark_missed() velocity predictions
            last_real_position = original_pos if original_pos else track.center
            if not final_position_history or final_position_history[-1] != last_real_position:
                final_position_history.append(last_real_position)

            # Log the drift for debugging
            drifted_center = track.center
            drift_x = drifted_center[0] - last_real_position[0]
            drift_y = drifted_center[1] - last_real_position[1]

            # DIAGNOSTIC: Check if this ghost *would have* reached the top exit
            # This is for monitoring only — it does NOT change counting.
            ghost_exit_result = self._validate_ghost_as_completed(
                ghost_id, track, ghost_info, last_real_position, now
            )
            would_have_exited = ghost_exit_result is not None

            if would_have_exited:
                self._ghost_exits_promoted += 1  # counter name kept for stats compatibility
                logger.info(
                    f"[TRACK_LIFECYCLE] T{ghost_id} GHOST_NEAR_TOP_UNCOUNTED | "
                    f"would have exited top but NOT counted (diagnostic only) | "
                    f"last_real_pos={last_real_position}, entry_y={track.entry_center_y}, "
                    f"hits={track.hits}, travel_ratio={ghost_exit_result['travel_ratio']:.2f}, "
                    f"concurrent_tracks={ghost_exit_result.get('concurrent_count', 0)}, "
                    f"total_near_top={self._ghost_exits_promoted}"
                )

            # Always emit as track_lost — counting is ONLY at top_exit
            event = TrackEvent(
                track_id=ghost_id,
                event_type='track_lost',
                bbox_history=list(track.bbox_history),
                position_history=final_position_history,
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
                shadow_count=len(track.shadow_tracks),
                merge_events=list(track.merge_events),
                ghost_exit_promoted=would_have_exited,  # diagnostic flag only
                concurrent_track_count=len(track.concurrent_track_ids),
            )
            self.completed_tracks.append(event)

            # If this ghost had shadows, mark them as lost too
            if track.shadow_tracks:
                self._lose_shadows_with_survivor(track, now)

            logger.info(
                f"[TRACK_LIFECYCLE] T{ghost_id} GHOST_EXPIRED | "
                f"expired after {self._ghost_max_age}s, exit_pos={last_real_position}, "
                f"drift_was=({drift_x}, {drift_y}), shadows_lost={len(track.shadow_tracks)}, "
                f"would_have_exited={would_have_exited}"
            )

    def _validate_ghost_as_completed(
        self,
        ghost_id: int,
        track: TrackedObject,
        ghost_info: dict,
        last_real_position: Tuple[int, int],
        now: float
    ) -> Optional[Dict]:
        """
        Validate whether an expiring ghost track should be promoted to track_completed.

        This is a CONSERVATIVE validation to prevent undercounting of bags that lose
        detection near the top of the frame but clearly would have exited. All checks
        must pass for promotion.

        Criteria (ALL must be met):
        1. Ghost exit validation is enabled
        2. Frame dimensions are known
        3. Track is NOT a shadow (shadows are handled by their survivor)
        4. Track entered from bottom zone (not midway/thrown)
        5. Track has minimum detection hits (enough evidence)
        6. Last real detected position is in the upper portion of the frame
        7. Track traveled at least min_travel_ratio of the frame height (bottom→up)
        8. Predicted conveyor position would be in/past the top exit zone

        Args:
            ghost_id: Track ID of the ghost
            track: The TrackedObject
            ghost_info: Ghost buffer info dict
            last_real_position: Last actually-detected position (not drifted)
            now: Current timestamp

        Returns:
            Dict with validation details if promoted, None if rejected.
        """
        # --- Gate 0: Feature enabled? ---
        if not self._ghost_exit_validation_enabled:
            return None

        # --- Gate 1: Frame dimensions known? ---
        if not self.frame_height or not self.frame_width:
            logger.debug(f"[GHOST_EXIT] T{ghost_id} REJECT: frame dimensions unknown")
            return None

        # --- Gate 2: Not a shadow track (shadows are handled by survivor) ---
        if track.shadow_of is not None:
            logger.debug(f"[GHOST_EXIT] T{ghost_id} REJECT: shadow of T{track.shadow_of}")
            return None

        # --- Gate 3: Entry type must be bottom_entry (most reliable path) ---
        if track.entry_type not in ("bottom_entry",):
            logger.debug(
                f"[GHOST_EXIT] T{ghost_id} REJECT: entry_type={track.entry_type} "
                f"(only bottom_entry qualifies)"
            )
            return None

        # --- Gate 4: Minimum detection hits ---
        if track.hits < self._ghost_exit_min_hits:
            logger.debug(
                f"[GHOST_EXIT] T{ghost_id} REJECT: hits={track.hits} < "
                f"min={self._ghost_exit_min_hits}"
            )
            return None

        # --- Gate 5: Last real position must be in upper portion of frame ---
        last_y = last_real_position[1]
        near_top_threshold = int(self.frame_height * self._ghost_exit_near_top_ratio)
        if last_y > near_top_threshold:
            logger.debug(
                f"[GHOST_EXIT] T{ghost_id} REJECT: last_y={last_y} > "
                f"near_top_threshold={near_top_threshold} "
                f"(not close enough to top)"
            )
            return None

        # --- Gate 6: Sufficient vertical travel (bottom → up) ---
        entry_y = track.entry_center_y
        if entry_y is None:
            logger.debug(f"[GHOST_EXIT] T{ghost_id} REJECT: no entry_center_y")
            return None

        vertical_travel = entry_y - last_y  # positive = moved upward
        travel_ratio = vertical_travel / self.frame_height
        if travel_ratio < self._ghost_exit_min_travel_ratio:
            logger.debug(
                f"[GHOST_EXIT] T{ghost_id} REJECT: travel_ratio={travel_ratio:.2f} < "
                f"min={self._ghost_exit_min_travel_ratio} "
                f"(entry_y={entry_y}, last_y={last_y}, traveled={vertical_travel}px)"
            )
            return None

        # --- Gate 7: Predicted conveyor position reaches top exit zone ---
        conveyor_vel = self._get_conveyor_velocity()
        lost_at = ghost_info['lost_at']
        elapsed_seconds = now - lost_at
        target_fps = getattr(self.config, 'target_fps', 17.0)
        elapsed_frames = max(1, int(elapsed_seconds * target_fps))

        # Predict where the bag would be now using conveyor velocity
        pred_y = int(last_real_position[1] + conveyor_vel[1] * elapsed_frames)
        predicted_top_threshold = int(self.frame_height * self._ghost_exit_predicted_top_ratio)

        if pred_y > predicted_top_threshold:
            logger.debug(
                f"[GHOST_EXIT] T{ghost_id} REJECT: predicted_y={pred_y} > "
                f"top_threshold={predicted_top_threshold} "
                f"(conveyor_vel_y={conveyor_vel[1]:.1f}, elapsed_frames={elapsed_frames})"
            )
            return None

        # --- Gate 8: Must have had concurrent/overlapping tracks ---
        # Detection loss near top typically happens due to bag overlap.
        # If the track was always alone (no nearby tracks during its lifetime),
        # detection loss is suspicious and we should NOT promote.
        concurrent_count = len(track.concurrent_track_ids)
        if concurrent_count == 0:
            logger.debug(
                f"[GHOST_EXIT] T{ghost_id} REJECT: no concurrent tracks "
                f"(detection loss without overlap is not eligible)"
            )
            return None

        # --- All gates passed: promote to track_completed ---
        logger.info(
            f"[GHOST_EXIT] T{ghost_id} VALIDATED | "
            f"entry_y={entry_y}, last_y={last_y}, pred_y={pred_y}, "
            f"travel_ratio={travel_ratio:.2f}, hits={track.hits}, "
            f"concurrent_tracks={concurrent_count} ({sorted(track.concurrent_track_ids)}), "
            f"elapsed={elapsed_seconds:.1f}s, "
            f"conveyor_vy={conveyor_vel[1]:.1f} px/frame"
        )

        return {
            'predicted_y': pred_y,
            'travel_ratio': travel_ratio,
            'elapsed_seconds': elapsed_seconds,
            'concurrent_count': concurrent_count,
        }

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
        Detect when tracks merge due to occlusion/overlap.

        IMPROVED LOGIC:
        - Primary: Position proximity (lost track was near a surviving track)
        - Secondary: Bbox growth (surviving track grew, indicating merged detection)
        - Both tracks must be moving in same direction (toward top)

        When two bread bags overlap on the conveyor:
        1. Detector may see them as one large detection
        2. One track loses its match (becomes unmatched)
        3. We attach it as a "shadow" to the surviving track
        4. Shadow is counted when survivor exits top (both bags reached destination)
        5. Shadow is marked lost if survivor exits bottom/falls off
        """
        if not matches or not unmatched_track_indices:
            return

        matched_tracks = [(track_list[ti], ti) for ti, _ in matches]

        for unmatched_idx in list(unmatched_track_indices):
            lost_track = track_list[unmatched_idx]

            # Only consider tracks with enough history (not just created)
            if len(lost_track.bbox_history) < 3:
                continue

            # Skip tracks that are already shadows
            if lost_track.shadow_of is not None:
                continue

            lost_cx, lost_cy = lost_track.center
            lost_vel = lost_track.velocity

            # Skip if track is moving downward (falling off)
            if lost_vel is not None and lost_vel[1] > 5:
                continue

            best_survivor = None
            best_score = float('inf')

            for survivor, survivor_idx in matched_tracks:
                # Skip if survivor is already a shadow
                if survivor.shadow_of is not None:
                    continue

                surv_cx, surv_cy = survivor.center
                surv_vel = survivor.velocity

                # Skip if survivor is moving downward
                if surv_vel is not None and surv_vel[1] > 5:
                    continue

                # Check 1: Position proximity (PRIMARY)
                # Lost track should be very close to survivor
                x_gap = abs(surv_cx - lost_cx)
                y_gap = abs(surv_cy - lost_cy)

                # Must be within spatial tolerance
                if x_gap > self._merge_spatial_tol or y_gap > self._merge_y_tol:
                    continue

                # Calculate proximity score (lower is better)
                proximity_score = x_gap + y_gap

                # Check 2: Bbox growth (SECONDARY - bonus for detection)
                # If survivor's bbox grew, it's more likely they merged
                growth_bonus = 0
                if len(survivor.bbox_history) >= 4:
                    current_width = survivor.bbox[2] - survivor.bbox[0]
                    recent_widths = [b[2] - b[0] for b in list(survivor.bbox_history)[-5:-1]]
                    if recent_widths:
                        avg_width = sum(recent_widths) / len(recent_widths)
                        if growth_bonus >= 1.2:  # Lowered from 1.4
                            growth_bonus = -50  # Reduce score (better match)

                # Check 3: Similar Y position (both at similar height on conveyor)
                y_similarity_penalty = abs(surv_cy - lost_cy) * 0.5

                final_score = proximity_score + growth_bonus + y_similarity_penalty

                if final_score < best_score:
                    best_score = final_score
                    best_survivor = survivor

            # Attach to best survivor if found and score is good enough
            if best_survivor is not None and best_score < self._merge_spatial_tol:
                self._attach_shadow(best_survivor, lost_track)
                # Remove from unmatched set so it's not processed further
                unmatched_track_indices.discard(unmatched_idx)

    def _attach_shadow(self, survivor: TrackedObject, shadow: TrackedObject):
        """
        Attach a shadow track to a surviving track.

        IMPORTANT: We prefer keeping the MORE ESTABLISHED track as the survivor:
        - Track with more hits (more detections = more reliable)
        - Track with bottom_entry (started from valid entry point)
        - Track that's been alive longer

        If the "shadow" is actually more established, we SWAP them.
        """
        # Determine which track should actually be the survivor
        # Score based on: hits (more = better), entry_type (bottom = better), age
        def track_quality(t: TrackedObject) -> float:
            score = t.hits * 10  # More hits = more reliable
            if t.entry_type == "bottom_entry":
                score += 100  # Strong preference for bottom entry (real bags)
            elif t.entry_type == "midway_entry":
                score -= 50  # Penalty for midway (likely detector artifact)
            score += (time.time() - t.created_at) * 5  # Older tracks preferred
            return score

        survivor_quality = track_quality(survivor)
        shadow_quality = track_quality(shadow)

        # If shadow is actually better quality, swap them
        if shadow_quality > survivor_quality:
            logger.info(
                f"[TRACK_LIFECYCLE] SHADOW_SWAP | "
                f"T{shadow.track_id} (quality={shadow_quality:.0f}, entry={shadow.entry_type}, hits={shadow.hits}) "
                f"is more established than "
                f"T{survivor.track_id} (quality={survivor_quality:.0f}, entry={survivor.entry_type}, hits={survivor.hits}), "
                f"swapping roles"
            )
            # Swap: the better track becomes survivor
            # We need to update the tracks dictionary to reflect this
            original_survivor_id = survivor.track_id
            original_shadow_id = shadow.track_id

            # The original survivor (now shadow) should be removed from active tracks
            # The original shadow (now survivor) should be added to active tracks
            if original_survivor_id in self.tracks:
                del self.tracks[original_survivor_id]
            self.tracks[original_shadow_id] = shadow

            # Now swap the roles
            survivor, shadow = shadow, survivor

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
            # Build position history from checkpoints for DB storage
            final_position_history = list(shadow.checkpoint_positions)

            # Include exit position (same as parent track's exit)
            exit_position = track.center
            if not final_position_history or final_position_history[-1] != exit_position:
                final_position_history.append(exit_position)

            event = TrackEvent(
                track_id=shadow_id,
                event_type='track_completed',
                bbox_history=list(shadow.bbox_history),
                position_history=final_position_history,
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

    def _complete_ghost_companions(self, exiting_track: TrackedObject):
        """
        Complete ghost tracks that were concurrent with a track exiting the top.

        When two bags travel side-by-side and both lose detection near the top
        (because they overlap), the merge/shadow system can't help because there
        is no survivor — both went unmatched simultaneously. However, if one
        of them is later re-detected and exits top successfully, that exit is
        hard evidence that its companion also made it.

        This method checks the ghost buffer for tracks that:
        1. Were recorded as concurrent with the exiting track
        2. Entered from bottom (real bags, not noise)
        3. Had enough detection hits (minimum evidence)
        4. Had their last real position in the upper portion of the frame
        5. Were traveling upward (not falling off)

        Companions are emitted as track_completed with ghost_exit_promoted=True
        so the UI can show them distinctly.
        """
        if not self.ghost_tracks or not exiting_track.concurrent_track_ids:
            return

        if self.frame_height is None or self.frame_height <= 0:
            return

        # Find ghosts that were concurrent with the exiting track
        companions_to_complete = []
        near_top_threshold = int(self.frame_height * self._ghost_exit_near_top_ratio)
        min_hits = max(3, self._ghost_exit_min_hits)  # at least 3

        for ghost_id, ghost_info in self.ghost_tracks.items():
            # Gate 1: Must have been concurrent with the exiting track
            if ghost_id not in exiting_track.concurrent_track_ids:
                continue

            ghost_track = ghost_info['track']

            # Gate 2: Must have entered from bottom
            if ghost_track.entry_type != "bottom_entry":
                logger.debug(
                    f"[GHOST_COMPANION] T{ghost_id} SKIP: entry_type="
                    f"{ghost_track.entry_type} (not bottom_entry)"
                )
                continue

            # Gate 3: Must have enough detections
            if ghost_track.hits < min_hits:
                logger.debug(
                    f"[GHOST_COMPANION] T{ghost_id} SKIP: hits="
                    f"{ghost_track.hits} < {min_hits}"
                )
                continue

            # Gate 4: Last real position must be in upper portion of frame
            last_real_pos = ghost_info['predicted_pos']  # Last REAL detection pos
            if last_real_pos is None:
                continue
            last_y = last_real_pos[1]

            if last_y > near_top_threshold:
                logger.debug(
                    f"[GHOST_COMPANION] T{ghost_id} SKIP: last_y={last_y} > "
                    f"threshold={near_top_threshold}"
                )
                continue

            # Gate 5: Must have been moving upward (not falling off)
            ghost_vel = ghost_track.velocity
            if ghost_vel is not None and ghost_vel[1] > 5:  # moving down
                logger.debug(
                    f"[GHOST_COMPANION] T{ghost_id} SKIP: moving downward "
                    f"vy={ghost_vel[1]:.1f}"
                )
                continue

            # Gate 6: Must not be a shadow (already handled by shadow system)
            if ghost_track.shadow_of is not None:
                continue

            companions_to_complete.append(ghost_id)

        # Complete each companion
        now = time.time()
        for ghost_id in companions_to_complete:
            ghost_info = self.ghost_tracks.pop(ghost_id)
            ghost_track = ghost_info['track']
            last_real_pos = ghost_info['predicted_pos']

            # Build position history
            final_position_history = list(ghost_track.checkpoint_positions)
            if not final_position_history or final_position_history[-1] != last_real_pos:
                final_position_history.append(last_real_pos)

            # Append the exiting track's exit position as the companion's exit
            exit_position = exiting_track.center
            if final_position_history[-1] != exit_position:
                final_position_history.append(exit_position)

            event = TrackEvent(
                track_id=ghost_id,
                event_type='track_completed',
                bbox_history=list(ghost_track.bbox_history),
                position_history=final_position_history,
                total_frames=ghost_track.age + ghost_track.hits,
                created_at=ghost_track.created_at,
                ended_at=now,
                avg_confidence=ghost_track.confidence,
                exit_direction='top',
                entry_type=ghost_track.entry_type,
                suspected_duplicate=False,
                ghost_recovery_count=ghost_track.ghost_recovery_count,
                occlusion_events=list(ghost_track.occlusion_events),
                shadow_of=None,
                shadow_count=0,
                merge_events=list(ghost_track.merge_events),
                ghost_exit_promoted=True,
                concurrent_track_count=len(ghost_track.concurrent_track_ids),
            )
            self.completed_tracks.append(event)
            self._ghost_exits_promoted += 1

            # Also complete any shadows attached to this companion
            if ghost_track.shadow_tracks:
                self._complete_with_shadows(ghost_track, 'top')

            logger.info(
                f"[TRACK_LIFECYCLE] T{ghost_id} GHOST_COMPANION_COMPLETED | "
                f"concurrent ghost completed by T{exiting_track.track_id} exit | "
                f"last_real_pos={last_real_pos}, hits={ghost_track.hits}, "
                f"entry_y={ghost_track.entry_center_y}, "
                f"total_companions={self._ghost_exits_promoted}"
            )

    def _lose_shadows_with_survivor(self, survivor: TrackedObject, ended_at: float):
        """
        Mark all shadows as track_lost when their survivor falls off / times out.

        SAFETY RULE: If a bag falls off the conveyor, any bags that were
        "hidden" behind it (shadows) are also considered lost - they should
        NOT be counted because we cannot verify they reached the destination.

        Args:
            survivor: The track that fell off / timed out
            ended_at: Timestamp when survivor was marked as lost
        """
        for shadow_id, shadow in survivor.shadow_tracks.items():
            # Build position history from checkpoints
            final_position_history = list(shadow.checkpoint_positions)

            # Use last known position of shadow (not survivor's position)
            last_pos = shadow.last_detected_position or shadow.center
            if not final_position_history or final_position_history[-1] != last_pos:
                final_position_history.append(last_pos)

            event = TrackEvent(
                track_id=shadow_id,
                event_type='track_lost',  # NOT counted - survivor fell off
                bbox_history=list(shadow.bbox_history),
                position_history=final_position_history,
                total_frames=shadow.age + shadow.hits,
                created_at=shadow.created_at,
                ended_at=ended_at,
                avg_confidence=shadow.confidence,
                exit_direction="survivor_lost",  # Special indicator
                entry_type=shadow.entry_type,
                suspected_duplicate=(shadow.entry_type == "midway_entry"),
                ghost_recovery_count=shadow.ghost_recovery_count,
                occlusion_events=list(shadow.occlusion_events),
                shadow_of=survivor.track_id,
                shadow_count=0,
                merge_events=list(shadow.merge_events)
            )
            self.completed_tracks.append(event)

            logger.info(
                f"[TRACK_LIFECYCLE] T{shadow_id} SHADOW_LOST | "
                f"shadow of T{survivor.track_id} lost when survivor fell off"
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
    
    def get_statistics(self) -> Dict:
        """
        Get tracker statistics for UI display and diagnostics.

        Returns:
            Dict with statistics including:
            - tracks_created: Total tracks created
            - duplicates_prevented: Duplicate tracks prevented
            - active_tracks: Currently active tracks
            - ghost_tracks: Currently in ghost buffer
            - learned_velocity: Learned conveyor velocity (px/sec)
        """
        return {
            'tracks_created': self._tracks_created,
            'duplicates_prevented': self._duplicates_prevented,
            'active_tracks': len(self.tracks),
            'ghost_tracks': len(self.ghost_tracks),
            'learned_velocity_px_sec': self._learned_velocity_px_per_sec,
            'next_track_id': self._next_id
        }

    def cleanup(self):
        """Clean up tracker resources."""
        self.tracks.clear()
        self.completed_tracks.clear()
        self.ghost_tracks.clear()
        logger.info("[ConveyorTracker] Cleanup complete")
