"""
Container Tracker for QR-based direction tracking.

Tracks containers identified by QR codes (1-5) as they move through the frame.
Determines direction of travel (horizontal / X-axis):
- Right → Left (positive direction): Filled container leaving, INCREMENT count
- Left → Right (negative direction): Empty container returning, LOG only

Usage:
    tracker = ContainerTracker(frame_width=1280)
    
    # Process each frame
    if qr_detection:
        event = tracker.update(qr_detection)
        if event:
            if event.direction == Direction.POSITIVE:
                count += 1
            # Always save snapshot
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import time

from src.utils.AppLogging import logger


class Direction(Enum):
    """Direction of container movement."""
    UNKNOWN = "unknown"
    POSITIVE = "positive"   # Right → Left (filled container leaving)
    NEGATIVE = "negative"   # Left → Right (empty container returning)


class TrackState(Enum):
    """State of a tracked container."""
    TRACKING = "tracking"       # Currently being tracked
    COMPLETED = "completed"     # Exited frame with known direction
    LOST = "lost"              # QR code lost before exit


@dataclass
class TrackedContainer:
    """A container being tracked through the frame."""
    qr_value: int                           # QR code value (1-5)
    track_id: int                           # Unique track ID
    entry_x: int                            # X position when first seen
    entry_time: float                       # Timestamp when first seen
    last_x: int                             # Most recent X position
    last_time: float                        # Most recent timestamp
    state: TrackState = TrackState.TRACKING
    
    # Position history for trajectory analysis
    positions: List[Tuple[int, int, float]] = field(default_factory=list)  # (x, y, time)
    
    # Exit information (filled when track completes)
    exit_x: Optional[int] = None
    exit_time: Optional[float] = None
    direction: Direction = Direction.UNKNOWN
    
    def add_position(self, x: int, y: int, timestamp: float) -> None:
        """Record a position in the history."""
        self.positions.append((x, y, timestamp))
        self.last_x = x
        self.last_time = timestamp
    
    @property
    def duration_seconds(self) -> float:
        """Duration from entry to last seen (or exit)."""
        end_time = self.exit_time if self.exit_time else self.last_time
        return end_time - self.entry_time
    
    @property
    def x_displacement(self) -> int:
        """Total X displacement (negative = moved left, positive = moved right)."""
        end_x = self.exit_x if self.exit_x is not None else self.last_x
        return end_x - self.entry_x
    
    @property
    def avg_velocity(self) -> float:
        """Average velocity in pixels/second (negative = leftward)."""
        if self.duration_seconds > 0:
            return self.x_displacement / self.duration_seconds
        return 0.0


@dataclass
class ContainerEvent:
    """Event emitted when a container exits the frame."""
    track_id: int
    qr_value: int
    direction: Direction
    timestamp: datetime
    entry_x: int
    exit_x: int
    duration_seconds: float
    positions: List[Tuple[int, int, float]]      # Full position history
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'track_id': self.track_id,
            'qr_value': self.qr_value,
            'direction': self.direction.value,
            'timestamp': self.timestamp.isoformat(),
            'entry_x': self.entry_x,
            'exit_x': self.exit_x,
            'duration_seconds': round(self.duration_seconds, 2),
            'position_count': len(self.positions),
        }


class ContainerTracker:
    """
    Tracks containers by QR code and determines movement direction.
    
    The tracker maintains state for each active container (identified by QR value)
    and emits events when containers exit the frame or are lost.
    
    Direction Detection (horizontal):
    - Positive (right→left): Container x position decreases (filled leaving)
    - Negative (left→right): Container x position increases (empty returning)
    
    Exit Zones:
    - Left exit zone: x < exit_zone_ratio * frame_width
    - Right exit zone: x > (1 - exit_zone_ratio) * frame_width
    
    Attributes:
        frame_width: Width of the video frame in pixels
        exit_zone_ratio: Fraction of frame width for exit zones (default 0.15)
        lost_timeout: Seconds without detection before marking track as lost
    """
    
    def __init__(
        self,
        frame_width: int = 1280,
        exit_zone_ratio: float = 0.15,
        lost_timeout: float = 2.0,
        min_displacement_ratio: float = 0.3,
    ):
        """
        Initialize the container tracker.
        
        Args:
            frame_width: Width of the video frame in pixels
            exit_zone_ratio: Fraction of frame width for exit zones
            lost_timeout: Seconds without detection before marking lost
            min_displacement_ratio: Minimum x displacement as fraction of frame width
                                    to determine direction (filters small movements)
        """
        self.frame_width = frame_width
        self.exit_zone_ratio = exit_zone_ratio
        self.lost_timeout = lost_timeout
        self.min_displacement_ratio = min_displacement_ratio
        
        # Calculate exit zone boundaries
        self.left_exit_x = int(frame_width * exit_zone_ratio)
        self.right_exit_x = int(frame_width * (1 - exit_zone_ratio))
        
        # Active tracks by QR value (only one container per QR value at a time)
        self._tracks: Dict[int, TrackedContainer] = {}
        
        # Track ID counter
        self._next_track_id = 1
        
        # Statistics
        self.total_positive = 0
        self.total_negative = 0
        self.total_lost = 0
        
        # Per-QR stats
        self.qr_stats: Dict[int, Dict[str, int]] = {
            i: {'positive': 0, 'negative': 0, 'lost': 0}
            for i in range(1, 6)
        }
        
        logger.info(
            f"[ContainerTracker] Initialized: frame_width={frame_width}, "
            f"exit_zones=[left<={self.left_exit_x}, right>={self.right_exit_x}], "
            f"lost_timeout={lost_timeout}s"
        )
    
    def update(
        self,
        qr_value: int,
        center: Tuple[int, int],
        timestamp: Optional[float] = None
    ) -> Optional[ContainerEvent]:
        """
        Update tracker with a new QR detection.
        
        Args:
            qr_value: QR code value (1-5)
            center: (x, y) center position of QR code
            timestamp: Detection timestamp (defaults to current time)
            
        Returns:
            ContainerEvent if the container exited the frame, None otherwise
        """
        if timestamp is None:
            timestamp = time.time()
        
        x, y = center
        
        # Check if we have an existing track for this QR value
        if qr_value in self._tracks:
            track = self._tracks[qr_value]
            track.add_position(x, y, timestamp)
            
            # Check if container has exited
            event = self._check_exit(track)
            if event:
                del self._tracks[qr_value]
                return event
        else:
            # New container
            track = TrackedContainer(
                qr_value=qr_value,
                track_id=self._next_track_id,
                entry_x=x,
                entry_time=timestamp,
                last_x=x,
                last_time=timestamp,
            )
            track.add_position(x, y, timestamp)
            self._tracks[qr_value] = track
            self._next_track_id += 1
            
            logger.debug(
                f"[ContainerTracker] New track #{track.track_id}: "
                f"QR={qr_value} entry_x={x} "
                f"exit_zones=[left<={self.left_exit_x} / right>={self.right_exit_x}]"
            )
        
        return None
    
    def check_lost_tracks(self, current_time: Optional[float] = None) -> List[ContainerEvent]:
        """
        Check for and handle lost tracks (QR code not seen for too long).
        
        Should be called periodically (e.g., every frame) to detect lost containers.
        
        Args:
            current_time: Current timestamp (defaults to time.time())
            
        Returns:
            List of ContainerEvents for lost tracks
        """
        if current_time is None:
            current_time = time.time()
        
        events = []
        lost_qr_values = []
        
        for qr_value, track in self._tracks.items():
            time_since_seen = current_time - track.last_time
            
            if time_since_seen > self.lost_timeout:
                # Mark as lost and determine direction from trajectory
                track.state = TrackState.LOST
                track.exit_time = track.last_time
                track.exit_x = track.last_x
                track.direction = self._determine_direction(track)
                
                event = ContainerEvent(
                    track_id=track.track_id,
                    qr_value=track.qr_value,
                    direction=track.direction,
                    timestamp=datetime.now(),
                    entry_x=track.entry_x,
                    exit_x=track.exit_x,
                    duration_seconds=track.duration_seconds,
                    positions=track.positions.copy(),
                )
                events.append(event)
                lost_qr_values.append(qr_value)
                
                # Update statistics
                self.total_lost += 1
                self.qr_stats[qr_value]['lost'] += 1
                
                # Also count based on final direction
                if track.direction == Direction.POSITIVE:
                    self.total_positive += 1
                    self.qr_stats[qr_value]['positive'] += 1
                elif track.direction == Direction.NEGATIVE:
                    self.total_negative += 1
                    self.qr_stats[qr_value]['negative'] += 1
                
                logger.info(
                    f"[ContainerTracker] Track #{track.track_id} TIMEOUT "
                    f"QR={qr_value} "
                    f"dir={track.direction.value} "
                    f"entry_x={track.entry_x} last_x={track.last_x} "
                    f"disp={track.x_displacement:+}px "
                    f"dur={track.duration_seconds:.2f}s "
                    f"time_since_seen={time_since_seen:.2f}s "
                    f"exit_zones=[left<={self.left_exit_x} / right>={self.right_exit_x}] "
                    f"positions={len(track.positions)}"
                )
        
        # Remove lost tracks
        for qr_value in lost_qr_values:
            del self._tracks[qr_value]
        
        return events
    
    def _check_exit(self, track: TrackedContainer) -> Optional[ContainerEvent]:
        """
        Check if a track has exited via left or right of frame.
        
        Args:
            track: The tracked container to check
            
        Returns:
            ContainerEvent if exited, None otherwise
        """
        x = track.last_x
        
        # Check left exit (positive direction: right → left, filled leaving)
        if x <= self.left_exit_x:
            track.state = TrackState.COMPLETED
            track.exit_x = x
            track.exit_time = track.last_time
            track.direction = Direction.POSITIVE
            
            self.total_positive += 1
            self.qr_stats[track.qr_value]['positive'] += 1
            
            logger.info(
                f"[ContainerTracker] Track #{track.track_id} EXIT-LEFT (positive): "
                f"QR={track.qr_value} "
                f"entry_x={track.entry_x} exit_x={x} "
                f"disp={track.x_displacement:+}px "
                f"dur={track.duration_seconds:.2f}s "
                f"positions={len(track.positions)}"
            )
            
            return ContainerEvent(
                track_id=track.track_id,
                qr_value=track.qr_value,
                direction=Direction.POSITIVE,
                timestamp=datetime.now(),
                entry_x=track.entry_x,
                exit_x=track.exit_x,
                duration_seconds=track.duration_seconds,
                positions=track.positions.copy(),
            )
        
        # Check right exit (negative direction: left → right, empty returning)
        if x >= self.right_exit_x:
            track.state = TrackState.COMPLETED
            track.exit_x = x
            track.exit_time = track.last_time
            track.direction = Direction.NEGATIVE
            
            self.total_negative += 1
            self.qr_stats[track.qr_value]['negative'] += 1
            
            logger.info(
                f"[ContainerTracker] Track #{track.track_id} EXIT-RIGHT (negative): "
                f"QR={track.qr_value} "
                f"entry_x={track.entry_x} exit_x={x} "
                f"disp={track.x_displacement:+}px "
                f"dur={track.duration_seconds:.2f}s "
                f"positions={len(track.positions)}"
            )
            
            return ContainerEvent(
                track_id=track.track_id,
                qr_value=track.qr_value,
                direction=Direction.NEGATIVE,
                timestamp=datetime.now(),
                entry_x=track.entry_x,
                exit_x=track.exit_x,
                duration_seconds=track.duration_seconds,
                positions=track.positions.copy(),
            )
        
        return None
    
    def _determine_direction(self, track: TrackedContainer) -> Direction:
        """
        Determine direction from trajectory when track is lost mid-frame.
        
        Uses overall displacement to infer direction:
        - Significant leftward movement → positive (filled leaving)
        - Significant rightward movement → negative (empty returning)
        - Insufficient movement → unknown
        
        Args:
            track: The tracked container
            
        Returns:
            Direction based on trajectory analysis
        """
        displacement = track.x_displacement
        min_displacement = int(self.frame_width * self.min_displacement_ratio)
        
        if displacement < -min_displacement:
            # Moved left significantly (filled leaving)
            return Direction.POSITIVE
        elif displacement > min_displacement:
            # Moved right significantly (empty returning)
            return Direction.NEGATIVE
        else:
            # Not enough movement to determine
            return Direction.UNKNOWN
    
    def get_active_tracks(self) -> Dict[int, TrackedContainer]:
        """Get all currently active tracks."""
        return self._tracks.copy()
    
    def get_stats(self) -> dict:
        """Get tracker statistics."""
        return {
            'total_positive': self.total_positive,
            'total_negative': self.total_negative,
            'total_lost': self.total_lost,
            'active_tracks': len(self._tracks),
            'qr_stats': self.qr_stats.copy(),
            'config': {
                'frame_width': self.frame_width,
                'exit_zone_ratio': self.exit_zone_ratio,
                'lost_timeout': self.lost_timeout,
                'left_exit_x': self.left_exit_x,
                'right_exit_x': self.right_exit_x,
            }
        }
    
    def reset(self) -> None:
        """Reset all tracks and statistics."""
        self._tracks.clear()
        self.total_positive = 0
        self.total_negative = 0
        self.total_lost = 0
        self.qr_stats = {
            i: {'positive': 0, 'negative': 0, 'lost': 0}
            for i in range(1, 6)
        }
        logger.info("[ContainerTracker] Reset complete")
    
    def update_frame_width(self, new_width: int) -> None:
        """Update frame width and recalculate exit zones."""
        self.frame_width = new_width
        self.left_exit_x = int(new_width * self.exit_zone_ratio)
        self.right_exit_x = int(new_width * (1 - self.exit_zone_ratio))
        logger.info(
            f"[ContainerTracker] Frame width updated: {new_width}, "
            f"exit_zones=[left<={self.left_exit_x}, right>={self.right_exit_x}]"
        )
