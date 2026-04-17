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
    entry_time: float                       # Wall-clock timestamp when first seen
    last_x: int                             # Most recent X position
    last_time: float                        # Most recent wall-clock timestamp
    state: TrackState = TrackState.TRACKING

    # Monotonic-clock counterpart of ``entry_time`` — used to align the
    # event's pre-roll against the content-camera ring buffer (which is
    # indexed by ``time.monotonic()``).  Kept separate from ``entry_time``
    # because the two clocks can drift relative to each other.
    entry_time_monotonic: float = 0.0

    # Number of real QR detections that have fed this track.  Used as a
    # false-positive gate: a track with only 1-2 sightings is likely a
    # decoder glitch and should not emit a counted event.
    detection_count: int = 0

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
    # Monotonic-clock time the track first appeared — the correct anchor
    # for aligning a pre-event video roll against the ring buffer.
    entry_time_monotonic: float = 0.0
    # Number of confirmed QR detections that contributed to this track
    # (informational — already filtered by the tracker's min-detections gate).
    detection_count: int = 0
    
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
        min_detections_for_event: int = 3,
    ):
        """
        Initialize the container tracker.

        Args:
            frame_width: Width of the video frame in pixels
            exit_zone_ratio: Fraction of frame width for exit zones
            lost_timeout: Seconds without detection before marking lost
            min_displacement_ratio: Minimum x displacement as fraction of frame width
                                    to determine direction (filters small movements)
            min_detections_for_event: Minimum number of *real* QR detections
                a track must accumulate before it is allowed to emit an
                event.  Filters out single-frame decoder glitches that
                would otherwise count as containers.  Set to ``1`` to
                disable (legacy behaviour).
        """
        self.frame_width = frame_width
        self.exit_zone_ratio = exit_zone_ratio
        self.lost_timeout = lost_timeout
        self.min_displacement_ratio = min_displacement_ratio
        self.min_detections_for_event = max(1, int(min_detections_for_event))
        
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
        timestamp: Optional[float] = None,
        is_prediction: bool = False,
    ) -> Optional[ContainerEvent]:
        """
        Update tracker with a new QR detection.

        Args:
            qr_value: QR code value (1-5)
            center: (x, y) center position of QR code
            timestamp: Detection timestamp (defaults to current time)
            is_prediction: ``True`` when ``center`` is extrapolated from a
                linear predictor rather than an actual QR decoder output.
                Predicted positions maintain the track's trajectory but
                do **not** count toward the ``min_detections_for_event``
                false-positive gate.

        Returns:
            ContainerEvent if the container exited the frame, None otherwise
        """
        if timestamp is None:
            timestamp = time.time()
        
        x, y = center
        # Monotonic timestamp captured alongside the wall-clock one so
        # downstream consumers (EventVideoCoordinator, ContentCameraRecorder)
        # can align the pre-roll ring buffer precisely.
        t_mono = time.monotonic()

        # Check if we have an existing track for this QR value
        if qr_value in self._tracks:
            track = self._tracks[qr_value]
            track.add_position(x, y, timestamp)
            if not is_prediction:
                track.detection_count += 1

            # Check if container has exited
            event = self._check_exit(track)
            if event:
                del self._tracks[qr_value]
                return event
            # The FP gate inside ``_check_exit`` sets state=COMPLETED
            # when it drops a track for too few detections.  Remove the
            # track here so callers stop feeding it frames and its
            # downstream buffers can be reclaimed on the next tick.
            if track.state == TrackState.COMPLETED:
                del self._tracks[qr_value]
                return None
        else:
            # New container.  We allow predicted positions to start a
            # track only if a detection has already happened on a prior
            # frame \u2014 otherwise a single prediction can't create a track.
            if is_prediction:
                return None
            track = TrackedContainer(
                qr_value=qr_value,
                track_id=self._next_track_id,
                entry_x=x,
                entry_time=timestamp,
                last_x=x,
                last_time=timestamp,
                entry_time_monotonic=t_mono,
                detection_count=1,
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
                # False-positive gate: a track that never accumulated
                # enough detections is likely a QR decoder glitch.
                if track.detection_count < self.min_detections_for_event:
                    logger.info(
                        f"[ContainerTracker] Track #{track.track_id} DROPPED "
                        f"(likely false positive on lost): QR={qr_value} "
                        f"detections={track.detection_count} "
                        f"required>={self.min_detections_for_event}"
                    )
                    lost_qr_values.append(qr_value)
                    continue

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
                    entry_time_monotonic=track.entry_time_monotonic,
                    detection_count=track.detection_count,
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
            ContainerEvent if exited AND the track accumulated enough real
            detections to be considered valid, ``None`` otherwise.  A track
            that exits with too few detections is dropped silently and
            logged at ``info`` level as a suspected false positive.
        """
        x = track.last_x

        # Left or right exit?
        exited_left = x <= self.left_exit_x
        exited_right = x >= self.right_exit_x
        if not (exited_left or exited_right):
            return None

        # False-positive gate.
        if track.detection_count < self.min_detections_for_event:
            logger.info(
                f"[ContainerTracker] Track #{track.track_id} DROPPED "
                f"(likely false positive): QR={track.qr_value} "
                f"detections={track.detection_count} "
                f"required>={self.min_detections_for_event} "
                f"entry_x={track.entry_x} exit_x={x} "
                f"dur={track.duration_seconds:.2f}s"
            )
            track.state = TrackState.COMPLETED
            return None

        if exited_left:
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
                entry_time_monotonic=track.entry_time_monotonic,
                detection_count=track.detection_count,
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
                entry_time_monotonic=track.entry_time_monotonic,
                detection_count=track.detection_count,
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

    def get_track(self, qr_value: int) -> Optional['TrackedContainer']:
        """Return the active track for *qr_value*, or ``None`` if not present.

        Prefer this over accessing ``_tracks`` directly so call-sites are
        insulated from internal data-structure changes.
        """
        return self._tracks.get(qr_value)
    
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
