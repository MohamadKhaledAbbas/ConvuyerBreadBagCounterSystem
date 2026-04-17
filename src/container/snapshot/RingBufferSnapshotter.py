"""
Ring Buffer Snapshotter for container events.

Maintains a circular buffer of frames to capture context before and after
container events (QR code disappearance). Saves 5 seconds before and 5 seconds
after the event.

At 20 FPS:
- Pre-event buffer: 100 frames (5 seconds)
- Post-event capture: 100 frames (5 seconds)
- Total per event: 200 frames

Usage:
    snapshotter = RingBufferSnapshotter(fps=20)
    
    # Feed frames continuously
    snapshotter.add_frame(frame)
    
    # When QR disappears, trigger capture
    snapshotter.trigger_event(qr_value=3, direction='positive')
    
    # Continue feeding frames for post-event capture
    snapshotter.add_frame(frame)
    
    # Check for completed captures to save
    completed = snapshotter.get_completed_captures()
"""

from dataclasses import dataclass, field
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple
from enum import Enum
import os
import json
import threading
import cv2
import numpy as np

from src.utils.AppLogging import logger


class CaptureState(Enum):
    """State of an event capture."""
    PENDING = "pending"       # Waiting for post-event frames
    COMPLETE = "complete"     # All frames collected
    SAVED = "saved"          # Saved to disk
    FAILED = "failed"        # Save failed


@dataclass
class EventCapture:
    """Represents a captured event with pre and post frames."""
    event_id: str                           # Unique event ID
    qr_value: int                           # Container QR code value
    direction: str                          # 'positive' or 'negative'
    trigger_time: datetime                  # When the event was triggered
    pre_frames: List[np.ndarray]            # Frames before the event
    post_frames: List[np.ndarray] = field(default_factory=list)
    state: CaptureState = CaptureState.PENDING
    frames_needed: int = 100                # Post-event frames still needed
    save_path: Optional[str] = None         # Path where capture was saved
    
    @property
    def total_frames(self) -> int:
        """Total frames in this capture."""
        return len(self.pre_frames) + len(self.post_frames)


class RingBufferSnapshotter:
    """
    Maintains a ring buffer of frames for event capture.
    
    When an event is triggered (e.g., QR code disappears), the snapshotter:
    1. Copies the current pre-event buffer
    2. Continues collecting post-event frames
    3. Saves the complete capture to disk
    
    Supports multiple simultaneous captures (rare but possible).
    
    Attributes:
        fps: Expected frames per second
        pre_event_seconds: Seconds to capture before event
        post_event_seconds: Seconds to capture after event
        output_dir: Directory for saved captures
    """
    
    def __init__(
        self,
        fps: int = 20,
        pre_event_seconds: float = 5.0,
        post_event_seconds: float = 5.0,
        output_dir: str = "data/container_snapshots",
        max_concurrent_captures: int = 5,
        jpeg_quality: int = 85,
    ):
        """
        Initialize the ring buffer snapshotter.
        
        Args:
            fps: Expected frames per second
            pre_event_seconds: Seconds to capture before event
            post_event_seconds: Seconds to capture after event
            output_dir: Directory for saved captures
            max_concurrent_captures: Maximum simultaneous event captures
            jpeg_quality: JPEG save quality (0-100)
        """
        self.fps = fps
        self.pre_event_seconds = pre_event_seconds
        self.post_event_seconds = post_event_seconds
        self.output_dir = output_dir
        self.max_concurrent_captures = max_concurrent_captures
        self.jpeg_quality = jpeg_quality
        
        # Calculate buffer sizes
        self.pre_event_frames = int(fps * pre_event_seconds)
        self.post_event_frames = int(fps * post_event_seconds)
        
        # Ring buffer for pre-event frames
        self._buffer: Deque[np.ndarray] = deque(maxlen=self.pre_event_frames)
        
        # Active captures being filled with post-event frames
        self._active_captures: Dict[str, EventCapture] = {}
        
        # Completed captures ready for saving
        self._completed_captures: List[EventCapture] = []
        
        # Statistics
        self.total_events_triggered = 0
        self.total_events_saved = 0
        self.total_events_failed = 0
        
        # Thread safety
        self._lock = threading.RLock()  # RLock allows get_stats() to call get_buffer_status() under the same lock
        
        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"[RingBufferSnapshotter] Initialized: fps={fps}, "
            f"pre={pre_event_seconds}s ({self.pre_event_frames} frames), "
            f"post={post_event_seconds}s ({self.post_event_frames} frames), "
            f"output={output_dir}"
        )
    
    def add_frame(self, frame: np.ndarray) -> None:
        """
        Add a frame to the ring buffer.
        
        Also updates any active captures with post-event frames.
        
        Args:
            frame: BGR image (numpy array)
        """
        if frame is None or frame.size == 0:
            return

        # Downscale to half resolution before buffering.
        # At 1280×720 this reduces ring-buffer RAM from ~330 MB to ~83 MB
        # (150 frames × 3 bytes/px × 640×360 ≈ 83 MB) with no visible
        # quality loss in the saved JPEG sequence.
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (w // 2, h // 2))

        with self._lock:
            # Add to ring buffer (pre-event)
            self._buffer.append(small)

            # Update active captures with post-event frame
            completed_ids = []

            for event_id, capture in self._active_captures.items():
                if capture.frames_needed > 0:
                    capture.post_frames.append(small)
                    capture.frames_needed -= 1
                    
                    if capture.frames_needed == 0:
                        capture.state = CaptureState.COMPLETE
                        completed_ids.append(event_id)
            
            # Move completed captures
            for event_id in completed_ids:
                capture = self._active_captures.pop(event_id)
                self._completed_captures.append(capture)
                logger.debug(
                    f"[RingBufferSnapshotter] Capture {event_id} complete: "
                    f"{capture.total_frames} frames"
                )
    
    def trigger_event(
        self,
        qr_value: int,
        direction: str,
        event_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Trigger an event capture.
        
        Copies the current pre-event buffer and starts collecting post-event frames.
        
        Args:
            qr_value: Container QR code value (1-5)
            direction: 'positive' or 'negative'
            event_id: Optional custom event ID (generated if not provided)
            
        Returns:
            Event ID if capture started, None if at capacity
        """
        with self._lock:
            # Check capacity
            if len(self._active_captures) >= self.max_concurrent_captures:
                logger.warning(
                    f"[RingBufferSnapshotter] At capacity ({self.max_concurrent_captures}), "
                    "dropping event"
                )
                return None
            
            # Generate event ID
            if event_id is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                event_id = f"qr{qr_value}_{direction}_{timestamp}"
            
            # Copy pre-event frames
            pre_frames = list(self._buffer)
            
            # Create capture
            capture = EventCapture(
                event_id=event_id,
                qr_value=qr_value,
                direction=direction,
                trigger_time=datetime.now(),
                pre_frames=pre_frames,
                frames_needed=self.post_event_frames,
            )
            
            self._active_captures[event_id] = capture
            self.total_events_triggered += 1
            
            logger.info(
                f"[RingBufferSnapshotter] Event triggered: {event_id}, "
                f"pre_frames={len(pre_frames)}, post_needed={self.post_event_frames}"
            )
            
            return event_id
    
    def get_completed_captures(self) -> List[EventCapture]:
        """
        Get and remove completed captures.
        
        Returns:
            List of completed EventCapture objects
        """
        with self._lock:
            completed = self._completed_captures.copy()
            self._completed_captures.clear()
            return completed
    
    def save_capture(
        self,
        capture: EventCapture,
        save_video: bool = False
    ) -> Optional[str]:
        """
        Save a capture to disk.
        
        Creates a directory with:
        - metadata.json: Event information
        - pre/: Pre-event frames as JPEGs
        - post/: Post-event frames as JPEGs
        - (optional) video.mp4: Combined video
        
        Args:
            capture: The EventCapture to save
            save_video: Whether to also create an MP4 video
            
        Returns:
            Path to the saved directory, or None if failed
        """
        try:
            # Create capture directory
            capture_dir = os.path.join(self.output_dir, capture.event_id)
            pre_dir = os.path.join(capture_dir, "pre")
            post_dir = os.path.join(capture_dir, "post")
            
            os.makedirs(pre_dir, exist_ok=True)
            os.makedirs(post_dir, exist_ok=True)
            
            jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
            
            # Save pre-event frames
            for i, frame in enumerate(capture.pre_frames):
                filename = os.path.join(pre_dir, f"frame_{i:04d}.jpg")
                cv2.imwrite(filename, frame, jpeg_params)
            
            # Save post-event frames
            for i, frame in enumerate(capture.post_frames):
                filename = os.path.join(post_dir, f"frame_{i:04d}.jpg")
                cv2.imwrite(filename, frame, jpeg_params)
            
            # Save metadata
            metadata = {
                'event_id': capture.event_id,
                'qr_value': capture.qr_value,
                'direction': capture.direction,
                'trigger_time': capture.trigger_time.isoformat(),
                'pre_frame_count': len(capture.pre_frames),
                'post_frame_count': len(capture.post_frames),
                'total_frames': capture.total_frames,
                'fps': self.fps,
                'pre_event_seconds': self.pre_event_seconds,
                'post_event_seconds': self.post_event_seconds,
            }
            
            metadata_path = os.path.join(capture_dir, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Optional: Create video
            if save_video and capture.total_frames > 0:
                video_path = os.path.join(capture_dir, "video.mp4")
                self._create_video(
                    capture.pre_frames + capture.post_frames,
                    video_path
                )
            
            capture.state = CaptureState.SAVED
            capture.save_path = capture_dir
            self.total_events_saved += 1
            
            logger.info(
                f"[RingBufferSnapshotter] Saved capture: {capture_dir} "
                f"({capture.total_frames} frames)"
            )
            
            return capture_dir
            
        except Exception as e:
            capture.state = CaptureState.FAILED
            self.total_events_failed += 1
            logger.error(f"[RingBufferSnapshotter] Failed to save capture: {e}")
            return None
    
    def _create_video(self, frames: List[np.ndarray], output_path: str) -> bool:
        """
        Create an MP4 video from frames.
        
        Args:
            frames: List of BGR frames
            output_path: Output video path
            
        Returns:
            True if successful, False otherwise
        """
        if not frames:
            return False
        
        try:
            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))
            
            for frame in frames:
                writer.write(frame)
            
            writer.release()
            return True
            
        except Exception as e:
            logger.error(f"[RingBufferSnapshotter] Video creation failed: {e}")
            return False
    
    def get_buffer_status(self) -> dict:
        """Get current buffer status."""
        with self._lock:
            return {
                'buffer_size': len(self._buffer),
                'buffer_capacity': self.pre_event_frames,
                'buffer_fill_percent': (
                    len(self._buffer) / self.pre_event_frames * 100
                    if self.pre_event_frames > 0 else 0
                ),
                'active_captures': len(self._active_captures),
                'pending_saves': len(self._completed_captures),
            }
    
    def get_stats(self) -> dict:
        """Get snapshotter statistics."""
        with self._lock:
            return {
                'config': {
                    'fps': self.fps,
                    'pre_event_seconds': self.pre_event_seconds,
                    'post_event_seconds': self.post_event_seconds,
                    'pre_event_frames': self.pre_event_frames,
                    'post_event_frames': self.post_event_frames,
                    'output_dir': self.output_dir,
                },
                'buffer': self.get_buffer_status(),
                'totals': {
                    'triggered': self.total_events_triggered,
                    'saved': self.total_events_saved,
                    'failed': self.total_events_failed,
                },
            }
    
    def clear_buffer(self) -> None:
        """Clear the ring buffer and cancel active captures."""
        with self._lock:
            self._buffer.clear()
            self._active_captures.clear()
            self._completed_captures.clear()
            logger.info("[RingBufferSnapshotter] Buffer and captures cleared")
