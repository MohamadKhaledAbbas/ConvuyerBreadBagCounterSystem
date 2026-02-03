"""
Spool Utility Functions.

Provides common utilities for spool operations:
- Structured logging helpers
- State persistence
- CRC calculation
- File system helpers
"""

import json
import os
import struct
import threading
import time
import zlib
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from src.utils.AppLogging import logger


def format_structured_log(
        event: str,
        segment: Optional[int] = None,
        frame: Optional[int] = None,
        latency_ms: Optional[float] = None,
        **kwargs
) -> str:
    """
    Format a structured log message for spool operations.

    Args:
        event: Event name (e.g., "segment_complete", "frame_written")
        segment: Segment number
        frame: Frame number or index
        latency_ms: Processing latency in milliseconds
        **kwargs: Additional key-value pairs

    Returns:
        Formatted log string
    """
    parts = [f"event={event}"]

    if segment is not None:
        parts.append(f"segment={segment}")
    if frame is not None:
        parts.append(f"frame={frame}")
    if latency_ms is not None:
        parts.append(f"latency_ms={latency_ms:.2f}")

    for key, value in kwargs.items():
        if value is not None:
            if isinstance(value, float):
                parts.append(f"{key}={value:.3f}")
            else:
                parts.append(f"{key}={value}")

    return " | ".join(parts)


@dataclass
class ProcessorState:
    """
    Persistent state for spool processor.

    Tracks processing position to allow resumption after restart.
    """
    last_segment: int = -1
    last_frame_index: int = -1
    last_update_time: float = 0.0
    total_frames_processed: int = 0
    total_segments_processed: int = 0
    session_start_time: float = 0.0

    def update(self, segment: int, frame_index: int):
        """Update state with current position."""
        self.last_segment = segment
        self.last_frame_index = frame_index
        self.last_update_time = time.time()
        self.total_frames_processed += 1

    def complete_segment(self, segment: int):
        """Mark a segment as completely processed."""
        self.last_segment = segment
        self.total_segments_processed += 1
        self.last_update_time = time.time()


def save_processor_state(state: ProcessorState, state_file: str) -> bool:
    """
    Save processor state to disk atomically.

    Args:
        state: Current processor state
        state_file: Path to state file

    Returns:
        True if successful
    """
    try:
        state_path = Path(state_file)
        tmp_path = state_path.with_suffix('.tmp')

        data = asdict(state)
        data['saved_at'] = datetime.now().isoformat()

        with open(tmp_path, 'w') as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        tmp_path.rename(state_path)
        return True

    except Exception as e:
        logger.error(f"[SpoolUtils] Error saving state: {e}")
        return False


def load_processor_state(state_file: str) -> Optional[ProcessorState]:
    """
    Load processor state from disk.

    Args:
        state_file: Path to state file

    Returns:
        ProcessorState if found and valid, None otherwise
    """
    try:
        state_path = Path(state_file)

        if not state_path.exists():
            return None

        with open(state_path, 'r') as f:
            data = json.load(f)

        state = ProcessorState(
            last_segment=data.get('last_segment', -1),
            last_frame_index=data.get('last_frame_index', -1),
            last_update_time=data.get('last_update_time', 0.0),
            total_frames_processed=data.get('total_frames_processed', 0),
            total_segments_processed=data.get('total_segments_processed', 0),
            session_start_time=data.get('session_start_time', 0.0)
        )

        logger.info(
            f"[SpoolUtils] Loaded state: segment={state.last_segment}, "
            f"frame={state.last_frame_index}"
        )
        return state

    except Exception as e:
        logger.error(f"[SpoolUtils] Error loading state: {e}")
        return None


def calculate_crc32(data: bytes) -> int:
    """
    Calculate CRC32 checksum for data.

    Args:
        data: Bytes to checksum

    Returns:
        CRC32 value as unsigned integer
    """
    return zlib.crc32(data) & 0xFFFFFFFF


def verify_segment_integrity(segment_path: str) -> tuple:
    """
    Verify segment file integrity.

    Args:
        segment_path: Path to segment file

    Returns:
        Tuple of (is_valid, frame_count, error_message)
    """
    from src.spool.segment_io import SEGMENT_MAGIC, SEGMENT_HEADER_SIZE, RECORD_HEADER_SIZE

    path = Path(segment_path)
    if not path.exists():
        return False, 0, "File not found"

    try:
        with open(path, 'rb') as f:
            # Check header
            header = f.read(SEGMENT_HEADER_SIZE)
            if len(header) < SEGMENT_HEADER_SIZE:
                return False, 0, "Truncated header"

            if header[:6] != SEGMENT_MAGIC:
                return False, 0, "Invalid magic bytes"

            # Count frames
            frame_count = 0
            while True:
                record_header = f.read(RECORD_HEADER_SIZE)
                if len(record_header) == 0:
                    break
                if len(record_header) < RECORD_HEADER_SIZE:
                    return False, frame_count, f"Truncated record header at frame {frame_count}"

                # Extract data length
                data_len = struct.unpack_from("<I", record_header, 50)[0]
                data = f.read(data_len)

                if len(data) < data_len:
                    return False, frame_count, f"Truncated data at frame {frame_count}"

                frame_count += 1

        return True, frame_count, None

    except Exception as e:
        return False, 0, str(e)


def get_segment_info(segment_path: str) -> Optional[Dict[str, Any]]:
    """
    Get information about a segment file.

    Args:
        segment_path: Path to segment file

    Returns:
        Dict with segment info or None on error
    """
    path = Path(segment_path)
    if not path.exists():
        return None

    try:
        stat = path.stat()

        # Extract segment number from filename
        segment_num = -1
        if path.name.startswith('seg_') and path.suffix == '.bin':
            try:
                segment_num = int(path.stem[4:])
            except ValueError:
                pass

        # Verify and count frames
        is_valid, frame_count, error = verify_segment_integrity(segment_path)

        return {
            'path': str(path),
            'segment_num': segment_num,
            'size_bytes': stat.st_size,
            'modified_time': stat.st_mtime,
            'is_valid': is_valid,
            'frame_count': frame_count,
            'error': error
        }

    except Exception as e:
        logger.error(f"[SpoolUtils] Error getting segment info: {e}")
        return None


def cleanup_tmp_files(spool_dir: str) -> int:
    """
    Clean up orphaned .tmp files from incomplete writes.

    Args:
        spool_dir: Spool directory path

    Returns:
        Number of files cleaned up
    """
    count = 0
    try:
        for entry in os.scandir(spool_dir):
            if entry.name.endswith('.tmp'):
                try:
                    os.unlink(entry.path)
                    count += 1
                    logger.info(f"[SpoolUtils] Cleaned up orphaned: {entry.name}")
                except OSError as e:
                    logger.warning(f"[SpoolUtils] Error removing {entry.name}: {e}")
    except OSError as e:
        logger.error(f"[SpoolUtils] Error scanning for tmp files: {e}")

    return count


class RateLimiter:
    """
    Simple rate limiter for controlling frame publishing rate.

    Uses token bucket algorithm for smooth rate limiting.
    """

    def __init__(self, target_fps: float, burst_frames: int = 5):
        """
        Initialize rate limiter.

        Args:
            target_fps: Target frames per second
            burst_frames: Maximum burst size
        """
        self.target_fps = target_fps
        self.burst_frames = burst_frames

        self._tokens = float(burst_frames)
        self._last_time = time.monotonic()
        self._lock = threading.Lock()

        # Interval between frames
        self._interval = 1.0 / target_fps if target_fps > 0 else 0.0

    def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire permission to publish a frame.

        Blocks until rate limit allows or timeout expires.

        Args:
            timeout: Maximum time to wait (None = wait forever)

        Returns:
            True if acquired, False if timed out
        """
        deadline = None
        if timeout is not None:
            deadline = time.monotonic() + timeout

        while True:
            with self._lock:
                now = time.monotonic()

                # Refill tokens
                elapsed = now - self._last_time
                self._tokens = min(
                    float(self.burst_frames),
                    self._tokens + elapsed * self.target_fps
                )
                self._last_time = now

                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return True

                # Calculate wait time
                wait_time = (1.0 - self._tokens) / self.target_fps

            # Check timeout
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                wait_time = min(wait_time, remaining)

            time.sleep(wait_time)

    def update_rate(self, new_fps: float):
        """Update target FPS."""
        with self._lock:
            self.target_fps = new_fps
            self._interval = 1.0 / new_fps if new_fps > 0 else 0.0


class AdaptivePacer:
    """
    Adaptive frame pacing for matching playback rate to recording rate.

    Adjusts playback speed based on segment queue depth to prevent
    falling too far behind or running too far ahead.
    """

    def __init__(
            self,
            base_fps: float = 30.0,
            min_fps: float = 15.0,
            max_fps: float = 60.0,
            target_queue_depth: int = 2
    ):
        """
        Initialize adaptive pacer.

        Args:
            base_fps: Base playback FPS
            min_fps: Minimum FPS when falling behind
            max_fps: Maximum FPS when catching up
            target_queue_depth: Target number of segments in queue
        """
        self.base_fps = base_fps
        self.min_fps = min_fps
        self.max_fps = max_fps
        self.target_queue_depth = target_queue_depth

        self._current_fps = base_fps
        self._last_frame_time = 0.0

    def calculate_fps(self, queue_depth: int) -> float:
        """
        Calculate target FPS based on queue depth.

        Args:
            queue_depth: Number of segments waiting to be processed

        Returns:
            Target FPS
        """
        if queue_depth > self.target_queue_depth:
            # Speed up to catch up
            excess = queue_depth - self.target_queue_depth
            speedup = 1.0 + (0.1 * excess)  # 10% faster per extra segment
            self._current_fps = min(self.base_fps * speedup, self.max_fps)

        elif queue_depth == 0:
            # Slow down to avoid running out
            self._current_fps = self.min_fps

        else:
            # Normal speed
            self._current_fps = self.base_fps

        return self._current_fps

    def wait_for_next_frame(self):
        """Wait appropriate time for next frame."""
        if self._last_frame_time > 0:
            target_interval = 1.0 / self._current_fps
            elapsed = time.monotonic() - self._last_frame_time
            remaining = target_interval - elapsed

            if remaining > 0:
                time.sleep(remaining)

        self._last_frame_time = time.monotonic()

    def get_current_fps(self) -> float:
        """Get current FPS setting."""
        return self._current_fps
