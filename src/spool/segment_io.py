"""
Segment I/O Module for H.264 Frame Spooling.

Provides writer and reader classes for binary segment files that store
H.264 frames with metadata in a compact format suitable for 24/7 operation.

Segment File Format (Version 1):
================================
Header:
  - Magic bytes: "SPOOL1" (6 bytes)
  - Version: uint8 (1 byte)
  - Flags: uint8 (1 byte, reserved)

Each Record:
  - Magic: "FR" (2 bytes, record marker)
  - Index: uint32 (frame index from source)
  - Width: uint32
  - Height: uint32
  - DTS seconds: int64
  - DTS nanoseconds: uint32
  - PTS seconds: int64
  - PTS nanoseconds: uint32
  - Encoding: 12 bytes (null-padded string, e.g., "H264")
  - Data length: uint32
  - Data: raw bytes

Atomic Writes:
==============
Files are written with .tmp extension and atomically renamed to .bin
when complete. This prevents corruption during crashes and allows
safe retention policy operation.
"""

import os
import struct
import time
import json
import threading
from dataclasses import dataclass, field
from typing import Optional, Iterator, List, Dict, Any, Tuple
from pathlib import Path

from src.utils.AppLogging import logger

# Segment file constants
SEGMENT_MAGIC = b"SPOOL1"
SEGMENT_VERSION = 1
SEGMENT_HEADER_SIZE = 8  # 6 (magic) + 1 (version) + 1 (flags)

# Record constants
RECORD_MAGIC = b"FR"
RECORD_HEADER_SIZE = 54  # 2 + 4 + 4 + 4 + 8 + 4 + 8 + 4 + 12 + 4 = 54 bytes

# Record header struct format
RECORD_STRUCT = struct.Struct("<2sIIIqIqI12sI")


@dataclass
class FrameRecord:
    """
    Represents a single frame record in a segment file.

    Attributes:
        index: Frame index from the original source
        width: Frame width in pixels
        height: Frame height in pixels
        dts_sec: Decode timestamp seconds
        dts_nsec: Decode timestamp nanoseconds
        pts_sec: Presentation timestamp seconds
        pts_nsec: Presentation timestamp nanoseconds
        encoding: Encoding type string (e.g., "H264", "H265")
        data: Raw encoded frame data
    """
    index: int
    width: int
    height: int
    dts_sec: int
    dts_nsec: int
    pts_sec: int
    pts_nsec: int
    encoding: str
    data: bytes

    def to_bytes(self) -> bytes:
        """Serialize the frame record to bytes."""
        # Handle encoding field that might be str, bytes, or numpy array
        if isinstance(self.encoding, str):
            encoding_bytes = self.encoding.encode('utf-8')[:12].ljust(12, b'\x00')
        elif isinstance(self.encoding, bytes):
            encoding_bytes = self.encoding[:12].ljust(12, b'\x00')
        else:
            try:
                encoding_bytes = bytes(self.encoding)[:12].ljust(12, b'\x00')
            except (TypeError, ValueError):
                encoding_bytes = b'H264'.ljust(12, b'\x00')
        
        header = RECORD_STRUCT.pack(
            RECORD_MAGIC,
            self.index,
            self.width,
            self.height,
            self.dts_sec,
            self.dts_nsec,
            self.pts_sec,
            self.pts_nsec,
            encoding_bytes,
            len(self.data)
        )
        return header + self.data

    @classmethod
    def from_bytes(cls, header_bytes: bytes, data: bytes) -> 'FrameRecord':
        """Deserialize a frame record from bytes."""
        (
            magic, index, width, height,
            dts_sec, dts_nsec, pts_sec, pts_nsec,
            encoding_bytes, data_len
        ) = RECORD_STRUCT.unpack(header_bytes)

        if magic != RECORD_MAGIC:
            raise ValueError(f"Invalid record magic: {magic!r}")

        encoding = encoding_bytes.rstrip(b'\x00').decode('utf-8')

        return cls(
            index=index,
            width=width,
            height=height,
            dts_sec=dts_sec,
            dts_nsec=dts_nsec,
            pts_sec=pts_sec,
            pts_nsec=pts_nsec,
            encoding=encoding,
            data=data
        )

    @property
    def dts_ns(self) -> int:
        """Get DTS as total nanoseconds."""
        return self.dts_sec * 1_000_000_000 + self.dts_nsec

    @property
    def pts_ns(self) -> int:
        """Get PTS as total nanoseconds."""
        return self.pts_sec * 1_000_000_000 + self.pts_nsec


@dataclass
class SegmentMetadata:
    """Metadata for a segment file."""
    segment_num: int
    start_time: float
    end_time: Optional[float] = None
    frame_count: int = 0
    bytes_written: int = 0
    first_frame_index: int = -1
    last_frame_index: int = -1
    has_idr: bool = False


class SegmentWriter:
    """
    Writes H.264 frames to segment files with atomic completion.

    Features:
    - Atomic writes using .tmp -> .bin rename
    - Segment rotation based on duration
    - IDR-aligned rotation when possible
    - Metadata file generation
    - Thread-safe operations
    """

    def __init__(
            self,
            spool_dir: str,
            segment_duration: float = 5.0,
            max_segment_duration: float = 10.0,
            write_metadata: bool = True
    ):
        """
        Initialize segment writer.
        
        Args:
            spool_dir: Directory for segment files
            segment_duration: Target segment duration in seconds
            max_segment_duration: Maximum segment duration before forced rotation
            write_metadata: Whether to write JSON metadata files
        """
        self.spool_dir = Path(spool_dir)
        self.segment_duration = segment_duration
        self.max_segment_duration = max_segment_duration
        self.write_metadata = write_metadata

        self._current_segment: int = -1
        self._current_file = None
        self._current_metadata: Optional[SegmentMetadata] = None
        self._segment_start_time: Optional[float] = None
        self._lock = threading.Lock()

        # SPS/PPS caching
        self._cached_sps: Optional[bytes] = None
        self._cached_pps: Optional[bytes] = None

        # Statistics
        self.total_bytes_written: int = 0
        self.total_frames_written: int = 0
        self.segments_completed: int = 0

    def start(self):
        """Initialize the writer and create spool directory."""
        self.spool_dir.mkdir(parents=True, exist_ok=True)
        self._find_next_segment_number()
        logger.info(f"[SegmentWriter] Started in {self.spool_dir}, next segment: {self._current_segment}")

    def _find_next_segment_number(self):
        """Find the next available segment number."""
        existing = []
        for f in self.spool_dir.glob("seg_*.bin"):
            try:
                num = int(f.stem.split('_')[1])
                existing.append(num)
            except (IndexError, ValueError):
                continue
        
        # Also check .tmp files
        for f in self.spool_dir.glob("seg_*.tmp"):
            try:
                num = int(f.stem.split('_')[1])
                existing.append(num)
            except (IndexError, ValueError):
                continue

        self._current_segment = max(existing, default=-1) + 1

    def _get_segment_path(self, segment_num: int, tmp: bool = False) -> Path:
        """Get path for a segment file."""
        ext = '.tmp' if tmp else '.bin'
        return self.spool_dir / f"seg_{segment_num:06d}{ext}"

    def _get_metadata_path(self, segment_num: int) -> Path:
        """Get path for segment metadata file."""
        return self.spool_dir / f"seg_{segment_num:06d}.json"

    def _open_new_segment(self):
        """Open a new segment file."""
        tmp_path = self._get_segment_path(self._current_segment, tmp=True)
        self._current_file = open(tmp_path, 'wb')

        # Write header
        header = SEGMENT_MAGIC + bytes([SEGMENT_VERSION, 0])
        self._current_file.write(header)

        self._segment_start_time = time.time()
        self._current_metadata = SegmentMetadata(
            segment_num=self._current_segment,
            start_time=self._segment_start_time
        )

        logger.debug(f"[SegmentWriter] Opened segment {self._current_segment}")

    def _close_current_segment(self):
        """Close and finalize the current segment."""
        if self._current_file is None:
            return

        # Flush and close file
        self._current_file.flush()
        os.fsync(self._current_file.fileno())
        self._current_file.close()
        self._current_file = None

        # Atomic rename .tmp -> .bin
        tmp_path = self._get_segment_path(self._current_segment, tmp=True)
        final_path = self._get_segment_path(self._current_segment, tmp=False)
        tmp_path.rename(final_path)

        # Write metadata
        if self.write_metadata and self._current_metadata:
            self._current_metadata.end_time = time.time()
            meta_path = self._get_metadata_path(self._current_segment)
            with open(meta_path, 'w') as f:
                json.dump({
                    'segment_num': self._current_metadata.segment_num,
                    'start_time': self._current_metadata.start_time,
                    'end_time': self._current_metadata.end_time,
                    'frame_count': self._current_metadata.frame_count,
                    'bytes_written': self._current_metadata.bytes_written,
                    'first_frame_index': self._current_metadata.first_frame_index,
                    'last_frame_index': self._current_metadata.last_frame_index,
                    'has_idr': self._current_metadata.has_idr
                }, f, indent=2)

        self.segments_completed += 1
        logger.info(
            f"[SegmentWriter] Closed segment {self._current_segment}: "
            f"{self._current_metadata.frame_count} frames, "
            f"{self._current_metadata.bytes_written} bytes"
        )

        self._current_segment += 1
        self._current_metadata = None
        self._segment_start_time = None

    def _should_rotate(self, has_idr: bool) -> bool:
        """Check if segment should be rotated."""
        if self._segment_start_time is None:
            return False

        elapsed = time.time() - self._segment_start_time

        # Force rotation at max duration
        if elapsed >= self.max_segment_duration:
            return True

        # Prefer rotation at IDR boundaries after target duration
        if elapsed >= self.segment_duration and has_idr:
            return True

        return False

    def write_frame(self, record: FrameRecord, has_idr: bool = False) -> bool:
        """
        Write a frame record to the current segment.

        Args:
            record: The frame record to write
            has_idr: Whether this frame contains an IDR

        Returns:
            True if successful, False on error
        """
        with self._lock:
            try:
                # Check for rotation
                if self._should_rotate(has_idr):
                    self._close_current_segment()

                # Open new segment if needed
                if self._current_file is None:
                    self._open_new_segment()

                # Write the record
                record_bytes = record.to_bytes()
                self._current_file.write(record_bytes)

                # Update statistics
                self.total_bytes_written += len(record_bytes)
                self.total_frames_written += 1

                if self._current_metadata:
                    self._current_metadata.frame_count += 1
                    self._current_metadata.bytes_written += len(record_bytes)
                    if self._current_metadata.frame_count == 1:
                        self._current_metadata.first_frame_index = record.index
                        self._current_metadata.has_idr = has_idr
                    self._current_metadata.last_frame_index = record.index

                return True

            except Exception as e:
                logger.error(f"[SegmentWriter] Error writing frame: {e}")
                return False

    def update_sps_pps(self, sps: Optional[bytes], pps: Optional[bytes]):
        """Update cached SPS/PPS for segment boundary insertion."""
        with self._lock:
            if sps:
                self._cached_sps = sps
            if pps:
                self._cached_pps = pps

    def flush(self):
        """Flush the current file to disk."""
        with self._lock:
            if self._current_file:
                self._current_file.flush()
                os.fsync(self._current_file.fileno())

    def close(self):
        """Close the writer and finalize any open segment."""
        with self._lock:
            if self._current_file:
                self._close_current_segment()
        logger.info(
            f"[SegmentWriter] Closed. Total: {self.total_frames_written} frames, "
            f"{self.total_bytes_written} bytes, {self.segments_completed} segments"
        )


class SegmentReader:
    """
    Reads H.264 frames from segment files.

    Supports:
    - Sequential reading of segment files in order
    - Iterator interface for frame-by-frame access
    - Automatic segment progression
    - Metadata reading
    - Cached segment list with configurable refresh rate
    """

    def __init__(self, spool_dir: str, cache_refresh_interval: float = 1.0):
        """
        Initialize segment reader.
        
        Args:
            spool_dir: Directory containing segment files
            cache_refresh_interval: How often to refresh segment list cache
        """
        self.spool_dir = Path(spool_dir)
        self.cache_refresh_interval = cache_refresh_interval

        self._segment_cache: List[int] = []
        self._cache_time: float = 0.0
        self._cache_lock = threading.Lock()

        # Tracking for current read position
        self._current_segment: int = -1
        self._last_completed_segment: int = -1
        self._segment_lock = threading.Lock()

    def list_segments(self, use_cache: bool = True) -> List[int]:
        """
        List available segment numbers, sorted ascending.
        
        Args:
            use_cache: Whether to use cached list if fresh
            
        Returns:
            Sorted list of segment numbers
        """
        with self._cache_lock:
            now = time.time()
            if use_cache and (now - self._cache_time) < self.cache_refresh_interval:
                return self._segment_cache.copy()

            segments = []
            try:
                with os.scandir(self.spool_dir) as it:
                    for entry in it:
                        if not entry.name.endswith('.bin'):
                            continue
                        if not entry.name.startswith('seg_'):
                            continue
                        try:
                            num_str = entry.name[4:-4]
                            num = int(num_str)
                            segments.append(num)
                        except (ValueError, IndexError):
                            continue

                segments.sort()
                self._segment_cache = segments
                self._cache_time = now

            except Exception as e:
                logger.error(f"[SegmentReader] Error listing segments: {e}")

            return segments

    def get_oldest_segment(self) -> Optional[int]:
        """Get the oldest available segment number."""
        segments = self.list_segments()
        return segments[0] if segments else None

    def get_newest_segment(self) -> Optional[int]:
        """Get the newest available segment number."""
        segments = self.list_segments()
        return segments[-1] if segments else None

    def read_segment_metadata(self, segment_num: int) -> Optional[SegmentMetadata]:
        """Read metadata for a specific segment."""
        meta_path = self.spool_dir / f"seg_{segment_num:06d}.json"
        if not meta_path.exists():
            return None

        try:
            with open(meta_path, 'r') as f:
                data = json.load(f)
            return SegmentMetadata(
                segment_num=data['segment_num'],
                start_time=data['start_time'],
                end_time=data.get('end_time'),
                frame_count=data['frame_count'],
                bytes_written=data['bytes_written'],
                first_frame_index=data.get('first_frame_index', -1),
                last_frame_index=data.get('last_frame_index', -1),
                has_idr=data.get('has_idr', False)
            )
        except Exception as e:
            logger.error(f"[SegmentReader] Error reading metadata for segment {segment_num}: {e}")
            return None

    def read_segment(self, segment_num: int) -> Iterator[FrameRecord]:
        """
        Read all frames from a specific segment.

        Args:
            segment_num: Segment number to read

        Yields:
            FrameRecord objects
        """
        path = self.spool_dir / f"seg_{segment_num:06d}.bin"
        if not path.exists():
            logger.warning(f"[SegmentReader] Segment {segment_num} not found")
            return

        try:
            with open(path, 'rb') as f:
                # Read and verify header
                header = f.read(SEGMENT_HEADER_SIZE)
                if len(header) < SEGMENT_HEADER_SIZE:
                    logger.error(f"[SegmentReader] Truncated header in segment {segment_num}")
                    return

                if header[:6] != SEGMENT_MAGIC:
                    logger.error(f"[SegmentReader] Invalid magic in segment {segment_num}")
                    return

                version = header[6]
                if version != SEGMENT_VERSION:
                    logger.warning(f"[SegmentReader] Unknown version {version} in segment {segment_num}")

                # Read records
                while True:
                    record_header = f.read(RECORD_HEADER_SIZE)
                    if len(record_header) < RECORD_HEADER_SIZE:
                        break

                    # Extract data length from header (at offset 50)
                    data_len = struct.unpack_from("<I", record_header, 50)[0]
                    data = f.read(data_len)

                    if len(data) < data_len:
                        logger.error(f"[SegmentReader] Truncated record in segment {segment_num}")
                        break

                    try:
                        record = FrameRecord.from_bytes(record_header, data)
                        yield record
                    except Exception as e:
                        logger.error(f"[SegmentReader] Error parsing record: {e}")
                        continue

        except Exception as e:
            logger.error(f"[SegmentReader] Error reading segment {segment_num}: {e}")

    def read_single_segment(self, segment_num: int) -> Iterator[FrameRecord]:
        """
        Read frames from a SINGLE segment only.
        
        Updates tracking state for completion detection.
        """
        with self._segment_lock:
            self._current_segment = segment_num
        
        logger.debug(f"[SegmentReader] Reading segment {segment_num}")
        
        yield from self.read_segment(segment_num)
        
        with self._segment_lock:
            self._last_completed_segment = segment_num

    def read_frames(self, start_segment: Optional[int] = None) -> Iterator[FrameRecord]:
        """
        Read frames from all segments in order.

        Args:
            start_segment: Optional segment number to start from

        Yields:
            FrameRecord objects in chronological order
        """
        segments = self.list_segments()

        if start_segment is not None:
            segments = [s for s in segments if s >= start_segment]

        for seg_num in segments:
            with self._segment_lock:
                self._current_segment = seg_num
            
            logger.debug(f"[SegmentReader] Reading segment {seg_num}")
            yield from self.read_segment(seg_num)
            
            with self._segment_lock:
                self._last_completed_segment = seg_num

    def get_current_segment(self) -> int:
        """Get the segment number currently being read."""
        with self._segment_lock:
            return self._current_segment

    def get_last_completed_segment(self) -> int:
        """Get the last segment that was fully processed."""
        with self._segment_lock:
            return self._last_completed_segment


def validate_segment_file(path: str) -> bool:
    """
    Validate a segment file's integrity.
    
    Args:
        path: Path to segment file
        
    Returns:
        True if file is valid, False otherwise
    """
    try:
        with open(path, 'rb') as f:
            header = f.read(SEGMENT_HEADER_SIZE)
            if len(header) < SEGMENT_HEADER_SIZE:
                return False
            if header[:6] != SEGMENT_MAGIC:
                return False
        return True
    except Exception:
        return False
