"""
Retention Policy Module for Segment Cleanup.

Manages automatic cleanup of processed segment files based on:
1. Age-based retention (max_age_hours)
2. Processed segment tracking (delete only after processing)
3. Storage limits (max_storage_bytes)
4. Minimum segment retention (keep N most recent)
"""

import os
import time
import threading
from pathlib import Path
from typing import Optional, List, Set
from dataclasses import dataclass, field

from src.utils.AppLogging import logger


@dataclass
class RetentionConfig:
    """Configuration for segment retention policy."""
    max_age_hours: float = 24.0
    max_storage_bytes: int = 10 * 1024 * 1024 * 1024  # 10 GB
    min_segments_keep: int = 5
    check_interval_seconds: float = 60.0
    only_delete_processed: bool = True


class RetentionPolicy:
    """
    Manages segment file retention based on configured policies.

    Features:
    - Age-based deletion (configurable max age)
    - Storage-based deletion (keep under limit)
    - Processing-aware deletion (only delete processed segments)
    - Minimum retention (always keep N most recent)
    - Background cleanup thread
    """

    def __init__(
            self,
            spool_dir: str,
            config: Optional[RetentionConfig] = None
    ):
        """
        Initialize retention policy.

        Args:
            spool_dir: Directory containing segment files
            config: Retention configuration
        """
        self.spool_dir = Path(spool_dir)
        self.config = config or RetentionConfig()

        # Track processed segments
        self._processed_segments: Set[int] = set()
        self._processed_lock = threading.Lock()

        # Background thread control
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Statistics
        self.total_deleted: int = 0
        self.bytes_freed: int = 0
        self.last_check_time: float = 0.0

    def mark_processed(self, segment_num: int):
        """
        Mark a segment as fully processed and eligible for deletion.

        Args:
            segment_num: Segment number to mark
        """
        with self._processed_lock:
            self._processed_segments.add(segment_num)
            logger.debug(f"[Retention] Marked segment {segment_num} as processed")

    def is_processed(self, segment_num: int) -> bool:
        """Check if a segment has been processed."""
        with self._processed_lock:
            return segment_num in self._processed_segments

    def _list_segment_files(self) -> List[tuple]:
        """
        List all segment files with metadata.

        Returns:
            List of (segment_num, path, size, mtime) tuples, sorted oldest first
        """
        segments = []

        try:
            for entry in os.scandir(self.spool_dir):
                if not entry.name.endswith('.bin'):
                    continue
                if not entry.name.startswith('seg_'):
                    continue

                try:
                    # Extract segment number
                    num_str = entry.name[4:-4]  # seg_NNNNNN.bin
                    segment_num = int(num_str)

                    stat = entry.stat()
                    segments.append((
                        segment_num,
                        Path(entry.path),
                        stat.st_size,
                        stat.st_mtime
                    ))
                except (ValueError, OSError) as e:
                    logger.warning(f"[Retention] Error parsing segment {entry.name}: {e}")
                    continue

        except OSError as e:
            logger.error(f"[Retention] Error scanning spool directory: {e}")

        # Sort by segment number (oldest first)
        segments.sort(key=lambda x: x[0])
        return segments

    def _delete_segment(self, segment_num: int, path: Path):
        """
        Delete a segment file and its metadata.

        Args:
            segment_num: Segment number
            path: Path to segment file
        """
        deleted_size = 0

        try:
            # Delete main segment file
            if path.exists():
                deleted_size = path.stat().st_size
                path.unlink()

            # Delete metadata file if exists
            meta_path = path.with_suffix('.json')
            if meta_path.exists():
                meta_path.unlink()

            # Remove from processed tracking
            with self._processed_lock:
                self._processed_segments.discard(segment_num)

            self.total_deleted += 1
            self.bytes_freed += deleted_size

            logger.info(f"[Retention] Deleted segment {segment_num} ({deleted_size} bytes)")

        except OSError as e:
            logger.error(f"[Retention] Error deleting segment {segment_num}: {e}")

    def _get_total_storage(self, segments: List[tuple]) -> int:
        """Calculate total storage used by segments."""
        return sum(s[2] for s in segments)

    def run_cleanup(self) -> int:
        """
        Run a single cleanup cycle.

        Returns:
            Number of segments deleted
        """
        self.last_check_time = time.time()
        segments = self._list_segment_files()

        if not segments:
            return 0

        deleted_count = 0
        now = time.time()
        max_age_seconds = self.config.max_age_hours * 3600

        # Calculate how many we must keep
        min_keep = self.config.min_segments_keep
        can_delete_count = max(0, len(segments) - min_keep)

        if can_delete_count == 0:
            return 0

        # Candidates for deletion (oldest first, excluding min_keep newest)
        deletion_candidates = segments[:can_delete_count]

        # First pass: Delete old segments
        for segment_num, path, size, mtime in deletion_candidates:
            age = now - mtime

            # Check age policy
            if age < max_age_seconds:
                continue

            # Check processing policy
            if self.config.only_delete_processed:
                if not self.is_processed(segment_num):
                    logger.debug(f"[Retention] Skipping unprocessed segment {segment_num}")
                    continue

            self._delete_segment(segment_num, path)
            deleted_count += 1

        # Second pass: Check storage limit
        segments = self._list_segment_files()
        total_storage = self._get_total_storage(segments)

        if total_storage > self.config.max_storage_bytes:
            can_delete_count = max(0, len(segments) - min_keep)
            deletion_candidates = segments[:can_delete_count]

            for segment_num, path, size, mtime in deletion_candidates:
                if total_storage <= self.config.max_storage_bytes:
                    break

                # Check processing policy (less strict for storage limit)
                if self.config.only_delete_processed:
                    if not self.is_processed(segment_num):
                        continue

                self._delete_segment(segment_num, path)
                total_storage -= size
                deleted_count += 1

        if deleted_count > 0:
            logger.info(
                f"[Retention] Cleanup complete: {deleted_count} deleted, "
                f"{self._get_total_storage(self._list_segment_files()) / (1024 ** 3):.2f}GB used"
            )

        return deleted_count

    def _cleanup_loop(self):
        """Background cleanup loop."""
        logger.info("[Retention] Background cleanup started")

        while not self._stop_event.is_set():
            try:
                self.run_cleanup()
            except Exception as e:
                logger.error(f"[Retention] Cleanup error: {e}")

            self._stop_event.wait(self.config.check_interval_seconds)

        logger.info("[Retention] Background cleanup stopped")

    def start(self):
        """Start background retention management."""
        if self._running:
            return

        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._cleanup_loop,
            name="RetentionPolicy",
            daemon=True
        )
        self._thread.start()
        logger.info(
            f"[Retention] Started: max_age={self.config.max_age_hours}h, "
            f"max_storage={self.config.max_storage_bytes / (1024 ** 3):.1f}GB"
        )

    def stop(self, timeout: float = 5.0):
        """Stop background retention management."""
        if not self._running:
            return

        self._running = False
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

        logger.info(
            f"[Retention] Stopped. Total deleted: {self.total_deleted}, "
            f"bytes freed: {self.bytes_freed / (1024 ** 3):.2f}GB"
        )

    def get_stats(self) -> dict:
        """Get retention statistics."""
        segments = self._list_segment_files()
        return {
            'total_segments': len(segments),
            'total_storage_bytes': self._get_total_storage(segments),
            'processed_count': len(self._processed_segments),
            'total_deleted': self.total_deleted,
            'bytes_freed': self.bytes_freed,
            'last_check_time': self.last_check_time
        }
