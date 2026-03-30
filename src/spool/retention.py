"""
Retention Policy Module for Segment Cleanup.

Manages automatic cleanup of processed segment files based on:
1. Age-based retention (max_age_hours)
2. Processed segment tracking (delete only after processing)
3. Storage limits (max_storage_bytes)
4. Minimum segment retention (keep N most recent)
"""

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Set

from src.utils.AppLogging import logger
from src.config.paths import PIPELINE_THROTTLE_STATE_FILE


@dataclass
class RetentionConfig:
    """Configuration for segment retention policy.

    Defaults are tuned for a live conveyor-counting pipeline where old
    footage has no replay value.  Keep only the last 5 minutes of
    segments (enough for sentinel wake catch-up) and cap total spool
    storage at 1 GB to avoid filling the eMMC.
    """
    max_age_hours: float = 5.0 / 60.0          # 5 minutes — old idle segments are useless
    max_storage_bytes: int = 200 * 1024 * 1024  # 200 MB hard cap
    min_segments_keep: int = 5                  # full mode: always keep last 5 for catch-up
    idle_max_segments: int = 10                 # sentinel/power-save: keep only last N segments
    check_interval_seconds: float = 30.0        # check every 30 s (tight retention needs faster checks)
    only_delete_processed: bool = True
    # Throttle state file — retention reads this DIRECTLY so it knows the
    # current pipeline mode without depending on set_idle_mode() being called.
    # Same file that ConveyorCounterApp writes and SpoolProcessorNode reads.
    # Set to "" to disable self-detection and rely only on set_idle_mode().
    throttle_state_path: str = PIPELINE_THROTTLE_STATE_FILE
    throttle_staleness_s: float = 120.0         # treat stale file as "full" mode


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

        # Operating mode — set by the spool processor on sentinel transitions
        self._idle_mode: bool = False

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

    def set_idle_mode(self, idle: bool) -> None:
        """Signal whether the pipeline is in idle / sentinel / power-save mode.

        In idle mode the cleanup strategy switches from age-based to
        count-based: only the most-recent ``idle_max_segments`` segments are
        kept; everything older is deleted unconditionally on the next cycle.
        This is correct because the processor intentionally skips all idle
        segments — they will never be marked as processed.

        In full mode the processing-aware two-pass strategy resumes:
        within-grace-period segments are only deleted after processing;
        segments that have aged out are deleted regardless.

        Args:
            idle: True when entering sentinel/power-save mode, False on wake.
        """
        if idle == self._idle_mode:
            return
        self._idle_mode = idle
        if idle:
            logger.info(
                f"[Retention] → IDLE/sentinel mode — "
                f"keeping last {self.config.idle_max_segments} segments, "
                f"all older deleted unconditionally"
            )
        else:
            logger.info(
                "[Retention] → FULL mode — "
                "processing-aware age/storage cleanup resumed"
            )

    def delete_processed_immediately(self, segment_num: int) -> bool:
        """
        Immediately delete a segment that has been fully processed.

        This is called by the spool processor after completing a segment
        to free disk space immediately rather than waiting for periodic cleanup.

        Args:
            segment_num: Segment number to delete

        Returns:
            True if segment was deleted, False otherwise
        """
        segment_path = self.spool_dir / f"seg_{segment_num:06d}.bin"
        t0 = time.monotonic()

        if not segment_path.exists():
            logger.warning(
                f"[Retention] SEG_DELETE phase=immediate_missing seg={segment_num} "
                f"path={segment_path} elapsed_ms={(time.monotonic() - t0) * 1000:.1f}"
            )
            return False

        try:
            # Delete main segment file
            size = segment_path.stat().st_size
            segment_path.unlink()

            # Delete metadata file if exists
            meta_path = segment_path.with_suffix('.json')
            if meta_path.exists():
                meta_path.unlink()

            # Remove from processed tracking
            with self._processed_lock:
                self._processed_segments.discard(segment_num)

            self.total_deleted += 1
            self.bytes_freed += size

            logger.debug(
                f"[Retention] SEG_DELETE phase=immediate_done seg={segment_num} "
                f"size_kb={size / 1024:.1f} elapsed_ms={(time.monotonic() - t0) * 1000:.1f} "
                f"deleted_total={self.total_deleted}"
            )
            return True

        except OSError as e:
            logger.error(
                f"[Retention] SEG_DELETE phase=immediate_error seg={segment_num} "
                f"path={segment_path} elapsed_ms={(time.monotonic() - t0) * 1000:.1f} err={e}"
            )
            return False

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
        t0 = time.monotonic()

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
        t0 = time.monotonic()

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

            logger.warning(
                f"[Retention] SEG_DELETE phase=bg_done seg={segment_num} "
                f"size_bytes={deleted_size} elapsed_ms={(time.monotonic() - t0) * 1000:.1f}"
            )

        except OSError as e:
            logger.error(
                f"[Retention] SEG_DELETE phase=bg_error seg={segment_num} "
                f"path={path} elapsed_ms={(time.monotonic() - t0) * 1000:.1f} err={e}"
            )

    def _is_idle_mode(self) -> bool:
        """Return True if the pipeline is currently in idle / sentinel mode.

        Primary source: the shared throttle state file written by
        ``ConveyorCounterApp``.  This is the same file the spool processor
        reads, so retention is always in sync with the actual pipeline mode
        without depending on ``set_idle_mode()`` being called at the right
        time.

        Fall-back (file missing, unreadable, or stale): the manually-set
        ``_idle_mode`` flag (from ``set_idle_mode()``).
        """
        path = self.config.throttle_state_path
        if not path:
            return self._idle_mode

        try:
            with open(path, "r") as f:
                state = json.load(f)

            mode = state.get("mode", "full")
            updated_at = float(state.get("updated_at", 0.0))
            age = time.time() - updated_at

            if age > self.config.throttle_staleness_s:
                # Stale → treat as full (same safety logic as read_throttle_state)
                return False

            return mode == "degraded"

        except FileNotFoundError:
            # File not yet written (early startup) — fall back to manual flag
            return self._idle_mode
        except Exception as e:
            logger.debug(f"[Retention] Could not read throttle state: {e}")
            return self._idle_mode

    def _get_total_storage(self, segments: List[tuple]) -> int:
        """Calculate total storage used by segments."""
        return sum(s[2] for s in segments)

    def run_cleanup(self) -> int:
        """
        Run a single cleanup cycle.

        Strategy depends on the current operating mode:

        **Idle / sentinel mode** — count-based:
            Keep only the last ``idle_max_segments`` segments and delete
            everything older unconditionally.  No age check, no processing
            check.  The processor intentionally skips idle segments so they
            will never be marked as processed; count-based eviction is the
            only mechanism that works.

        **Full mode** — processing-aware two-pass:
            Pass 1 (age): segments older than ``max_age_hours`` are deleted
            unconditionally — age overrides ``only_delete_processed``.
            Pass 2 (storage cap): if still over ``max_storage_bytes``, delete
            oldest segments; within the grace period only delete processed
            ones.

        In both modes ``min_segments_keep`` (full) / ``idle_max_segments``
        (idle) act as a hard floor on the number of segments retained.

        Returns:
            Number of segments deleted in this cycle.
        """
        self.last_check_time = time.time()
        cycle_t0 = time.monotonic()
        segments = self._list_segment_files()

        if not segments:
            return 0

        deleted_count = 0
        now = time.time()

        # Determine mode once for this cycle.  Reads the throttle state file
        # directly so it works even before set_idle_mode() has been called.
        idle_mode = self._is_idle_mode()

        # ── IDLE / SENTINEL mode: count-based cleanup ─────────────────────
        if idle_mode:
            idle_keep = self.config.idle_max_segments
            can_delete = max(0, len(segments) - idle_keep)
            if can_delete == 0:
                return 0

            for segment_num, path, size, mtime in segments[:can_delete]:
                self._delete_segment(segment_num, path)
                deleted_count += 1

            if deleted_count > 0:
                logger.info(
                    f"[Retention] CLEANUP_CYCLE mode=idle deleted={deleted_count} "
                    f"kept={idle_keep} total_on_disk={len(segments)} "
                    f"elapsed_ms={(time.monotonic() - cycle_t0) * 1000:.1f}"
                )
            return deleted_count

        # ── FULL mode: processing-aware two-pass cleanup ───────────────────
        max_age_seconds = self.config.max_age_hours * 3600
        min_keep = self.config.min_segments_keep
        can_delete_count = max(0, len(segments) - min_keep)

        if can_delete_count == 0:
            return 0

        # Candidates for deletion (oldest first, excluding min_keep newest)
        deletion_candidates = segments[:can_delete_count]

        # ── Pass 1: age-based deletion ──
        # A segment that has outlived max_age is deleted unconditionally.
        # only_delete_processed is intentionally NOT checked here: in idle /
        # sentinel mode the processor never calls mark_processed(), so the set
        # stays empty and the age gate would never fire — that was the original
        # bug.  min_segments_keep already protects the most-recent N segments
        # needed for sentinel wake-up catch-up.
        for segment_num, path, size, mtime in deletion_candidates:
            age = now - mtime
            if age < max_age_seconds:
                continue  # still within grace period — leave for processor

            self._delete_segment(segment_num, path)
            deleted_count += 1

        # ── Pass 2: storage-cap enforcement ──
        # Delete oldest segments until we are under the storage limit.
        # Respect only_delete_processed ONLY while a segment is still within
        # its grace period (age < max_age).  If it has already aged out, size
        # pressure overrides the processing requirement.
        segments = self._list_segment_files()
        total_storage = self._get_total_storage(segments)

        if total_storage > self.config.max_storage_bytes:
            can_delete_count = max(0, len(segments) - min_keep)
            deletion_candidates = segments[:can_delete_count]
            for segment_num, path, size, mtime in deletion_candidates:
                if total_storage <= self.config.max_storage_bytes:
                    break

                age = now - mtime
                if self.config.only_delete_processed and age < max_age_seconds:
                    # Within grace period — only delete if already processed
                    if not self.is_processed(segment_num):
                        continue

                self._delete_segment(segment_num, path)
                total_storage -= size
                deleted_count += 1

        if deleted_count > 0:
            logger.info(
                f"[Retention] CLEANUP_CYCLE mode=full deleted={deleted_count} "
                f"used_gb={self._get_total_storage(self._list_segment_files()) / (1024 ** 3):.2f} "
                f"elapsed_ms={(time.monotonic() - cycle_t0) * 1000:.1f}"
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
