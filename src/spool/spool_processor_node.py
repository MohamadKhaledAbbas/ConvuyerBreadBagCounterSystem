"""
ROS2 Spool Processor Node.

Reads H.264 frames from disk segments and publishes to hobot-codec
for decoding to NV12 format.

Flow:
    Disk segments → SpoolProcessorNode → /spool_image_ch_0 (H26XFrame) → hobot-codec

Features:
- Adaptive pacing to match recording rate
- State persistence for crash recovery
- Segment tracking for retention
- Configurable playback modes (realtime, fast-forward, catchup)
- Pipeline-wide sentinel power-save mode
- Cross-process status file for health endpoint monitoring
"""

import json
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.spool.retention import RetentionPolicy, RetentionConfig
from src.spool.segment_io import SegmentReader, FrameRecord
from src.spool.spool_utils import (
    ProcessorState,
    save_processor_state,
    load_processor_state,
    AdaptivePacer,
    format_structured_log
)
from src.utils.AppLogging import logger
from src.utils.platform import is_rdk_platform

# Pipeline-wide power-save coordination (reads throttle mode from main app)
try:
    from src.app.pipeline_throttle_state import read_throttle_state
except ImportError:
    # Fallback if module not available — always full speed
    def read_throttle_state(**_kwargs):  # type: ignore[misc]
        return "full", 1.0

# ROS2 imports (conditional)
if is_rdk_platform():
    import rclpy
    from rclpy.node import Node # type: ignore
    from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy # type: ignore

    try:
        from builtin_interfaces.msg import Time as RosTime  # type: ignore
    except ImportError:
        RosTime = None

    try:
        from img_msgs.msg import H26XFrame # type: ignore
        HAS_H26X_MSG = True
    except ImportError:
        HAS_H26X_MSG = False
        logger.warning("[SpoolProcessor] img_msgs not available")
else:
    Node = object
    HAS_H26X_MSG = False
    RosTime = None


class PlaybackMode(Enum):
    """Playback mode for processor."""
    REALTIME = "realtime"       # Match original recording FPS
    FAST = "fast"               # Process as fast as possible
    ADAPTIVE = "adaptive"       # Adjust speed based on queue depth
    CATCHUP = "catchup"         # Fast until caught up, then realtime


# ── Cross-process status file ────────────────────────────────────────
# Written by the processor's _process_loop every few seconds so the
# FastAPI health endpoint (separate process) can report spool stats.
SPOOL_PROCESSOR_STATUS_FILE = "/tmp/spool_processor_status.json"
_STATUS_WRITE_INTERVAL_S = 5.0  # write status file every 5 seconds


class RollingFPSCounter:
    """Thread-safe rolling FPS counter using a fixed time window.

    Maintains a deque of recent frame timestamps and computes FPS
    as count / window.  Used by the spool processor to report current
    throughput rather than a lifetime average.
    """

    __slots__ = ("_timestamps", "_window_s")

    def __init__(self, window_s: float = 10.0):
        self._timestamps: deque = deque()
        self._window_s = window_s

    def tick(self) -> None:
        """Record that a frame was published right now."""
        now = time.monotonic()
        self._timestamps.append(now)
        # Evict stale entries outside the window
        cutoff = now - self._window_s
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()

    def fps(self) -> float:
        """Current FPS within the rolling window."""
        if len(self._timestamps) < 2:
            return 0.0
        now = time.monotonic()
        cutoff = now - self._window_s
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()
        if len(self._timestamps) < 2:
            return 0.0
        span = self._timestamps[-1] - self._timestamps[0]
        if span <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / span


@dataclass
class ProcessorConfig:
    """Configuration for spool processor."""
    spool_dir: str = "/tmp/spool"
    output_topic: str = "/spool_image_ch_0"
    state_file: str = "/tmp/spool/processor_state.json"
    playback_mode: PlaybackMode = PlaybackMode.FAST  # Process as fast as possible to catch up
    base_fps: float = 30.0
    min_frame_interval_ms: float = 33.0  # ~30fps max (1000/33 = 30.3 fps) - prevents CPU overload
    qos_depth: int = 10
    state_save_interval: float = 10.0
    enable_retention: bool = True
    delete_processed_segments: bool = True  # Delete segments after processing to save disk space
    retention_config: Optional[RetentionConfig] = None
    sentinel_wake_buffer_segments: int = 3  # Segments to keep before wake point for catch-up


class SpoolProcessorNode(Node if is_rdk_platform() else object):
    """
    ROS2 node that reads segments and publishes H.264 frames.

    Reads frames from disk segments and publishes them for decoding.
    Supports multiple playback modes and crash recovery via state persistence.
    """

    def __init__(self, config: Optional[ProcessorConfig] = None):
        """
        Initialize processor node.

        Args:
            config: Processor configuration
        """
        self.config = config or ProcessorConfig()

        if is_rdk_platform():
            super().__init__('spool_processor_node')

            # Configure QoS - use RELIABLE for spool pipeline consistency
            qos = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=self.config.qos_depth
            )

            # Create publisher
            if HAS_H26X_MSG:
                self._publisher = self.create_publisher(
                    H26XFrame,
                    self.config.output_topic,
                    qos
                )
            else:
                self._publisher = None
                logger.warning("[SpoolProcessor] H26XFrame not available, publishing disabled")

            # Create segment reader
            self.reader = SegmentReader(
                spool_dir=self.config.spool_dir
            )

            # Adaptive pacing
            self.pacer = AdaptivePacer(
                base_fps=self.config.base_fps,
                min_fps=15.0,
                max_fps=60.0,
                target_queue_depth=2
            )

            # State persistence
            self.state = load_processor_state(self.config.state_file)
            if self.state is None:
                self.state = ProcessorState(session_start_time=time.time())

            self._last_state_save = time.monotonic()

            # Retention management
            self.retention: Optional[RetentionPolicy] = None
            if self.config.enable_retention:
                ret_config = self.config.retention_config or RetentionConfig()
                self.retention = RetentionPolicy(
                    spool_dir=self.config.spool_dir,
                    config=ret_config
                )
                self.retention.start()

            # Processing thread
            self._running = False
            self._stop_event = threading.Event()
            self._process_thread: Optional[threading.Thread] = None

            # Statistics
            self._frames_published = 0
            self._segments_processed = 0
            self._start_time = time.time()

            # Rolling FPS counter for current throughput (not lifetime avg)
            self._rolling_fps = RollingFPSCounter(window_s=10.0)

            # Status file tracking (for health endpoint cross-process visibility)
            self._last_status_write = 0.0
            self._sentinel_active_flag = False
            self._sentinel_frames_total = 0

            # Probe H26XFrame message capabilities once at init instead of
            # calling hasattr() on every frame in the hot publish path.
            if HAS_H26X_MSG:
                _probe = H26XFrame()
                self._msg_has_dts = hasattr(_probe, 'dts')
                self._msg_has_pts = hasattr(_probe, 'pts')
                self._msg_has_encoding = hasattr(_probe, 'encoding')
                del _probe
            else:
                self._msg_has_dts = False
                self._msg_has_pts = False
                self._msg_has_encoding = False

            logger.info(
                f"[SpoolProcessor] Initialized: {self.config.spool_dir} → "
                f"{self.config.output_topic}, mode={self.config.playback_mode.value}, "
                f"min_interval={self.config.min_frame_interval_ms}ms (~{1000/self.config.min_frame_interval_ms:.0f}fps max), "
                f"delete_after_process={self.config.delete_processed_segments}"
            )

        else:
            logger.warning("[SpoolProcessor] Not on RDK platform, processor disabled")
            self._publisher = None
            self.reader = None
            self.pacer = None
            self.state = None
            self.retention = None

    def _publish_frame(self, record: FrameRecord) -> bool:
        """
        Publish a frame record as H26XFrame message.

        Args:
            record: Frame record to publish

        Returns:
            True if published successfully
        """
        if self._publisher is None:
            return False

        try:
            msg = H26XFrame()
            msg.width = record.width
            msg.height = record.height

            # Pass bytes directly — ROS2 sequence<uint8> accepts bytes.
            # PERF: list(record.data) converted every byte to a Python int
            # object, allocating ~28 bytes each.  For a 100 KB H.264 frame
            # that is a 2.7 MB temporary list created 30× per second.
            msg.data = record.data

            # Set timestamps using cached capability flags (probed once at init)
            if self._msg_has_dts and RosTime is not None:
                dts_time = RosTime()
                dts_time.sec = record.dts_sec
                dts_time.nanosec = record.dts_nsec
                msg.dts = dts_time

            if self._msg_has_pts and RosTime is not None:
                pts_time = RosTime()
                pts_time.sec = record.pts_sec
                pts_time.nanosec = record.pts_nsec
                msg.pts = pts_time

            # Set encoding
            if self._msg_has_encoding:
                enc_bytes = record.encoding.encode('utf-8')[:12].ljust(12, b'\x00')
                msg.encoding = list(enc_bytes)

            self._publisher.publish(msg)
            self._frames_published += 1
            return True

        except Exception as e:
            logger.error(f"[SpoolProcessor] Error publishing frame: {e}")
            return False

    def _get_queue_depth(self) -> int:
        """Get number of pending segments to process."""
        segments = self.reader.list_segments()
        current = self.reader.get_current_segment()

        if current < 0:
            return len(segments)

        return sum(1 for s in segments if s > current)

    def _write_status_file(
        self,
        last_processed_segment: int,
        sentinel_active: bool,
        sentinel_frames_sent: int,
    ) -> None:
        """Write processor status to a shared JSON file for the health endpoint.

        Called periodically from _process_loop (~every 5 s).  The FastAPI
        health endpoint (separate process) reads this file to display
        spool processor statistics including time-behind-recorder and FPS.

        The write is atomic (tmp + os.replace) so readers never see
        partial data.
        """
        now_mono = time.monotonic()
        if now_mono - self._last_status_write < _STATUS_WRITE_INTERVAL_S:
            return
        self._last_status_write = now_mono

        try:
            segments = self.reader.list_segments()
            latest_on_disk = segments[-1] if segments else -1
            segments_behind = max(0, latest_on_disk - last_processed_segment) if latest_on_disk >= 0 else 0

            # Estimate time behind recorder: segments_behind * segment_duration
            # RecorderConfig.segment_duration defaults to 5.0 s.
            segment_duration_s = 5.0
            time_behind_s = segments_behind * segment_duration_s

            current_fps = self._rolling_fps.fps()

            status = {
                "timestamp": time.time(),
                "sentinel_active": sentinel_active,
                "sentinel_frames_sent": sentinel_frames_sent,
                "frames_published": self._frames_published,
                "segments_processed": self._segments_processed,
                "last_processed_segment": last_processed_segment,
                "latest_recorder_segment": latest_on_disk,
                "segments_on_disk": len(segments),
                "segments_behind": segments_behind,
                "time_behind_recorder_s": round(time_behind_s, 1),
                "current_fps": round(current_fps, 1),
                "avg_fps": round(
                    self._frames_published / max(time.time() - self._start_time, 1), 1
                ),
                # Retention mode so health endpoint can display it
                "retention_idle_mode": self.retention._idle_mode if self.retention else False,
                "retention_segments_kept": (
                    self.retention.config.idle_max_segments
                    if (self.retention and self.retention._idle_mode)
                    else (self.retention.config.min_segments_keep if self.retention else 5)
                ),
            }

            tmp_path = SPOOL_PROCESSOR_STATUS_FILE + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(status, f)
            os.replace(tmp_path, SPOOL_PROCESSOR_STATUS_FILE)

        except Exception as e:
            logger.debug(f"[SpoolProcessor] Status file write failed: {e}")

    @staticmethod
    def _cleanup_status_file() -> None:
        """Remove the status file on shutdown (best-effort)."""
        try:
            if os.path.exists(SPOOL_PROCESSOR_STATUS_FILE):
                os.unlink(SPOOL_PROCESSOR_STATUS_FILE)
        except OSError:
            pass

    def _process_loop(self):
        """Main processing loop with pipeline-wide sentinel power-save."""
        logger.info("[SpoolProcessor] Processing started")

        # ── Persistent cursor: only process segments STRICTLY AFTER this ──
        # This is the key fix: prevents re-processing a segment that was
        # already read+deleted in a previous batch when the cached segment
        # list still contains the old number.
        _last_processed_segment = -1

        # Determine starting position from persisted state
        if self.state.last_segment >= 0:
            _last_processed_segment = self.state.last_segment
            logger.info(
                f"[SpoolProcessor] Resuming: cursor after segment "
                f"{_last_processed_segment}"
            )

        # For REALTIME mode, track first frame timestamp
        first_frame_timestamp_ns = None
        first_frame_wall_time = None

        # For FAST mode, track last frame time for minimum interval throttling
        last_frame_time = None
        # Cache the minimum interval to avoid per-frame division
        fast_min_interval_s = self.config.min_frame_interval_ms / 1000.0

        # --- Frame-gap detection ---
        # Track DTS of the last published frame.  If the gap between
        # consecutive published DTS values exceeds 2× the expected frame
        # interval we log a FRAME_GAP event so storage stalls / dropped
        # segments become clearly visible in the logs.
        _last_dts_ns: int = 0
        _expected_interval_ns: int = int(1_000_000_000 / max(self.config.base_fps, 1))
        _gap_threshold_ns: int = _expected_interval_ns * 3  # 3× to avoid false positives on jitter

        # --- Diagnostic counters ---
        _diag_batch_num = 0

        # ── Pipeline-wide sentinel mode state ────────────────────────────
        # When the main app (ConveyorCounterApp) switches to DEGRADED mode
        # it writes {"mode": "degraded"} to /tmp/pipeline_throttle.json.
        # We read this file periodically and switch to sentinel mode:
        #   • Publish 1 frame from the LATEST segment every sentinel_interval
        #   • hobot_codec only decodes these sentinel frames (VPU ~6% load)
        #   • Main app detects bags on sentinel frames → wake → resume FAST
        _sentinel_active = False
        _sentinel_cursor_saved = -1       # cursor when we entered sentinel mode
        _sentinel_check_interval_s = 2.0  # how often to poll the state file
        _last_sentinel_check = 0.0
        _sentinel_frames_sent = 0
        _sentinel_log_interval_s = 60.0   # log sentinel status every 60s
        _last_sentinel_log = 0.0
        sentinel_interval_s = 1.0         # current sentinel interval (updated from state file)

        while not self._stop_event.is_set():
            try:
                # ── Check pipeline-wide throttle state ───────────────────
                now_mono = time.monotonic()
                if now_mono - _last_sentinel_check >= _sentinel_check_interval_s:
                    _last_sentinel_check = now_mono
                    throttle_mode, sentinel_interval_s = read_throttle_state()

                    # ── Transition: FULL → sentinel (DEGRADED) ──────────
                    if throttle_mode == "degraded" and not _sentinel_active:
                        _sentinel_active = True
                        _sentinel_cursor_saved = _last_processed_segment
                        _sentinel_frames_sent = 0
                        if self.retention:
                            self.retention.set_idle_mode(True)
                        logger.info(
                            f"[SpoolProcessor] SENTINEL_ENTER | "
                            f"cursor_saved={_sentinel_cursor_saved} | "
                            f"interval={sentinel_interval_s:.1f}s | "
                            f"Pipeline-wide power save active — "
                            f"publishing 1 probe frame every {sentinel_interval_s:.1f}s"
                        )

                    # ── Transition: sentinel (DEGRADED) → FULL ──────────
                    elif throttle_mode == "full" and _sentinel_active:
                        _sentinel_active = False
                        if self.retention:
                            self.retention.set_idle_mode(False)

                        # Skip idle segments: advance cursor to near-latest
                        # to avoid reprocessing hours of empty belt footage.
                        segments = self.reader.list_segments()
                        buf = self.config.sentinel_wake_buffer_segments
                        if segments:
                            latest = segments[-1]
                            # Keep `buf` segments before latest for catch-up
                            resume_from = max(
                                _sentinel_cursor_saved,
                                latest - buf
                            )

                            # Count how many idle segments we're skipping
                            skipped = [
                                s for s in segments
                                if _sentinel_cursor_saved < s < resume_from
                            ]

                            # Delete skipped idle segments to free disk space
                            if skipped and self.retention:
                                for s in skipped:
                                    self.retention.delete_processed_immediately(s)
                                self.reader.invalidate_cache()

                            _last_processed_segment = resume_from

                            logger.info(
                                f"[SpoolProcessor] SENTINEL_EXIT → FULL | "
                                f"cursor {_sentinel_cursor_saved} → {resume_from} | "
                                f"skipped_idle={len(skipped)} segments | "
                                f"sentinel_frames_sent={_sentinel_frames_sent} | "
                                f"Resuming FAST sequential processing"
                            )
                        else:
                            logger.info(
                                f"[SpoolProcessor] SENTINEL_EXIT → FULL | "
                                f"no segments on disk | "
                                f"sentinel_frames_sent={_sentinel_frames_sent}"
                            )

                        # Reset frame-gap detection baseline after sentinel gap
                        _last_dts_ns = 0

                # ── Sentinel mode: publish probe frame from latest segment ──
                if _sentinel_active:
                    segments = self.reader.list_segments()
                    if segments and len(segments) >= 2:
                        # Use second-to-last segment (latest complete; the
                        # very last may still be actively written by recorder)
                        sentinel_seg = segments[-2]
                        record = self.reader.read_first_record(sentinel_seg)
                        if record is not None:
                            if self._publish_frame(record):
                                _sentinel_frames_sent += 1

                    # Periodic sentinel status log
                    if now_mono - _last_sentinel_log >= _sentinel_log_interval_s:
                        _last_sentinel_log = now_mono
                        seg_count = len(segments) if segments else 0
                        logger.info(
                            f"[SpoolProcessor] SENTINEL_STATUS | "
                            f"probe_frames_sent={_sentinel_frames_sent} | "
                            f"segments_on_disk={seg_count} | "
                            f"cursor_saved={_sentinel_cursor_saved} | "
                            f"interval={sentinel_interval_s:.1f}s"
                        )

                    # Sleep for sentinel interval (interruptible by stop event)
                    self._write_status_file(
                        _last_processed_segment, _sentinel_active, _sentinel_frames_sent,
                    )
                    self._stop_event.wait(sentinel_interval_s)
                    continue

                # ── Normal sequential processing (FULL mode) ─────────────
                # Get available segments (fresh scan — cache was invalidated
                # after the last delete, or has naturally expired).
                segments = self.reader.list_segments()

                if not segments:
                    # No segments, wait and retry
                    self._stop_event.wait(1.0)
                    continue

                # ── FIX: always filter out already-processed segments ──
                segments = [s for s in segments if s > _last_processed_segment]


                if not segments:
                    self._stop_event.wait(1.0)
                    continue

                _diag_batch_num += 1
                logger.debug(
                    f"[SpoolProcessor] BATCH #{_diag_batch_num} | "
                    f"count={len(segments)} | "
                    f"range={segments[0]}..{segments[-1]} | "
                    f"cursor_after={_last_processed_segment}"
                )

                # Process each segment
                for seg_idx, segment_num in enumerate(segments):
                    if self._stop_event.is_set():
                        break

                    # ── Double-check guard (belt-and-suspenders) ──
                    if segment_num <= _last_processed_segment:
                        logger.warning(
                            f"[SpoolProcessor] SEG_SKIP_DUPLICATE | "
                            f"seg={segment_num} <= cursor={_last_processed_segment}"
                        )
                        continue

                    seg_t0 = time.monotonic()

                    frame_count = 0
                    for record in self.reader.read_single_segment(segment_num):
                        if self._stop_event.is_set():
                            break

                        # Pace based on mode
                        if self.config.playback_mode == PlaybackMode.REALTIME:
                            # Use actual frame timestamps for pacing
                            frame_timestamp_ns = record.dts_sec * 1_000_000_000 + record.dts_nsec

                            if first_frame_timestamp_ns is None:
                                # First frame - establish baseline
                                first_frame_timestamp_ns = frame_timestamp_ns
                                first_frame_wall_time = time.monotonic()
                            else:
                                # Calculate how long to wait based on timestamp difference
                                elapsed_since_first_ns = frame_timestamp_ns - first_frame_timestamp_ns
                                elapsed_since_first_s = elapsed_since_first_ns / 1_000_000_000.0

                                wall_time_elapsed = time.monotonic() - first_frame_wall_time
                                time_to_wait = elapsed_since_first_s - wall_time_elapsed

                                if time_to_wait > 0:
                                    time.sleep(time_to_wait)

                        elif self.config.playback_mode == PlaybackMode.FAST:
                            # FAST mode: enforce minimum interval to prevent CPU overload
                            if last_frame_time is not None:
                                elapsed = time.monotonic() - last_frame_time
                                remaining = fast_min_interval_s - elapsed

                                if remaining > 0:
                                    time.sleep(remaining)

                            last_frame_time = time.monotonic()

                        else:
                            # ADAPTIVE or CATCHUP mode - use FPS-based pacing
                            queue_depth = self._get_queue_depth()
                            self.pacer.calculate_fps(queue_depth)
                            self.pacer.wait_for_next_frame()

                        # Publish frame
                        if self._publish_frame(record):
                            self._rolling_fps.tick()
                            self.state.update(segment_num, record.index)
                            frame_count += 1

                            # ── Frame-gap detection ──
                            dts_ns = record.dts_sec * 1_000_000_000 + record.dts_nsec
                            if _last_dts_ns > 0 and dts_ns > _last_dts_ns:
                                gap_ns = dts_ns - _last_dts_ns
                                if gap_ns > _gap_threshold_ns:
                                    gap_ms = gap_ns / 1_000_000
                                    logger.warning(
                                        f"[SpoolProcessor] FRAME_GAP | "
                                        f"gap={gap_ms:.0f}ms "
                                        f"(expected<{_expected_interval_ns / 1_000_000:.0f}ms) | "
                                        f"seg={segment_num} frame={record.index}"
                                    )
                            _last_dts_ns = dts_ns

                        # Periodic state save (monotonic clock avoids NTP jumps)
                        now_mono = time.monotonic()
                        if (now_mono - self._last_state_save) >= self.config.state_save_interval:
                            save_processor_state(self.state, self.config.state_file)
                            self._last_state_save = now_mono

                    seg_elapsed_ms = (time.monotonic() - seg_t0) * 1000


                    # ── Advance cursor BEFORE delete ──
                    _last_processed_segment = segment_num

                    # Segment complete
                    self.state.complete_segment(segment_num)
                    self._segments_processed += 1

                    # Delete processed segment immediately if configured
                    if self.retention and self.config.delete_processed_segments:
                        self.retention.delete_processed_immediately(segment_num)
                        # Invalidate cached segment list so the next
                        # list_segments() call does a fresh disk scan and
                        # won't return this just-deleted segment.
                        self.reader.invalidate_cache()
                    elif self.retention:
                        # Otherwise just mark for later cleanup
                        self.retention.mark_processed(segment_num)

                    logger.info(
                        format_structured_log(
                            "segment_complete",
                            segment=segment_num,
                            frames=frame_count,
                            fps=self.pacer.get_current_fps() if self.pacer else 0
                        )
                    )

                    # Save state after each segment
                    save_processor_state(self.state, self.config.state_file)
                    self._last_state_save = time.monotonic()

                    # Write status file periodically for health endpoint
                    self._write_status_file(
                        _last_processed_segment, _sentinel_active, _sentinel_frames_sent,
                    )


                # Brief wait before checking for new segments
                # Keep this short (50ms) to minimize dead time between
                # segment batches — previously 500ms, which added up to
                # half a second of no-frame gaps at segment boundaries.
                self._stop_event.wait(0.05)

            except Exception as e:
                logger.error(f"[SpoolProcessor] Processing error: {e}")
                self._stop_event.wait(1.0)

        logger.info(
            f"[SpoolProcessor] Processing stopped | "
            f"batches={_diag_batch_num}"
        )

    def start(self):
        """Start processing in background thread."""
        if self._running:
            return

        self._running = True
        self._stop_event.clear()
        self._process_thread = threading.Thread(
            target=self._process_loop,
            name="SpoolProcessor",
            daemon=True
        )
        self._process_thread.start()
        logger.info("[SpoolProcessor] Started processing thread")

    def stop(self, timeout: float = 5.0):
        """Stop processing."""
        if not self._running:
            return

        self._running = False
        self._stop_event.set()

        if self._process_thread is not None:
            self._process_thread.join(timeout=timeout)
            self._process_thread = None

        # Save final state
        if self.state:
            save_processor_state(self.state, self.config.state_file)

        # Stop retention
        if self.retention:
            self.retention.stop()

        # Remove status file so the health endpoint knows we're stopped
        self._cleanup_status_file()

        logger.info(
            f"[SpoolProcessor] Stopped. Processed {self._frames_published} frames, "
            f"{self._segments_processed} segments"
        )

    def get_stats(self) -> dict:
        """Get processor statistics."""
        elapsed = time.time() - self._start_time
        return {
            'frames_published': self._frames_published,
            'segments_processed': self._segments_processed,
            'elapsed_seconds': elapsed,
            'avg_fps': self._frames_published / elapsed if elapsed > 0 else 0,
            'current_segment': self.reader.get_current_segment() if self.reader else -1,
            'queue_depth': self._get_queue_depth() if self.reader else 0,
            'state': {
                'last_segment': self.state.last_segment if self.state else -1,
                'last_frame': self.state.last_frame_index if self.state else -1,
                'total_processed': self.state.total_frames_processed if self.state else 0
            }
        }


def create_processor_node(config: Optional[ProcessorConfig] = None) -> Optional[SpoolProcessorNode]:
    """
    Factory function to create processor node.

    Args:
        config: Processor configuration

    Returns:
        SpoolProcessorNode if on RDK platform, None otherwise
    """
    if not is_rdk_platform():
        logger.info("[SpoolProcessor] Not on RDK platform, skipping processor creation")
        return None

    return SpoolProcessorNode(config)


# Standalone runner
def main():
    """Run processor node standalone."""
    if not is_rdk_platform():
        print("SpoolProcessorNode requires RDK platform")
        return

    rclpy.init()

    config = ProcessorConfig(
        spool_dir="/tmp/spool",
        output_topic="/spool_image_ch_0",
        playback_mode=PlaybackMode.ADAPTIVE,
        base_fps=30.0
    )

    node = SpoolProcessorNode(config)
    node.start()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
