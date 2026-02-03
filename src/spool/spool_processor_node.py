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
"""

import threading
import time
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

# ROS2 imports (conditional)
if is_rdk_platform():
    import rclpy
    from rclpy.node import Node # type: ignore
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy # type: ignore

    try:
        from hobot_cv_msgs.msg import H26XFrame # type: ignore
        HAS_H26X_MSG = True
    except ImportError:
        HAS_H26X_MSG = False
        logger.warning("[SpoolProcessor] hobot_cv_msgs not available")
else:
    Node = object
    HAS_H26X_MSG = False


class PlaybackMode(Enum):
    """Playback mode for processor."""
    REALTIME = "realtime"       # Match original recording FPS
    FAST = "fast"               # Process as fast as possible
    ADAPTIVE = "adaptive"       # Adjust speed based on queue depth
    CATCHUP = "catchup"         # Fast until caught up, then realtime


@dataclass
class ProcessorConfig:
    """Configuration for spool processor."""
    spool_dir: str = "/tmp/spool"
    output_topic: str = "/spool_image_ch_0"
    state_file: str = "/tmp/spool/processor_state.json"
    playback_mode: PlaybackMode = PlaybackMode.ADAPTIVE
    base_fps: float = 30.0
    qos_depth: int = 10
    state_save_interval: float = 10.0
    enable_retention: bool = True
    retention_config: Optional[RetentionConfig] = None


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

            # Configure QoS
            qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
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

            self._last_state_save = time.time()

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

            logger.info(
                f"[SpoolProcessor] Initialized: {self.config.spool_dir} → "
                f"{self.config.output_topic}, mode={self.config.playback_mode.value}"
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
            msg.data = list(record.data)

            # Set timestamps
            if hasattr(msg, 'dts'):
                msg.dts = record.dts_ns
            if hasattr(msg, 'pts'):
                msg.pts = record.pts_ns

            # Set encoding
            if hasattr(msg, 'encoding'):
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

    def _process_loop(self):
        """Main processing loop."""
        logger.info("[SpoolProcessor] Processing started")

        # Determine starting position
        start_segment = None
        if self.state.last_segment >= 0:
            start_segment = self.state.last_segment
            logger.info(f"[SpoolProcessor] Resuming from segment {start_segment}")

        while not self._stop_event.is_set():
            try:
                # Get available segments
                segments = self.reader.list_segments()

                if not segments:
                    # No segments, wait and retry
                    self._stop_event.wait(1.0)
                    continue

                # Find starting point
                if start_segment is not None:
                    segments = [s for s in segments if s >= start_segment]
                    start_segment = None

                if not segments:
                    self._stop_event.wait(1.0)
                    continue

                # Process each segment
                for segment_num in segments:
                    if self._stop_event.is_set():
                        break

                    logger.debug(f"[SpoolProcessor] Processing segment {segment_num}")

                    frame_count = 0
                    for record in self.reader.read_single_segment(segment_num):
                        if self._stop_event.is_set():
                            break

                        # Pace based on mode
                        if self.config.playback_mode != PlaybackMode.FAST:
                            queue_depth = self._get_queue_depth()
                            self.pacer.calculate_fps(queue_depth)
                            self.pacer.wait_for_next_frame()

                        # Publish frame
                        if self._publish_frame(record):
                            self.state.update(segment_num, record.index)
                            frame_count += 1

                        # Periodic state save
                        if (time.time() - self._last_state_save) >= self.config.state_save_interval:
                            save_processor_state(self.state, self.config.state_file)
                            self._last_state_save = time.time()

                    # Segment complete
                    self.state.complete_segment(segment_num)
                    self._segments_processed += 1

                    # Mark for retention
                    if self.retention:
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
                    self._last_state_save = time.time()

                # Brief wait before checking for new segments
                self._stop_event.wait(0.5)

            except Exception as e:
                logger.error(f"[SpoolProcessor] Processing error: {e}")
                self._stop_event.wait(1.0)

        logger.info("[SpoolProcessor] Processing stopped")

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
