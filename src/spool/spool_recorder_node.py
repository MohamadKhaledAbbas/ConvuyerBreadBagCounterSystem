"""
ROS2 Spool Recorder Node.

Subscribes to H.264 frames from RTSP client and writes to segment files
on disk for later processing and recovery.

Flow:
    RTSP Client → /rtsp_image_ch_0 (H26XFrame) → SpoolRecorderNode → disk segments

Features:
- H.264 NAL parsing for SPS/PPS extraction
- IDR-aligned segment rotation
- Atomic segment writes
- Frame indexing with timestamps
"""

import time
import threading
from typing import Optional
from dataclasses import dataclass

from src.utils.platform import is_rdk_platform
from src.utils.AppLogging import logger
from src.spool.segment_io import SegmentWriter, FrameRecord
from src.spool.h264_nal import extract_sps_pps, is_idr_frame
from src.spool.spool_utils import format_structured_log

# ROS2 imports (conditional)
if is_rdk_platform():
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

    try:
        from hobot_cv_msgs.msg import H26XFrame
        HAS_H26X_MSG = True
    except ImportError:
        HAS_H26X_MSG = False
        logger.warning("[SpoolRecorder] hobot_cv_msgs not available")
else:
    Node = object
    HAS_H26X_MSG = False


@dataclass
class RecorderConfig:
    """Configuration for spool recorder."""
    topic: str = "/rtsp_image_ch_0"
    spool_dir: str = "/tmp/spool"
    segment_duration: float = 5.0
    max_segment_duration: float = 10.0
    write_metadata: bool = True
    qos_depth: int = 10
    channel_id: int = 0


class SpoolRecorderNode(Node if is_rdk_platform() else object):
    """
    ROS2 node that records H.264 frames from RTSP to disk segments.

    Subscribes to H26XFrame messages and writes them to binary segment
    files using the segment_io writer. Handles:
    - SPS/PPS extraction and caching
    - IDR frame detection for segment boundaries
    - Timestamping with DTS/PTS preservation
    """

    def __init__(self, config: Optional[RecorderConfig] = None):
        """
        Initialize recorder node.

        Args:
            config: Recorder configuration
        """
        self.config = config or RecorderConfig()

        if is_rdk_platform():
            super().__init__('spool_recorder_node')

            # Configure QoS for reliable delivery
            qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                depth=self.config.qos_depth
            )

            # Create segment writer
            self.writer = SegmentWriter(
                spool_dir=self.config.spool_dir,
                segment_duration=self.config.segment_duration,
                max_segment_duration=self.config.max_segment_duration,
                write_metadata=self.config.write_metadata
            )
            self.writer.start()

            # State tracking
            self._frame_index = 0
            self._cached_sps: Optional[bytes] = None
            self._cached_pps: Optional[bytes] = None
            self._last_idr_time = 0.0

            # Statistics
            self._frames_received = 0
            self._idr_count = 0
            self._start_time = time.time()

            # Create subscription if message type available
            if HAS_H26X_MSG:
                self._subscription = self.create_subscription(
                    H26XFrame,
                    self.config.topic,
                    self._frame_callback,
                    qos
                )
                logger.info(
                    f"[SpoolRecorder] Subscribed to {self.config.topic}, "
                    f"writing to {self.config.spool_dir}"
                )
            else:
                self._subscription = None
                logger.warning("[SpoolRecorder] H26XFrame not available, recorder disabled")

        else:
            logger.warning("[SpoolRecorder] Not on RDK platform, recorder disabled")
            self.writer = None
            self._subscription = None

    def _frame_callback(self, msg):
        """
        Handle incoming H.264 frame.

        Args:
            msg: H26XFrame message
        """
        try:
            # Extract frame data
            data = bytes(msg.data)
            width = msg.width
            height = msg.height

            # Handle timestamps - prefer msg timestamps, fall back to current time
            now_ns = time.time_ns()

            if hasattr(msg, 'dts') and msg.dts > 0:
                dts_ns = msg.dts
            else:
                dts_ns = now_ns

            if hasattr(msg, 'pts') and msg.pts > 0:
                pts_ns = msg.pts
            else:
                pts_ns = now_ns

            # Determine encoding type
            encoding = "H264"
            if hasattr(msg, 'encoding'):
                enc_bytes = bytes(msg.encoding) if hasattr(msg.encoding, '__iter__') else msg.encoding
                if isinstance(enc_bytes, bytes):
                    encoding = enc_bytes.rstrip(b'\x00').decode('utf-8', errors='ignore')
                elif isinstance(enc_bytes, str):
                    encoding = enc_bytes

            # Check for IDR and extract SPS/PPS
            has_idr = is_idr_frame(data)
            sps, pps = extract_sps_pps(data)

            if sps:
                self._cached_sps = sps
            if pps:
                self._cached_pps = pps

            if has_idr:
                self._idr_count += 1
                self._last_idr_time = time.time()

            # Update writer's cached parameters
            self.writer.update_sps_pps(self._cached_sps, self._cached_pps)

            # Create frame record
            record = FrameRecord(
                index=self._frame_index,
                width=width,
                height=height,
                dts_sec=dts_ns // 1_000_000_000,
                dts_nsec=dts_ns % 1_000_000_000,
                pts_sec=pts_ns // 1_000_000_000,
                pts_nsec=pts_ns % 1_000_000_000,
                encoding=encoding,
                data=data
            )

            # Write to segment
            success = self.writer.write_frame(record, has_idr=has_idr)

            if success:
                self._frames_received += 1
                self._frame_index += 1

                # Periodic logging
                if self._frames_received % 300 == 0:
                    elapsed = time.time() - self._start_time
                    fps = self._frames_received / elapsed if elapsed > 0 else 0
                    logger.info(
                        format_structured_log(
                            "recorder_stats",
                            frame=self._frame_index,
                            frames_total=self._frames_received,
                            idr_count=self._idr_count,
                            fps=fps,
                            segments=self.writer.segments_completed
                        )
                    )

        except Exception as e:
            logger.error(f"[SpoolRecorder] Error processing frame: {e}")

    def get_stats(self) -> dict:
        """Get recorder statistics."""
        elapsed = time.time() - self._start_time
        return {
            'frames_received': self._frames_received,
            'frame_index': self._frame_index,
            'idr_count': self._idr_count,
            'segments_completed': self.writer.segments_completed if self.writer else 0,
            'total_bytes': self.writer.total_bytes_written if self.writer else 0,
            'elapsed_seconds': elapsed,
            'avg_fps': self._frames_received / elapsed if elapsed > 0 else 0
        }

    def stop(self):
        """Stop the recorder and close writer."""
        if self.writer:
            self.writer.close()
        logger.info("[SpoolRecorder] Stopped")


def create_recorder_node(config: Optional[RecorderConfig] = None) -> Optional[SpoolRecorderNode]:
    """
    Factory function to create recorder node.

    Args:
        config: Recorder configuration

    Returns:
        SpoolRecorderNode if on RDK platform, None otherwise
    """
    if not is_rdk_platform():
        logger.info("[SpoolRecorder] Not on RDK platform, skipping recorder creation")
        return None

    if not HAS_H26X_MSG:
        logger.warning("[SpoolRecorder] H26XFrame message type not available")
        return None

    return SpoolRecorderNode(config)


# Standalone runner for testing
def main():
    """Run recorder node standalone."""
    if not is_rdk_platform():
        print("SpoolRecorderNode requires RDK platform")
        return

    rclpy.init()

    config = RecorderConfig(
        topic="/rtsp_image_ch_0",
        spool_dir="/tmp/spool",
        segment_duration=5.0
    )

    node = SpoolRecorderNode(config)

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
