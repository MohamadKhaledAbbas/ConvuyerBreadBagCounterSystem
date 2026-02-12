"""
ROS2 Frame Server for receiving frames from ROS2 topics.

Subscribes to NV12 images from hobot-codec decoder and provides
frames for the detection/classification pipeline.

Uses sensor_msgs.msg.Image (standard ROS2 package) for compatibility.

IMPORTANT: rclpy.init() must be called externally before instantiating this class.
Use init_ros2_context() from src.ros2.IPC to initialize the ROS2 context.
"""

import os
import queue
import time
from typing import Iterator, Tuple, Optional

import cv2
import numpy as np
import rclpy
from sensor_msgs.msg import Image
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from src.frame_source.FrameSource import FrameSource
from src.utils.AppLogging import logger


def nv12_to_bgr(nv12_data: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Convert NV12 YUV data to BGR format.

    Args:
        nv12_data: Raw NV12 data as numpy array
        width: Frame width
        height: Frame height

    Returns:
        BGR image as numpy array (height x width x 3)
    """
    yuv_image = nv12_data.reshape(int(height * 1.5), width)
    bgr = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_NV12)
    return bgr


class FrameServer(Node, FrameSource):
    """
    ROS 2 Subscriber that listens for incoming frames and buffers the latest
    frame for consumption by the main logic thread. This node is designed
    to be added to an external SingleThreadedExecutor.

    ACK-free mode - frames are buffered in input_queue and smart degraded mode handles overload.
    """

    def __init__(self, topic: str = '/nv12_images', target_fps: float = 20.0):
        """
        Initialize the ROS2 frame server.

        IMPORTANT: rclpy.init() must be called externally before this class is instantiated.

        Args:
            topic: ROS2 topic to subscribe to
            target_fps: Target frames per second (for logging only)
        """
        super().__init__('frame_server')

        # Support ROS_TARGET_FPS environment variable override
        env_fps = os.getenv('ROS_TARGET_FPS')
        if env_fps:
            try:
                target_fps = float(env_fps)
                logger.info(f"[Ros2FrameServer] Using ROS_TARGET_FPS from environment: {target_fps}")
            except ValueError:
                logger.warning(f"[Ros2FrameServer] Invalid ROS_TARGET_FPS value '{env_fps}', using default {target_fps}")

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=50,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE
        )

        self.subscription = self.create_subscription(
            Image,
            topic,
            self.listener_callback,
            qos)

        # Buffer more frames to reduce frame drops
        self.frame_queue = queue.Queue(maxsize=30)
        self.last_frame_time = time.time()

        # Store target_fps for logging only
        self.target_fps = target_fps

        # Proactive drop threshold (80% of queue size)
        self.proactive_drop_threshold = int(self.frame_queue.maxsize * 0.8)

        # Stats for debugging
        self.frames_received = 0
        self.frames_processed = 0
        self.frames_dropped = 0
        self.last_stats_log_time = time.time()
        self.stats_log_interval = 5.0

        # Timing stats
        self._timing_stats = {'callback': 0, 'reshape': 0, 'nv12_copy': 0, 'count': 0}

        logger.info(f"[Ros2FrameServer] Initialized: topic={topic}, queue_size=30, target_fps={target_fps}")

    def _get_timing_stats_string(self) -> str:
        """Get formatted timing stats string and reset counters."""
        if self._timing_stats['count'] == 0:
            return ""

        count = self._timing_stats['count']
        stats_str = (
            f", avg_callback={self._timing_stats['callback'] / count:.2f}ms"
            f" (reshape={self._timing_stats['reshape'] / count:.2f}ms"
            f", nv12_copy={self._timing_stats['nv12_copy'] / count:.2f}ms)"
        )
        # Reset timing stats
        self._timing_stats = {'callback': 0, 'reshape': 0, 'nv12_copy': 0, 'count': 0}
        return stats_str

    def listener_callback(self, msg):
        """Handle incoming ROS2 image message."""
        now = time.time()
        self.frames_received += 1

        t_callback_start = time.perf_counter()
        self.frames_processed += 1

        # Log stats periodically
        if now - self.last_stats_log_time >= self.stats_log_interval:
            queue_utilization = (self.frame_queue.qsize() / self.frame_queue.maxsize) * 100
            drop_rate = (self.frames_dropped / self.frames_received * 100) if self.frames_received > 0 else 0.0
            timing_stats_str = self._get_timing_stats_string()

            logger.info(
                f"[Ros2FrameServer] Stats: received={self.frames_received}, "
                f"processed={self.frames_processed}, dropped={self.frames_dropped}, "
                f"drop_rate={drop_rate:.2f}%, queue_util={queue_utilization:.1f}%{timing_stats_str}"
            )
            self.last_stats_log_time = now

        t_reshape_start = time.perf_counter()

        expected = msg.height * msg.width * 3 // 2
        img = np.frombuffer(msg.data, dtype=np.uint8)
        if img.size < expected:
            self.get_logger().error(f"Frame size mismatch: got {img.size}, expected {expected}")
            return

        try:
            # NV12 conversion - reshape to NV12 format
            nv12_img = img.reshape((msg.height * 3 // 2, msg.width))
            t_reshape_end = time.perf_counter()

            # Store raw NV12 data ONLY - BGR conversion is done lazily
            t_nv12_copy_start = time.perf_counter()
            nv12_data = nv12_img.copy()
            t_nv12_copy_end = time.perf_counter()

        except Exception as e:
            self.get_logger().error(f"Frame conversion error: {e}")
            return

        # Accumulate timing stats
        t_callback_end = time.perf_counter()
        self._timing_stats['callback'] += (t_callback_end - t_callback_start) * 1000
        self._timing_stats['reshape'] += (t_reshape_end - t_reshape_start) * 1000
        self._timing_stats['nv12_copy'] += (t_nv12_copy_end - t_nv12_copy_start) * 1000
        self._timing_stats['count'] += 1

        latency_ms = (now - self.last_frame_time) * 1000
        self.last_frame_time = now

        # Leaky queue with drop tracking
        queue_size = self.frame_queue.qsize()
        if queue_size >= self.proactive_drop_threshold:
            try:
                self.frame_queue.get_nowait()
                self.frames_dropped += 1
                logger.debug(f"[Ros2FrameServer] Proactive drop at queue_size={queue_size}")
            except queue.Empty:
                pass
        elif self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
                self.frames_dropped += 1
            except queue.Empty:
                pass

        # Enqueue new frame: (nv12_data, latency_ms, frame_index, (height, width))
        frame_size = (msg.height, msg.width)
        self.frame_queue.put((nv12_data, latency_ms, 0, frame_size))

    def frames(self) -> Iterator[Tuple[np.ndarray, float]]:
        """
        Yield frames from the queue.

        Yields:
            Tuple of (bgr_frame, latency_ms) - compatible with FrameSource interface

        Note: For RDK optimization, the original NV12 data is stored in
        self._last_nv12_data and can be accessed by the BPU detector to avoid
        BGR->NV12 reconversion.
        """
        logger.info("[Ros2FrameServer] Starting frame iteration loop")
        frame_count = 0

        # Store last NV12 data for BPU detector optimization
        self._last_nv12_data = None
        self._last_frame_size = None

        while rclpy.ok():
            # Spin once to process ROS2 callbacks (fills the queue)
            rclpy.spin_once(self, timeout_sec=0.001)

            try:
                item = self.frame_queue.get(timeout=0.1)  # Shorter timeout since we're spinning

                if len(item) == 4:
                    nv12_data, latency_ms, frame_index, frame_size = item
                    frame_count += 1

                    if frame_count == 1:
                        logger.info(f"[Ros2FrameServer] First frame yielded! {frame_size[1]}x{frame_size[0]}")
                    elif frame_count % 100 == 0:
                        logger.debug(f"[Ros2FrameServer] Yielded {frame_count} frames, queue size: {self.frame_queue.qsize()}")

                    # Store NV12 data for BPU detector optimization
                    self._last_nv12_data = nv12_data
                    self._last_frame_size = frame_size

                    # Convert NV12 to BGR for pipeline compatibility
                    height, width = frame_size
                    bgr_frame = nv12_to_bgr(nv12_data, width, height)

                    yield bgr_frame, latency_ms

                elif len(item) == 2:
                    # Legacy format
                    frame, latency_ms = item
                    frame_count += 1
                    self._last_nv12_data = None
                    self._last_frame_size = None
                    yield frame, latency_ms
            except queue.Empty:
                continue

    def get_last_nv12_data(self) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int]]]:
        """
        Get the last NV12 frame data for BPU detector optimization.

        Returns:
            Tuple of (nv12_data, frame_size) or (None, None) if not available
        """
        return self._last_nv12_data, self._last_frame_size

    def get_bgr_frame(self, nv12_data: np.ndarray, frame_size: Tuple[int, int]) -> np.ndarray:
        """
        Convert NV12 frame to BGR (lazy conversion).

        Args:
            nv12_data: NV12 frame data
            frame_size: (height, width) tuple

        Returns:
            BGR frame as numpy array
        """
        height, width = frame_size
        return nv12_to_bgr(nv12_data, width, height)

    def cleanup(self):
        """Destroys the node, relying on the main app to shutdown the ROS context."""
        logger.info("[Ros2FrameServer] cleanup called. Destroying node.")
        try:
            self.destroy_node()
        except Exception as e:
            logger.debug(f"[Ros2FrameServer] destroy_node() raised (ignored): {e}")
        logger.info("[Ros2FrameServer] cleanup finished")
