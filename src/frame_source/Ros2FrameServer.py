"""
ROS2 frame server for receiving frames from ROS2 topics.

Subscribes to NV12 images output by hobot-codec decoder and provides
frames for the detection/classification pipeline.

Flow:
    hobot-codec → /nv12_images (HbmNV12Image) → Ros2FrameServer

Frame Outputs:
    - NV12: Raw format for BPU detection (no conversion overhead)
    - BGR: Converted format for classification and visualization

Only available on RDK platform.
"""

import time
import cv2
import numpy as np
import threading
from typing import Iterator, Tuple, Optional, NamedTuple
from dataclasses import dataclass

from src.frame_source.FrameSource import FrameSource
from src.utils.AppLogging import logger


try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from hbmem_msgs.msg import HbmNV12Image
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False


class FrameData(NamedTuple):
    """
    Container for frame data with both NV12 and BGR representations.
    
    Attributes:
        nv12: Raw NV12 data (numpy array, shape: height*1.5 x width)
        bgr: BGR converted frame (numpy array, shape: height x width x 3)
        width: Frame width in pixels
        height: Frame height in pixels
        timestamp_ns: Frame timestamp in nanoseconds
        frame_index: Frame sequence number
    """
    nv12: np.ndarray
    bgr: np.ndarray
    width: int
    height: int
    timestamp_ns: int
    frame_index: int


def nv12_to_bgr(nv12_data: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Convert NV12 YUV data to BGR format.
    
    NV12 Layout:
        - Y plane: height x width bytes (luminance)
        - UV plane: height/2 x width bytes (interleaved chrominance)
        - Total size: height * 1.5 * width bytes
    
    Args:
        nv12_data: Raw NV12 data as 1D numpy array
        width: Frame width
        height: Frame height
        
    Returns:
        BGR image as numpy array (height x width x 3)
    """
    # Reshape to NV12 format: Y plane (height) + UV plane (height/2)
    total_height = int(height * 1.5)
    
    if nv12_data.size != width * total_height:
        # Handle potential data mismatch
        expected_size = width * total_height
        if nv12_data.size < expected_size:
            padded = np.zeros(expected_size, dtype=np.uint8)
            padded[:nv12_data.size] = nv12_data
            nv12_data = padded
        else:
            nv12_data = nv12_data[:expected_size]
    
    yuv_image = nv12_data.reshape(total_height, width)
    bgr = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_NV12)
    
    return bgr


def bgr_to_nv12(bgr: np.ndarray) -> np.ndarray:
    """
    Convert BGR frame to NV12 format.
    
    Args:
        bgr: BGR image (height x width x 3)
        
    Returns:
        NV12 data as numpy array (height*1.5 x width)
    """
    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
    height, width = bgr.shape[:2]
    
    # I420 to NV12 conversion
    y_size = width * height
    u_size = y_size // 4
    
    y = yuv[:height, :].flatten()
    u = yuv[height:height + height // 4, :].flatten()
    v = yuv[height + height // 4:, :].flatten()
    
    # Interleave U and V for NV12
    uv = np.empty(u_size * 2, dtype=np.uint8)
    uv[0::2] = u
    uv[1::2] = v
    
    nv12 = np.concatenate([y, uv])
    return nv12.reshape(int(height * 1.5), width)


@dataclass
class FrameServerConfig:
    """Configuration for ROS2 frame server."""
    topic: str = '/nv12_images'
    timeout: float = 30.0
    qos_depth: int = 10
    target_width: int = 1280
    target_height: int = 720
    convert_to_bgr: bool = True  # Set False for NV12-only mode


class Ros2FrameServer(FrameSource):
    """
    ROS2 frame server that subscribes to NV12 image topics from hobot-codec.
    
    Provides both NV12 (for BPU detection) and BGR (for classification) outputs.
    Detection can use NV12 directly without conversion overhead.
    """
    
    def __init__(
            self, 
            topic: str = '/nv12_images', 
            timeout: float = 30.0,
            config: Optional[FrameServerConfig] = None
    ):
        """
        Initialize ROS2 frame server.
        
        Args:
            topic: ROS2 topic to subscribe to
            timeout: Timeout in seconds waiting for first frame
            config: Full configuration (overrides topic/timeout if provided)
        """
        if not HAS_ROS2:
            raise ImportError(
                "ROS2 not available. Install rclpy and hbmem_msgs packages."
            )
        
        if config:
            self.config = config
        else:
            self.config = FrameServerConfig(topic=topic, timeout=timeout)
        
        self.topic = self.config.topic
        self.timeout = self.config.timeout
        self.running = True
        
        # Frame storage - both NV12 and BGR
        self._last_nv12: Optional[np.ndarray] = None
        self._last_bgr: Optional[np.ndarray] = None
        self._last_width: int = 0
        self._last_height: int = 0
        self._last_timestamp_ns: int = 0
        self._frame_lock = threading.Lock()
        
        self.frame_count = 0
        self._dropped_frames = 0
        
        # Initialize ROS2
        if not rclpy.ok():
            rclpy.init()
        
        self.node = rclpy.create_node('conveyer_frame_server')
        
        # Configure QoS for reliable delivery with some buffering
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=self.config.qos_depth
        )
        
        # Subscribe to NV12 image topic
        self.subscription = self.node.create_subscription(
            HbmNV12Image,
            self.topic,
            self._frame_callback,
            qos
        )
        
        logger.info(f"[Ros2FrameServer] Subscribed to {self.topic}")
    
    def _frame_callback(self, msg: 'HbmNV12Image'):
        """
        Process incoming ROS2 NV12 frame message.
        
        Stores both NV12 (for detection) and BGR (for classification).
        """
        try:
            # Extract dimensions
            width = msg.width
            height = msg.height
            
            # Get NV12 data
            nv12_data = np.frombuffer(msg.data, dtype=np.uint8)
            
            # Get timestamp (ROS2 header or generate)
            if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
                timestamp_ns = (
                    msg.header.stamp.sec * 1_000_000_000 + 
                    msg.header.stamp.nanosec
                )
            else:
                timestamp_ns = time.time_ns()
            
            # Reshape NV12 data
            total_height = int(height * 1.5)
            expected_size = width * total_height
            
            if nv12_data.size >= expected_size:
                nv12_frame = nv12_data[:expected_size].reshape(total_height, width)
            else:
                # Handle undersized data
                logger.warning(
                    f"[Ros2FrameServer] NV12 data size mismatch: "
                    f"{nv12_data.size} < {expected_size}"
                )
                self._dropped_frames += 1
                return
            
            # Convert to BGR if needed
            bgr_frame = None
            if self.config.convert_to_bgr:
                bgr_frame = nv12_to_bgr(nv12_data, width, height)
                
                # Resize if target dimensions differ
                if (width != self.config.target_width or 
                    height != self.config.target_height):
                    bgr_frame = cv2.resize(
                        bgr_frame, 
                        (self.config.target_width, self.config.target_height)
                    )
            
            # Store frame data (thread-safe)
            with self._frame_lock:
                self._last_nv12 = nv12_frame
                self._last_bgr = bgr_frame
                self._last_width = width
                self._last_height = height
                self._last_timestamp_ns = timestamp_ns
                self.frame_count += 1
                
        except Exception as e:
            logger.error(f"[Ros2FrameServer] Error processing frame: {e}")
            self._dropped_frames += 1
    
    def get_frame_data(self) -> Optional[FrameData]:
        """
        Get the latest frame data with both NV12 and BGR.
        
        Returns:
            FrameData with NV12 and BGR representations, or None if no frame
        """
        with self._frame_lock:
            if self._last_nv12 is None:
                return None
            
            return FrameData(
                nv12=self._last_nv12.copy(),
                bgr=self._last_bgr.copy() if self._last_bgr is not None else None,
                width=self._last_width,
                height=self._last_height,
                timestamp_ns=self._last_timestamp_ns,
                frame_index=self.frame_count
            )
    
    def get_nv12_frame(self) -> Optional[Tuple[np.ndarray, int, int]]:
        """
        Get the latest NV12 frame for BPU detection.
        
        No conversion overhead - direct NV12 for detection.
        
        Returns:
            Tuple of (nv12_data, width, height) or None
        """
        with self._frame_lock:
            if self._last_nv12 is None:
                return None
            return (self._last_nv12.copy(), self._last_width, self._last_height)
    
    def frames(self) -> Iterator[Tuple[np.ndarray, float]]:
        """
        Yield BGR frames from ROS2 topic.
        
        This is the standard FrameSource interface yielding BGR frames.
        For NV12 access, use get_nv12_frame() or get_frame_data().
        
        Yields:
            Tuple of (bgr_frame, latency_ms)
        """
        # Wait for first frame
        start_time = time.time()
        while self._last_bgr is None and self.running:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if time.time() - start_time > self.timeout:
                logger.error(
                    f"[Ros2FrameServer] Timeout waiting for frames on {self.topic}"
                )
                return
        
        logger.info(f"[Ros2FrameServer] Started receiving frames")
        
        last_yield_time = None
        prev_frame_count = 0
        
        while self.running:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            
            with self._frame_lock:
                if self._last_bgr is None:
                    continue
                
                # Only yield new frames
                current_count = self.frame_count
                if current_count == prev_frame_count:
                    continue
                
                prev_frame_count = current_count
                frame = self._last_bgr.copy()
            
            current_time = time.perf_counter()
            
            # Calculate inter-frame latency
            if last_yield_time is None:
                latency_ms = 0.0
            else:
                latency_ms = (current_time - last_yield_time) * 1000.0
            
            last_yield_time = current_time
            
            yield (frame, latency_ms)
            
            # Log progress periodically
            if self.frame_count % 300 == 0:
                logger.info(
                    f"[Ros2FrameServer] Frames: {self.frame_count}, "
                    f"dropped: {self._dropped_frames}"
                )
    
    def frames_with_nv12(self) -> Iterator[Tuple[FrameData, float]]:
        """
        Yield frames with both NV12 and BGR data.
        
        Use this when you need NV12 for detection AND BGR for classification.
        
        Yields:
            Tuple of (FrameData, latency_ms)
        """
        # Wait for first frame
        start_time = time.time()
        while self._last_nv12 is None and self.running:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if time.time() - start_time > self.timeout:
                logger.error(
                    f"[Ros2FrameServer] Timeout waiting for frames on {self.topic}"
                )
                return
        
        logger.info(f"[Ros2FrameServer] Started receiving NV12 frames")
        
        last_yield_time = None
        prev_frame_count = 0
        
        while self.running:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            
            frame_data = self.get_frame_data()
            if frame_data is None:
                continue
            
            # Only yield new frames
            if frame_data.frame_index == prev_frame_count:
                continue
            
            prev_frame_count = frame_data.frame_index
            current_time = time.perf_counter()
            
            # Calculate inter-frame latency
            if last_yield_time is None:
                latency_ms = 0.0
            else:
                latency_ms = (current_time - last_yield_time) * 1000.0
            
            last_yield_time = current_time
            
            yield (frame_data, latency_ms)
    
    def cleanup(self):
        """Clean up ROS2 resources."""
        self.running = False
        self.node.destroy_subscription(self.subscription)
        self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        logger.info(
            f"[Ros2FrameServer] Cleanup complete. "
            f"Received: {self.frame_count}, dropped: {self._dropped_frames}"
        )
    
    def get_stats(self) -> dict:
        """Get frame server statistics."""
        return {
            'frames_received': self.frame_count,
            'frames_dropped': self._dropped_frames,
            'topic': self.topic,
            'last_width': self._last_width,
            'last_height': self._last_height
        }


# Alias for factory compatibility
FrameServer = Ros2FrameServer
