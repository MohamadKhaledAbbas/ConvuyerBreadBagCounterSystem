"""
ROS2 Frame Server for Container Camera.

Subscribes to the /nv12_images_container topic and provides frames
for QR code processing. Reuses the same pattern as the bread camera
frame server with container-specific topic names.

Usage:
    from src.container.frame_source.ContainerFrameServer import ContainerFrameServer
    
    server = ContainerFrameServer()
    for frame, latency_ms in server:
        # Process frame for QR detection
        qr_result = detector.detect(frame)
"""

import time
import threading
from collections import deque
from typing import Iterator, Optional, Tuple
import numpy as np
import cv2

from src.utils.AppLogging import logger
from src.utils.platform import IS_RDK

# Container camera topic
CONTAINER_NV12_TOPIC = '/nv12_images_container'

# Conditionally import ROS2 components
if IS_RDK:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from sensor_msgs.msg import Image
    
    class ContainerFrameServer(Node):
        """
        ROS2 node that subscribes to /nv12_images_container and provides
        frames as a Python iterator for QR code processing.
        
        Features:
        - Bounded queue to prevent memory overflow
        - NV12 to BGR conversion
        - Frame timing and latency tracking
        - Proactive frame drops when queue fills
        
        Attributes:
            queue_size: Maximum frames to buffer
            topic: ROS2 topic to subscribe to
        """
        
        def __init__(
            self,
            topic: str = CONTAINER_NV12_TOPIC,
            queue_size: int = 30,
            node_name: str = 'container_frame_server'
        ):
            """
            Initialize the container frame server.
            
            Args:
                topic: ROS2 topic to subscribe to
                queue_size: Maximum frames to buffer
                node_name: ROS2 node name
            """
            super().__init__(node_name)
            
            self.topic = topic
            self.queue_size = queue_size
            
            # Frame queue (thread-safe deque)
            self._queue = deque(maxlen=queue_size)
            self._lock = threading.Lock()
            
            # NV12 data for BPU optimization (if needed)
            self._last_nv12_data: Optional[bytes] = None
            self._last_frame_size: Optional[Tuple[int, int]] = None
            
            # Statistics
            self._frames_received = 0
            self._frames_dropped = 0
            self._frames_processed = 0
            self._last_stats_time = time.time()
            
            # QoS profile for reliable transport
            qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
                depth=50
            )
            
            # Subscribe to container NV12 topic
            self.subscription = self.create_subscription(
                Image,
                topic,
                self._frame_callback,
                qos
            )
            
            logger.info(
                f"[ContainerFrameServer] Subscribed to {topic}, "
                f"queue_size={queue_size}"
            )
        
        def _frame_callback(self, msg: Image) -> None:
            """
            Handle incoming NV12 frame from ROS2.
            
            Converts NV12 to BGR and adds to queue.
            Implements proactive drops when queue is near full.
            """
            callback_start = time.time()
            self._frames_received += 1
            
            try:
                # Get frame dimensions
                height = msg.height
                width = msg.width
                
                # NV12 has 1.5 bytes per pixel
                expected_size = int(height * 1.5 * width)
                actual_size = len(msg.data)
                
                if actual_size < expected_size:
                    logger.warning(
                        f"[ContainerFrameServer] Frame size mismatch: "
                        f"expected {expected_size}, got {actual_size}"
                    )
                    return
                
                # Reshape NV12 data
                nv12_data = np.frombuffer(msg.data, dtype=np.uint8)
                nv12_frame = nv12_data.reshape((int(height * 1.5), width))
                
                # Store raw NV12 for potential BPU use
                self._last_nv12_data = bytes(msg.data)
                self._last_frame_size = (width, height)
                
                # Convert NV12 to BGR
                bgr_frame = cv2.cvtColor(nv12_frame, cv2.COLOR_YUV2BGR_NV12)
                
                # Calculate latency from message timestamp
                msg_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                latency_ms = (callback_start - msg_time) * 1000 if msg_time > 0 else 0
                
                # Add to queue with proactive drop
                with self._lock:
                    proactive_threshold = int(self.queue_size * 0.8)
                    
                    if len(self._queue) >= proactive_threshold:
                        # Drop oldest frame
                        self._queue.popleft()
                        self._frames_dropped += 1
                    
                    self._queue.append((bgr_frame, latency_ms))
                
            except Exception as e:
                logger.error(f"[ContainerFrameServer] Frame callback error: {e}")
        
        def __iter__(self) -> Iterator[Tuple[np.ndarray, float]]:
            """Iterate over frames from the queue."""
            return self
        
        def __next__(self) -> Tuple[np.ndarray, float]:
            """
            Get the next frame from the queue.
            
            Returns:
                Tuple of (BGR frame, latency_ms)
                
            Raises:
                StopIteration: Never raised (infinite iterator)
            """
            # Spin ROS2 to receive frames
            rclpy.spin_once(self, timeout_sec=0.001)
            
            with self._lock:
                if self._queue:
                    self._frames_processed += 1
                    return self._queue.popleft()
            
            # Return empty frame if queue is empty
            return (np.zeros((1, 1, 3), dtype=np.uint8), 0.0)
        
        def get_frame(self, timeout: float = 0.1) -> Optional[Tuple[np.ndarray, float]]:
            """
            Get a frame with timeout.
            
            Args:
                timeout: Maximum seconds to wait
                
            Returns:
                Tuple of (BGR frame, latency_ms) or None if timeout
            """
            start = time.time()
            
            while time.time() - start < timeout:
                rclpy.spin_once(self, timeout_sec=0.001)
                
                with self._lock:
                    if self._queue:
                        self._frames_processed += 1
                        return self._queue.popleft()
                
                time.sleep(0.001)
            
            return None
        
        def get_stats(self) -> dict:
            """Get frame server statistics."""
            with self._lock:
                queue_len = len(self._queue)
            
            elapsed = time.time() - self._last_stats_time
            fps = self._frames_received / elapsed if elapsed > 0 else 0
            
            return {
                'topic': self.topic,
                'frames_received': self._frames_received,
                'frames_processed': self._frames_processed,
                'frames_dropped': self._frames_dropped,
                'queue_size': queue_len,
                'queue_capacity': self.queue_size,
                'queue_fill_percent': queue_len / self.queue_size * 100,
                'fps': round(fps, 1),
                'drop_rate': (
                    self._frames_dropped / self._frames_received * 100
                    if self._frames_received > 0 else 0
                ),
            }
        
        @property
        def last_nv12_data(self) -> Optional[bytes]:
            """Get raw NV12 data from last frame (for BPU optimization)."""
            return self._last_nv12_data
        
        @property
        def last_frame_size(self) -> Optional[Tuple[int, int]]:
            """Get (width, height) of last frame."""
            return self._last_frame_size


else:
    # Non-RDK fallback (development mode)
    class ContainerFrameServer:
        """
        Fallback frame server for non-RDK platforms.
        
        Uses OpenCV to read from a video file or camera.
        For development and testing purposes.
        """
        
        def __init__(
            self,
            source: str = '',
            topic: str = CONTAINER_NV12_TOPIC,
            queue_size: int = 30,
            node_name: str = 'container_frame_server'
        ):
            """
            Initialize fallback frame server.
            
            Args:
                source: Video file path or camera index
                topic: Ignored (for API compatibility)
                queue_size: Ignored (for API compatibility)
                node_name: Ignored (for API compatibility)
            """
            self.topic = topic
            self.source = source
            self._cap = None
            self._frames_processed = 0
            self._video_ended = False
            self._target_fps = 25.0
            self._frame_interval = 1.0 / self._target_fps
            self._last_frame_time = 0.0
            
            if source:
                self._cap = cv2.VideoCapture(source)
                if not self._cap.isOpened():
                    logger.error(f"[ContainerFrameServer] Failed to open: {source}")
                else:
                    fps = self._cap.get(cv2.CAP_PROP_FPS)
                    if fps and fps > 0:
                        self._target_fps = fps
                        self._frame_interval = 1.0 / fps
                    logger.info(f"[ContainerFrameServer] Opened source: {source} ({self._target_fps:.0f} FPS)")
            else:
                logger.info(
                    "[ContainerFrameServer] No source specified, "
                    "will return empty frames"
                )
        
        def __iter__(self):
            return self
        
        def __next__(self) -> Tuple[np.ndarray, float]:
            """Get next frame from video source."""
            if self._video_ended:
                raise StopIteration
            
            if self._cap and self._cap.isOpened():
                ret, frame = self._cap.read()
                if ret:
                    self._frames_processed += 1
                    return (frame, 0.0)
                else:
                    # Video ended
                    self._video_ended = True
                    logger.info(
                        f"[ContainerFrameServer] Video ended after "
                        f"{self._frames_processed} frames"
                    )
                    raise StopIteration
            
            # No capture — return empty frame (pacing handled by caller)
            return (np.zeros((720, 1280, 3), dtype=np.uint8), 0.0)
        
        def get_frame(self, timeout: float = 0.1) -> Optional[Tuple[np.ndarray, float]]:
            """Get a frame (non-blocking in fallback mode)."""
            return next(self)
        
        def get_stats(self) -> dict:
            """Get frame server statistics."""
            return {
                'topic': self.topic,
                'source': self.source,
                'frames_processed': self._frames_processed,
                'mode': 'fallback',
            }
        
        @property
        def last_nv12_data(self) -> Optional[bytes]:
            return None
        
        @property
        def last_frame_size(self) -> Optional[Tuple[int, int]]:
            return None
        
        def destroy_node(self):
            """Cleanup (compatibility method)."""
            if self._cap:
                try:
                    self._cap.release()
                except Exception as e:
                    logger.warning(f"[ContainerFrameServer] cap.release error: {e}")
                self._cap = None

        def __del__(self):
            """Emergency cleanup in case destroy_node() was not called."""
            try:
                cap = getattr(self, '_cap', None)
                if cap is not None:
                    cap.release()
            except Exception:
                pass


# Factory function for convenience
def create_container_frame_server(
    source: str = '',
    topic: str = CONTAINER_NV12_TOPIC,
    queue_size: int = 30
) -> ContainerFrameServer:
    """
    Create a container frame server appropriate for the platform.
    
    Args:
        source: Video source for fallback mode (ignored on RDK)
        topic: ROS2 topic (used on RDK)
        queue_size: Frame queue size
        
    Returns:
        ContainerFrameServer instance
    """
    if IS_RDK:
        return ContainerFrameServer(topic=topic, queue_size=queue_size)
    else:
        return ContainerFrameServer(source=source, topic=topic, queue_size=queue_size)
