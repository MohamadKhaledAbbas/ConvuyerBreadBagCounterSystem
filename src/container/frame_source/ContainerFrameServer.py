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

import os
import queue
import time
from typing import Iterator, Optional, Tuple
import numpy as np
import cv2

from src.utils.AppLogging import logger
from src.utils.platform import IS_RDK

_FRAME_DEBUG = int(os.environ.get('FRAME_DEBUG', '0') or '0')

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
        ROS2 node subscribing to /nv12_images_container, providing
        frames via a yield-based frames() generator.
        
        Matches the proven bread-camera FrameServer pattern:
        - Callback stores lightweight NV12 copy only (no BGR conversion)
        - queue.Queue with blocking get (no sentinel frames)
        - Lazy NV12→BGR conversion at yield time
        - Drain-to-newest ensures consumer always gets the freshest frame
        """
        
        def __init__(
            self,
            topic: str = CONTAINER_NV12_TOPIC,
            queue_size: int = 30,
            node_name: str = 'container_frame_server',
            target_fps: float = 20.0,
        ):
            super().__init__(node_name)
            
            self.topic = topic
            self.queue_size = queue_size
            
            # Frame queue: stores (nv12_data, latency_ms, (height, width))
            self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
            
            # Proactive drop threshold (80% of queue capacity)
            self._proactive_drop_threshold = int(queue_size * 0.8)
            
            # NV12 data for BPU optimization (if needed)
            self._last_nv12_data: Optional[np.ndarray] = None
            self._last_frame_size: Optional[Tuple[int, int]] = None
            
            # Statistics
            self._frames_received = 0
            self._frames_dropped = 0
            self._frames_processed = 0
            self._last_stats_time = time.time()
            self._stats_log_interval = 5.0
            
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
                f"queue_size={queue_size}, target_fps={target_fps}"
            )
        
        def _frame_callback(self, msg: Image) -> None:
            """
            Handle incoming NV12 frame from ROS2.
            
            Stores lightweight NV12 copy only — NO BGR conversion here.
            This keeps the callback fast so spin_once returns quickly.
            """
            self._frames_received += 1
            
            try:
                height = msg.height
                width = msg.width
                
                expected_size = int(height * 1.5 * width)
                actual_size = len(msg.data)
                
                if actual_size < expected_size:
                    logger.warning(
                        f"[ContainerFrameServer] Frame size mismatch: "
                        f"expected {expected_size}, got {actual_size}"
                    )
                    return
                
                # Reshape NV12 and store lightweight copy (~1 MB at 720p)
                nv12_data = np.frombuffer(msg.data, dtype=np.uint8)
                nv12_frame = nv12_data.reshape((int(height * 1.5), width))
                nv12_copy = nv12_frame.copy()
                
                # Calculate latency from message timestamp
                callback_time = time.time()
                msg_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                latency_ms = (callback_time - msg_time) * 1000 if msg_time > 0 else 0
                
                # Leaky queue: proactive drop when near full
                qsize = self._queue.qsize()
                if qsize >= self._proactive_drop_threshold:
                    try:
                        self._queue.get_nowait()
                        self._frames_dropped += 1
                    except queue.Empty:
                        pass
                elif self._queue.full():
                    try:
                        self._queue.get_nowait()
                        self._frames_dropped += 1
                    except queue.Empty:
                        pass
                
                self._queue.put((nv12_copy, latency_ms, (height, width)))
                
            except Exception as e:
                logger.error(f"[ContainerFrameServer] Frame callback error: {e}")
        
        def frames(self) -> Iterator[Tuple[np.ndarray, float]]:
            """
            Yield BGR frames from the ROS2 subscription.
            
            Drains ALL pending DDS messages each iteration, then picks the
            newest frame from the queue.  NV12→BGR conversion happens lazily
            at yield time, not in the callback.
            
            Yields:
                Tuple of (BGR frame, latency_ms)
            """
            logger.info("[ContainerFrameServer] Starting frame iteration loop")
            frame_count = 0
            
            while rclpy.ok():
                # Pump ALL pending ROS2 callbacks into our queue.
                # spin_once(0.0) returns immediately if nothing is ready.
                for _ in range(self.queue_size):
                    rclpy.spin_once(self, timeout_sec=0.0)
                
                # Drain queue to newest frame
                item = None
                drained = 0
                while not self._queue.empty():
                    try:
                        latest = self._queue.get_nowait()
                        if item is not None:
                            drained += 1
                        item = latest
                    except queue.Empty:
                        break
                self._frames_dropped += drained
                
                if item is None:
                    # Nothing available — block briefly for new data
                    rclpy.spin_once(self, timeout_sec=0.05)
                    continue
                
                nv12_data, latency_ms, (height, width) = item
                
                # Lazy NV12 → BGR conversion
                t_convert_start = time.time()
                bgr_frame = cv2.cvtColor(nv12_data, cv2.COLOR_YUV2BGR_NV12)
                t_convert_ms = (time.time() - t_convert_start) * 1000
                
                # Store for BPU optimization
                self._last_nv12_data = nv12_data
                self._last_frame_size = (width, height)
                
                frame_count += 1
                self._frames_processed += 1
                
                if frame_count == 1:
                    logger.info(
                        f"[ContainerFrameServer] First frame yielded! "
                        f"{width}x{height}"
                    )
                if _FRAME_DEBUG and (_FRAME_DEBUG >= 2 or frame_count % 50 == 0):
                    logger.info(
                        f"[Frame-DBG] mode=ros2 frame={frame_count} "
                        f"convert_ms={t_convert_ms:.1f} latency_ms={latency_ms:.1f} "
                        f"drained={drained}"
                    )
                
                # Log stats periodically
                now = time.time()
                if now - self._last_stats_time >= self._stats_log_interval:
                    elapsed = now - self._last_stats_time
                    recv_fps = self._frames_received / elapsed if elapsed > 0 else 0
                    drop_rate = (
                        self._frames_dropped / self._frames_received * 100
                        if self._frames_received > 0 else 0
                    )
                    logger.info(
                        f"[ContainerFrameServer] Stats: "
                        f"recv_fps={recv_fps:.1f}, "
                        f"processed={self._frames_processed}, "
                        f"dropped={self._frames_dropped}, "
                        f"drop_rate={drop_rate:.1f}%, "
                        f"queue={self._queue.qsize()}"
                    )
                    # Reset counters for next interval
                    self._frames_received = 0
                    self._frames_dropped = 0
                    self._last_stats_time = now
                
                yield bgr_frame, latency_ms
        
        def __iter__(self) -> Iterator[Tuple[np.ndarray, float]]:
            """Iterate via frames() generator."""
            return self.frames()
        
        def get_frame(self, timeout: float = 0.1) -> Optional[Tuple[np.ndarray, float]]:
            """
            Get a single frame with timeout.
            
            Args:
                timeout: Maximum seconds to wait
                
            Returns:
                Tuple of (BGR frame, latency_ms) or None if timeout
            """
            start = time.time()
            
            while time.time() - start < timeout:
                rclpy.spin_once(self, timeout_sec=0.001)
                try:
                    nv12_data, latency_ms, (height, width) = self._queue.get_nowait()
                    bgr_frame = cv2.cvtColor(nv12_data, cv2.COLOR_YUV2BGR_NV12)
                    self._last_nv12_data = nv12_data
                    self._last_frame_size = (width, height)
                    self._frames_processed += 1
                    return (bgr_frame, latency_ms)
                except queue.Empty:
                    pass
            
            return None
        
        def get_stats(self) -> dict:
            """Get frame server statistics."""
            elapsed = time.time() - self._last_stats_time
            fps = self._frames_received / elapsed if elapsed > 0 else 0
            
            return {
                'topic': self.topic,
                'frames_received': self._frames_received,
                'frames_processed': self._frames_processed,
                'frames_dropped': self._frames_dropped,
                'queue_size': self._queue.qsize(),
                'queue_capacity': self.queue_size,
                'queue_fill_percent': self._queue.qsize() / self.queue_size * 100,
                'fps': round(fps, 1),
                'drop_rate': (
                    self._frames_dropped / self._frames_received * 100
                    if self._frames_received > 0 else 0
                ),
            }
        
        @property
        def last_nv12_data(self) -> Optional[np.ndarray]:
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
            self._read_time_total = 0.0
            
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
                t_read_start = time.time()
                ret, frame = self._cap.read()
                t_read_ms = (time.time() - t_read_start) * 1000
                if ret:
                    self._frames_processed += 1
                    self._read_time_total += t_read_ms
                    if _FRAME_DEBUG and (_FRAME_DEBUG >= 2 or self._frames_processed % 50 == 0):
                        mean_read_ms = self._read_time_total / max(1, self._frames_processed)
                        logger.info(
                            f"[Frame-DBG] mode=opencv frame={self._frames_processed} "
                            f"read_ms={t_read_ms:.1f} mean_read_ms={mean_read_ms:.1f} "
                            f"shape={frame.shape}"
                        )
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
