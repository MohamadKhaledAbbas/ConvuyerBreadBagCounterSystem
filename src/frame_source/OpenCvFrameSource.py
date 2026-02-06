"""
OpenCV-based frame source for video files, webcams, or RTSP streams.
With proper backpressure, graceful shutdown, and CPU management.
"""

import queue
import threading
import time
from typing import Iterator, Tuple, Optional

import cv2
import numpy as np

from src.frame_source.FrameSource import FrameSource
from src.utils.AppLogging import logger


class OpenCVFrameSource(FrameSource):
    """
    OpenCV-based frame source.
    
    Supports two modes:
    1. Production mode: Background thread reads frames into queue with backpressure
    2. Testing mode: Synchronous on-demand reading (no frame drops)

    Key features:
    - Backpressure: Queue blocks when full, preventing unbounded memory growth
    - Graceful shutdown: Proper stop mechanism with threading.Event
    - CPU management: Frame pacing with chunked sleeps for responsive interruption
    """

    def __init__(
        self,
        source,
        queue_size: int = 30,
        target_fps: Optional[float] = None,
        testing_mode: bool = False
    ):
        """
        Initialize OpenCV frame source.
        
        Args:
            source: Video source (file path, camera index, or RTSP URL)
            queue_size: Queue size for production mode (default 30, prevents memory issues)
            target_fps: Target FPS for frame pacing (None = source FPS)
            testing_mode: If True, read frames synchronously (no drops)
        """
        self.source = source
        self.testing_mode = testing_mode
        self.cap = cv2.VideoCapture(source)
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source: {source}")
        
        # Get source properties
        self.source_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(
            f"[OpenCVFrameSource] Source: {source}, "
            f"FPS: {self.source_fps}, Frames: {self.total_frames}, "
            f"Size: {self.frame_width}x{self.frame_height}"
        )
        
        self.running = True
        self._stopped = threading.Event()  # For graceful shutdown
        self.last_frame_time = None
        self._frame_count = 0
        self._dropped_frames = 0  # Track dropped frames for diagnostics
        self._queue_full_count = 0  # Track backpressure events

        if testing_mode:
            # Testing mode: synchronous reading
            logger.info("[OpenCVFrameSource] Testing mode - synchronous frame reading")
            self.queue = None
            self.read_thread = None
            self.target_fps = None
            self.frame_interval = None
        else:
            # Production mode: background thread with bounded queue
            logger.info("[OpenCVFrameSource] Production mode - background reading with backpressure")

            # Bounded queue for backpressure
            self.queue = queue.Queue(maxsize=queue_size)
            logger.info(f"[OpenCVFrameSource] Queue size: {queue_size} frames (prevents memory overflow)")

            # Determine frame pacing
            if target_fps and target_fps > 0:
                self.target_fps = target_fps
                self.frame_interval = 1.0 / target_fps
                logger.info(f"[OpenCVFrameSource] Target FPS: {target_fps}")
            else:
                self.target_fps = self.source_fps
                self.frame_interval = 1.0 / self.source_fps
                logger.info(f"[OpenCVFrameSource] Using source FPS: {self.source_fps}")

            self.read_thread = threading.Thread(target=self._read_frames, daemon=True)
            self.read_thread.start()
    
    def _read_frames(self):
        """
        Background thread for reading frames in production mode.

        Features:
        - Backpressure: Blocks when queue is full (consumer is slow)
        - Frame pacing: Matches source FPS to prevent CPU spinning
        - Chunked sleeps: Allows responsive shutdown
        - Graceful error handling
        """
        last_frame_time = time.perf_counter()
        last_log_time = time.perf_counter()

        logger.info("[OpenCVFrameSource] Background reader started")

        while self.running and not self._stopped.is_set():
            try:
                cycle_start = time.perf_counter()

                # Check stop signal before reading
                if self._stopped.is_set():
                    break

                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    logger.info("[OpenCVFrameSource] End of video stream")
                    self.running = False
                    break

                # NOTE: Do NOT resize here! Resize should be done by consumer.
                # Resizing in background thread causes performance issues due to
                # GIL contention and extra memory operations.

                self._frame_count += 1

                # Calculate inter-frame interval
                if self.last_frame_time is None:
                    inter_frame_ms = 0.0
                else:
                    inter_frame_ms = (cycle_start - self.last_frame_time) * 1000.0

                self.last_frame_time = cycle_start

                # BLOCKING put with timeout - implements backpressure
                # If consumer is slow, producer waits (prevents memory overflow)
                try:
                    self.queue.put((frame, inter_frame_ms), block=True, timeout=0.5)
                except queue.Full:
                    # Queue still full after timeout - consumer is very slow
                    self._queue_full_count += 1
                    if self._queue_full_count % 10 == 1:  # Log every 10th occurrence
                        logger.warning(
                            f"[OpenCVFrameSource] Queue full (consumer slow), "
                            f"waiting... (count: {self._queue_full_count})"
                        )
                    # Don't drop frame, retry or wait
                    time.sleep(0.01)
                    continue

                # Frame pacing to match target FPS
                # Prevents CPU from spinning at 100%
                if self.frame_interval is not None:
                    elapsed = time.perf_counter() - cycle_start
                    sleep_time = self.frame_interval - elapsed

                    if sleep_time > 0:
                        # Use chunked sleeps (10ms chunks) for responsive shutdown
                        # This allows us to check stop signal frequently
                        chunks = int(sleep_time / 0.01)
                        for _ in range(chunks):
                            if not self.running or self._stopped.is_set():
                                break
                            time.sleep(0.01)

                        # Sleep remainder
                        remainder = sleep_time - (chunks * 0.01)
                        if remainder > 0 and self.running and not self._stopped.is_set():
                            time.sleep(remainder)

                # Periodic logging
                if time.perf_counter() - last_log_time > 10.0:  # Every 10 seconds
                    if self.total_frames > 0:
                        progress = (self._frame_count / self.total_frames * 100)
                        logger.info(
                            f"[OpenCVFrameSource] Progress: {self._frame_count}/{self.total_frames} "
                            f"({progress:.1f}%), Queue full events: {self._queue_full_count}"
                        )
                    else:
                        logger.info(
                            f"[OpenCVFrameSource] Processed {self._frame_count} frames, "
                            f"Queue full events: {self._queue_full_count}"
                        )
                    last_log_time = time.perf_counter()

            except Exception as e:
                logger.error(f"[OpenCVFrameSource] Read error: {e}")
                time.sleep(0.1)  # Prevent tight error loop

        # Release capture before exiting thread
        if self.cap.isOpened():
            self.cap.release()

        logger.info(
            f"[OpenCVFrameSource] Background reader stopped. "
            f"Total frames: {self._frame_count}, Queue full events: {self._queue_full_count}"
        )


    def _read_frame_sync(self) -> Optional[Tuple[np.ndarray, float]]:
        """Synchronously read next frame (testing mode)."""
        if not self.running:
            return None
        
        cycle_start = time.perf_counter()
        
        ret, frame = self.cap.read()
        if not ret:
            self.running = False
            return None
        
        # Resize to standard resolution
        frame = cv2.resize(frame, (1280, 720))
        
        self._frame_count += 1
        
        # Calculate inter-frame interval
        if self.last_frame_time is None:
            inter_frame_ms = 0.0
        else:
            inter_frame_ms = (cycle_start - self.last_frame_time) * 1000.0
        
        self.last_frame_time = cycle_start
        
        # Log progress periodically
        if self._frame_count % 100 == 0:
            if self.total_frames > 0:
                progress = (self._frame_count / self.total_frames * 100)
                logger.info(
                    f"[OpenCVFrameSource] Progress: {self._frame_count}/{self.total_frames} "
                    f"({progress:.1f}%)"
                )
            else:
                logger.info(f"[OpenCVFrameSource] Processed {self._frame_count} frames")
        
        return frame, inter_frame_ms

    def frames(self) -> Iterator[Tuple[np.ndarray, float]]:
        """
        Yield frames from the video source.
        
        Yields:
            Tuple of (frame, latency_ms)
        """
        if self.testing_mode:
            # Testing mode: synchronous reading
            while self.running:
                result = self._read_frame_sync()
                if result is None:
                    break
                yield result
            
            logger.info(f"[OpenCVFrameSource] Completed: {self._frame_count} frames")
        else:
            # Production mode: read from queue
            while self.running or not self.queue.empty():
                try:
                    yield self.queue.get(timeout=0.1)
                except queue.Empty:
                    continue
    
    def cleanup(self):
        """
        Clean up resources gracefully.

        Ensures:
        - Stop signal is sent
        - Background thread terminates
        - Queue is cleared
        - Video capture is released
        """
        logger.info("[OpenCVFrameSource] Cleanup starting...")

        # Signal stop
        self.running = False
        self._stopped.set()

        # Wait for background thread to finish
        if self.read_thread is not None and self.read_thread.is_alive():
            logger.info("[OpenCVFrameSource] Waiting for background thread to stop...")
            self.read_thread.join(timeout=3.0)

            if self.read_thread.is_alive():
                logger.warning("[OpenCVFrameSource] Background thread did not stop cleanly")

        # Clear queue to unblock any waiting consumers
        if self.queue is not None:
            cleared = 0
            while not self.queue.empty():
                try:
                    self.queue.get_nowait()
                    cleared += 1
                except queue.Empty:
                    break
            if cleared > 0:
                logger.info(f"[OpenCVFrameSource] Cleared {cleared} frames from queue")

        # Release video capture if still open
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        
        logger.info(
            f"[OpenCVFrameSource] Cleanup complete. "
            f"Processed {self._frame_count} frames, "
            f"Queue full events: {self._queue_full_count}"
        )
