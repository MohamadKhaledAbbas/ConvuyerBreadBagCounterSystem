"""
OpenCV-based frame source for video files, webcams, or RTSP streams.
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
    1. Production mode: Background thread reads frames into queue
    2. Testing mode: Synchronous on-demand reading (no frame drops)
    """
    
    def __init__(
        self,
        source,
        queue_size: int = 0,
        target_fps: Optional[float] = None,
        testing_mode: bool = False
    ):
        """
        Initialize OpenCV frame source.
        
        Args:
            source: Video source (file path, camera index, or RTSP URL)
            queue_size: Queue size for production mode (0 = unlimited)
            target_fps: Target FPS for frame pacing (None = no limit)
            testing_mode: If True, read frames synchronously (no drops)
        """
        self.source = source
        self.testing_mode = testing_mode
        self.cap = cv2.VideoCapture(source)
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source: {source}")
        
        # Get source properties
        self.source_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(
            f"[OpenCVFrameSource] Source: {source}, "
            f"FPS: {self.source_fps}, Frames: {self.total_frames}, "
            f"Size: {self.frame_width}x{self.frame_height}"
        )
        
        self.running = True
        self.last_frame_time = None
        self._frame_count = 0
        
        if testing_mode:
            # Testing mode: synchronous reading
            logger.info("[OpenCVFrameSource] Testing mode - synchronous frame reading")
            self.queue = None
            self.read_thread = None
            self.target_fps = None
            self.frame_interval = None
        else:
            # Production mode: background thread
            logger.info("[OpenCVFrameSource] Production mode - background reading")
            self.queue = queue.Queue(maxsize=queue_size)
            self.target_fps = target_fps
            
            if target_fps and target_fps > 0:
                self.frame_interval = 1.0 / target_fps
                logger.info(f"[OpenCVFrameSource] Target FPS: {target_fps}")
            else:
                self.frame_interval = None
            
            self.read_thread = threading.Thread(target=self._read_frames, daemon=True)
            self.read_thread.start()
    
    def _read_frames(self):
        """Background thread for reading frames in production mode."""
        while self.running:
            cycle_start = time.perf_counter()
            
            ret, frame = self.cap.read()
            if not ret:
                self.running = False
                break
            
            # Resize to standard resolution
            frame = cv2.resize(frame, (1280, 720))
            
            self._frame_count += 1
            
            # Calculate inter-frame interval
            if self.last_frame_time is None:
                inter_frame_ms = 0.0
            else:
                inter_frame_ms = (cycle_start - self.last_frame_time) * 1000.0
            
            self.last_frame_time = cycle_start
            
            # Block if consumer is slower (no frame skipping)
            self.queue.put((frame, inter_frame_ms))
            
            # Frame pacing
            if self.frame_interval is not None:
                elapsed = time.perf_counter() - cycle_start
                sleep_time = self.frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        self.cap.release()
    
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
        """Clean up resources."""
        self.running = False
        
        if self.read_thread is not None and self.read_thread.is_alive():
            self.read_thread.join(timeout=2.0)
        
        if self.cap.isOpened():
            self.cap.release()
        
        logger.info(f"[OpenCVFrameSource] Cleanup complete, processed {self._frame_count} frames")
