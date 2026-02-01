"""
Abstract base class for frame sources.
"""

from abc import ABC, abstractmethod
from typing import Iterator, Tuple, Any


class FrameSource(ABC):
    """
    Abstract base class for video frame sources.
    
    Implementations provide frames from various sources:
    - OpenCV (files, webcams, RTSP)
    - ROS2 topics
    """
    
    @abstractmethod
    def frames(self) -> Iterator[Tuple[Any, float]]:
        """
        Yield frames from the source.
        
        Yields:
            Tuple of (frame, latency_ms)
            - frame: numpy array (BGR format)
            - latency_ms: inter-frame latency in milliseconds
        """
        pass
    
    @abstractmethod
    def cleanup(self):
        """Release resources."""
        pass
