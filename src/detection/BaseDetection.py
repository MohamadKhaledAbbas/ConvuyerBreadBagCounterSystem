"""
Base detector interface for bread bag detection.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class Detection:
    """Single detection result."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int = 0  # Default: bread-bag (single class)
    class_name: str = "bread-bag"
    
    @property
    def center(self) -> Tuple[int, int]:
        """Return center point of bbox."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    @property
    def area(self) -> int:
        """Return area of bbox."""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)
    
    @property
    def width(self) -> int:
        """Return width of bbox."""
        return self.bbox[2] - self.bbox[0]
    
    @property
    def height(self) -> int:
        """Return height of bbox."""
        return self.bbox[3] - self.bbox[1]


class BaseDetector(ABC):
    """
    Abstract base class for object detectors.
    
    The conveyor system uses a simple single-class detector
    for bread bags (no open/closing/closed states).
    """
    
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect bread bags in a frame.
        
        Args:
            frame: BGR image (numpy array)
            
        Returns:
            List of Detection objects
        """
        pass
    
    @abstractmethod
    def cleanup(self):
        """Release model resources."""
        pass
