"""
Utility functions for ConveyerBreadBagCounterSystem.
"""

import hashlib
import numpy as np
from typing import Tuple, Optional

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def compute_phash(image: np.ndarray, hash_size: int = 8) -> Optional[str]:
    """
    Compute perceptual hash of an image.
    
    Args:
        image: BGR numpy array
        hash_size: Size of hash (default 8 produces 64-bit hash)
        
    Returns:
        Hexadecimal hash string or None if computation fails
    """
    if not CV2_AVAILABLE or image is None:
        return None
    
    try:
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Resize to hash_size + 1 x hash_size for DCT-like computation
        resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
        
        # Compute difference hash
        diff = resized[:, 1:] > resized[:, :-1]
        
        # Convert to hash string
        hash_bytes = np.packbits(diff.flatten()).tobytes()
        return hash_bytes.hex()
    except Exception:
        return None


def compute_iou(box1: Tuple[float, float, float, float], 
                box2: Tuple[float, float, float, float]) -> float:
    """
    Compute Intersection over Union between two bounding boxes.
    
    Args:
        box1: (x1, y1, x2, y2) coordinates
        box2: (x1, y1, x2, y2) coordinates
        
    Returns:
        IoU value between 0 and 1
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    if union_area <= 0:
        return 0.0
    
    return inter_area / union_area


def compute_centroid(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """
    Compute centroid of a bounding box.
    
    Args:
        box: (x1, y1, x2, y2) coordinates
        
    Returns:
        (cx, cy) centroid coordinates
    """
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def compute_sharpness(image: np.ndarray) -> float:
    """
    Compute image sharpness using Laplacian variance.
    
    Args:
        image: BGR or grayscale numpy array
        
    Returns:
        Sharpness value (higher = sharper)
    """
    if not CV2_AVAILABLE or image is None:
        return 0.0
    
    try:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:
        return 0.0


def compute_brightness(image: np.ndarray) -> float:
    """
    Compute mean brightness of an image.
    
    Args:
        image: BGR numpy array
        
    Returns:
        Mean brightness value (0-255)
    """
    if image is None or image.size == 0:
        return 0.0
    
    try:
        if len(image.shape) == 3:
            # Use luminance formula for BGR
            return float(np.mean(image[:, :, 0] * 0.114 + 
                                 image[:, :, 1] * 0.587 + 
                                 image[:, :, 2] * 0.299))
        return float(np.mean(image))
    except Exception:
        return 0.0
