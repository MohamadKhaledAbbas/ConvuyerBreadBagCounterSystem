"""
Ultralytics-based detector for Windows/development platforms.
"""

from typing import List, Tuple, Optional

import numpy as np

from src.detection.BaseDetection import BaseDetector, Detection
from src.utils.AppLogging import logger


class UltralyticsDetector(BaseDetector):
    """
    Ultralytics YOLO detector for Windows/development.
    
    Uses Ultralytics library with .pt model files.
    """
    
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.5,
        device: Optional[str] = None,
        input_size: Tuple[int, int] = (640, 640)
    ):
        """
        Initialize Ultralytics detector.
        
        Args:
            model_path: Path to .pt model file
            confidence_threshold: Minimum confidence for detections
            device: Device to run inference on ('cpu', 'cuda', etc.)
            input_size: Model input size (not directly used, handled by YOLO)
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("ultralytics package not installed. Install with: pip install ultralytics")
        
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.input_size = input_size
        
        # Load model
        logger.info(f"[UltralyticsDetector] Loading model: {model_path}")
        self.model = YOLO(model_path)
        
        # Set device
        if device:
            self.device = device
        else:
            import torch
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        logger.info(f"[UltralyticsDetector] Model loaded, device: {self.device}")

        # Suppress YOLO verbose logging
        import logging as log
        log.getLogger('ultralytics').setLevel(log.WARNING)

        # GPU warmup for optimal performance
        if self.device == 'cuda':
            logger.info("[UltralyticsDetector] Warming up GPU...")
            dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            for _ in range(3):
                self.model(dummy_frame, device=self.device, verbose=False)
            logger.info("[UltralyticsDetector] GPU warmup complete")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect bread bags in a frame using Ultralytics YOLO.
        
        Args:
            frame: BGR image (numpy array)
            
        Returns:
            List of Detection objects
        """
        # Run inference
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False
        )
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            
            if boxes is None:
                continue
            
            for i in range(len(boxes)):
                # Get bounding box (xyxy format)
                xyxy = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                
                # Get confidence
                confidence = float(boxes.conf[i].cpu().numpy())
                
                # Get class (single class detector: always bread-bag)
                class_id = int(boxes.cls[i].cpu().numpy())
                
                # Get class name from model
                class_name = self.model.names.get(class_id, "bread-bag")
                
                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name
                ))
        
        return detections
    
    def cleanup(self):
        """Release model resources."""
        self.model = None
        logger.info("[UltralyticsDetector] Cleanup complete")
