"""
Ultralytics-based classifier for Windows/development platforms.
"""

from typing import List, Tuple, Optional

import numpy as np

from src.classifier.BaseClassifier import BaseClassifier, ClassificationResult
from src.utils.AppLogging import logger


class UltralyticsClassifier(BaseClassifier):
    """
    Ultralytics YOLO classifier for Windows/development.
    
    Uses Ultralytics library with .pt classification model.
    """
    
    # Periodic CUDA cache cleanup frequency (every N classifications)
    CUDA_CLEANUP_INTERVAL = 200

    def __init__(
        self,
        model_path: str,
        classes: List[str],
        device: Optional[str] = None,
        input_size: Tuple[int, int] = (224, 224)
    ):
        """
        Initialize Ultralytics classifier.
        
        Args:
            model_path: Path to .pt classification model
            classes: List of class names
            device: Device to run inference on
            input_size: Model input size (not directly used)
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("ultralytics package not installed. Install with: pip install ultralytics")
        
        self.model_path = model_path
        self._class_names = classes
        self.input_size = input_size
        self._inference_count = 0  # Track for periodic cleanup

        # Load model
        logger.info(f"[UltralyticsClassifier] Loading model: {model_path}")
        self.model = YOLO(model_path)
        
        # Set device
        if device:
            self.device = device
        else:
            import torch
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        logger.info(f"[UltralyticsClassifier] Model loaded, device: {self.device}")

        # Suppress YOLO verbose logging
        import logging as log
        log.getLogger('ultralytics').setLevel(log.WARNING)

        # GPU warmup for optimal performance
        if self.device == 'cuda':
            logger.info("[UltralyticsClassifier] Warming up GPU...")
            dummy_roi = np.zeros((224, 224, 3), dtype=np.uint8)
            for _ in range(3):
                self.model(dummy_roi, device=self.device, verbose=False)
            logger.info("[UltralyticsClassifier] GPU warmup complete")

    def classify(self, roi: np.ndarray) -> ClassificationResult:
        """
        Classify a single ROI image.
        
        Args:
            roi: BGR image crop of detected bread bag
            
        Returns:
            ClassificationResult
        """
        self._inference_count += 1

        # Periodic CUDA cache cleanup to prevent memory accumulation
        if self.device == 'cuda' and self._inference_count % self.CUDA_CLEANUP_INTERVAL == 0:
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

        # Run inference
        results = self.model(
            roi,
            device=self.device,
            verbose=False
        )
        
        if not results or len(results) == 0:
            return ClassificationResult(
                class_id=-1,
                class_name="Unknown",
                confidence=0.0
            )
        
        result = results[0]
        
        # Get probabilities
        if hasattr(result, 'probs') and result.probs is not None:
            probs = result.probs
            class_id = int(probs.top1)
            confidence = float(probs.top1conf.cpu().numpy())
            
            # Use model's class names or our configured ones
            if class_id < len(self._class_names):
                class_name = self._class_names[class_id]
            elif hasattr(self.model, 'names') and class_id in self.model.names:
                class_name = self.model.names[class_id]
            else:
                class_name = f"Class_{class_id}"
        else:
            # Fallback
            class_id = 0
            class_name = self._class_names[0] if self._class_names else "Unknown"
            confidence = 0.5
        
        return ClassificationResult(
            class_id=class_id,
            class_name=class_name,
            confidence=confidence
        )
    
    def classify_batch(self, rois: List[np.ndarray]) -> List[ClassificationResult]:
        """
        Classify multiple ROI images.
        
        Args:
            rois: List of BGR image crops
            
        Returns:
            List of ClassificationResult objects
        """
        if not rois:
            return []
        
        # Run batch inference
        results = self.model(
            rois,
            device=self.device,
            verbose=False
        )
        
        classifications = []
        
        for i, result in enumerate(results):
            if hasattr(result, 'probs') and result.probs is not None:
                probs = result.probs
                class_id = int(probs.top1)
                confidence = float(probs.top1conf.cpu().numpy())
                
                if class_id < len(self._class_names):
                    class_name = self._class_names[class_id]
                elif hasattr(self.model, 'names') and class_id in self.model.names:
                    class_name = self.model.names[class_id]
                else:
                    class_name = f"Class_{class_id}"
            else:
                class_id = 0
                class_name = self._class_names[0] if self._class_names else "Unknown"
                confidence = 0.5
            
            classifications.append(ClassificationResult(
                class_id=class_id,
                class_name=class_name,
                confidence=confidence
            ))
        
        return classifications
    
    def cleanup(self):
        """Release model resources."""
        self.model = None

        # Final CUDA cache cleanup
        if self.device == 'cuda':
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

        logger.info("[UltralyticsClassifier] Cleanup complete")
    
    @property
    def class_names(self) -> List[str]:
        """Return list of class names."""
        return self._class_names
