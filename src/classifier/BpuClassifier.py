"""
BPU-accelerated classifier for RDK platform.
"""

from typing import List, Tuple
import numpy as np
import cv2

from src.classifier.BaseClassifier import BaseClassifier, ClassificationResult
from src.utils.AppLogging import logger


try:
    from hobot_dnn import pyeasy_dnn as dnn
    HAS_BPU = True
except ImportError:
    HAS_BPU = False


class BpuClassifier(BaseClassifier):
    """
    BPU-accelerated image classifier for RDK platform.
    
    Uses Horizon BPU-optimized .bin models for efficient
    edge inference on RDK hardware.
    """
    
    def __init__(
        self,
        model_path: str,
        classes: List[str],
        input_size: Tuple[int, int] = (224, 224)
    ):
        """
        Initialize BPU classifier.
        
        Args:
            model_path: Path to .bin model file
            classes: List of class names
            input_size: Model input size (width, height)
        """
        if not HAS_BPU:
            raise ImportError("hobot_dnn not available. BPU classifier requires RDK platform.")
        
        self.model_path = model_path
        self._class_names = classes
        self.input_size = input_size
        
        # Load model
        logger.info(f"[BpuClassifier] Loading model: {model_path}")
        self.models = dnn.load(model_path)
        
        if not self.models:
            raise ValueError(f"Failed to load model: {model_path}")
        
        logger.info(f"[BpuClassifier] Model loaded, classes: {len(classes)}")
    
    def _preprocess(self, roi: np.ndarray) -> np.ndarray:
        """Preprocess ROI for BPU inference."""
        # Resize to model input size
        resized = cv2.resize(roi, self.input_size)
        
        # BPU expects BGR HWC format
        return resized
    
    def _postprocess(self, output: np.ndarray) -> ClassificationResult:
        """
        Post-process BPU classifier output.
        
        Args:
            output: Model output tensor (logits or probabilities)
            
        Returns:
            ClassificationResult
        """
        # Flatten if needed
        output = output.flatten()
        
        # Apply softmax if not already probabilities
        if output.max() > 1.0 or output.min() < 0.0:
            # Logits - apply softmax
            exp_output = np.exp(output - np.max(output))
            probabilities = exp_output / exp_output.sum()
        else:
            probabilities = output
        
        # Get top prediction
        class_id = int(np.argmax(probabilities))
        confidence = float(probabilities[class_id])
        
        class_name = self._class_names[class_id] if class_id < len(self._class_names) else "Unknown"
        
        return ClassificationResult(
            class_id=class_id,
            class_name=class_name,
            confidence=confidence
        )
    
    def classify(self, roi: np.ndarray) -> ClassificationResult:
        """
        Classify a single ROI image.
        
        Args:
            roi: BGR image crop of detected bread bag
            
        Returns:
            ClassificationResult
        """
        # Preprocess
        input_data = self._preprocess(roi)
        
        # Run inference
        outputs = self.models[0].forward(input_data)
        
        # Get output array
        output_array = outputs[0].buffer
        
        # Post-process
        return self._postprocess(output_array)
    
    def classify_batch(self, rois: List[np.ndarray]) -> List[ClassificationResult]:
        """
        Classify multiple ROI images.
        
        Note: BPU runs sequentially for simplicity.
        
        Args:
            rois: List of BGR image crops
            
        Returns:
            List of ClassificationResult objects
        """
        return [self.classify(roi) for roi in rois]
    
    def cleanup(self):
        """Release BPU resources."""
        self.models = None
        logger.info("[BpuClassifier] Cleanup complete")
    
    @property
    def class_names(self) -> List[str]:
        """Return list of class names."""
        return self._class_names
