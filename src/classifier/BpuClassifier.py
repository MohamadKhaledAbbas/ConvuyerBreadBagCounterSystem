"""
BPU-accelerated classifier for RDK platform.

Based on the working implementation from BreadBagCounterSystem.
Optimized for NV12 input format as required by BPU models.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional

from src.classifier.BaseClassifier import BaseClassifier, ClassificationResult
from src.utils.AppLogging import logger

# Import BPU Library safely
try:
    from hobot_dnn import pyeasy_dnn as dnn
    HAS_BPU = True
    logger.info("[BpuClassifier] Using hobot_dnn")
except ImportError:
    try:
        from hobot_dnn_rdkx5 import pyeasy_dnn as dnn
        HAS_BPU = True
        logger.info("[BpuClassifier] Using hobot_dnn_rdkx5")
    except ImportError:
        logger.warning("[BpuClassifier] hobot_dnn not found. BPU classification unavailable.")
        dnn = None
        HAS_BPU = False


class BpuClassifier(BaseClassifier):
    """
    BPU-accelerated image classifier for RDK platform.
    
    Uses Horizon BPU-optimized .bin models for efficient
    edge inference on RDK hardware.

    Requires NV12 format input (handled automatically in preprocessing).
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
        self.input_w, self.input_h = input_size

        # Pre-allocate NV12 buffer for efficiency
        self.area = self.input_h * self.input_w
        self.nv12_buffer = np.zeros((self.area * 3 // 2,), dtype=np.uint8)

        # Load model
        logger.info(f"[BpuClassifier] Loading model: {model_path}")
        self.model = dnn.load(model_path)

        if not self.model:
            raise ValueError(f"Failed to load model: {model_path}")
        
        # Try to log model input properties
        try:
            input_shape = self.model[0].inputs[0].properties.shape
            logger.info(f"[BpuClassifier] Model loaded. Input shape: {input_shape}")
        except Exception as e:
            logger.debug(f"[BpuClassifier] Could not read model properties: {e}")

        logger.info(f"[BpuClassifier] Model loaded, classes: {len(classes)}")
    
    # Minimum ROI dimensions for meaningful classification
    MIN_ROI_SIZE = 10

    def _validate_roi(self, roi: np.ndarray) -> Optional[str]:
        """
        Validate ROI image before processing.

        Catches invalid inputs that would produce incorrect NV12 data
        (e.g., grayscale images, tiny crops, wrong dtype).

        Args:
            roi: Input image to validate

        Returns:
            Error message string if invalid, None if valid
        """
        if roi is None:
            return "ROI is None"

        if not isinstance(roi, np.ndarray):
            return f"ROI is not ndarray: {type(roi)}"

        if roi.size == 0:
            return "ROI is empty"

        if len(roi.shape) != 3:
            return f"ROI must be 3-channel BGR, got shape {roi.shape}"

        if roi.shape[2] != 3:
            return f"ROI must have 3 channels, got {roi.shape[2]}"

        if roi.shape[0] < self.MIN_ROI_SIZE or roi.shape[1] < self.MIN_ROI_SIZE:
            return f"ROI too small: {roi.shape[1]}x{roi.shape[0]} (min {self.MIN_ROI_SIZE}x{self.MIN_ROI_SIZE})"

        return None

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocess BGR image to NV12 format for BPU.

        Args:
            img: BGR image (numpy array, must be 3-channel)

        Returns:
            NV12 formatted buffer
        """
        # Use INTER_LINEAR for classification quality (smooth upscaling of small ROIs)
        resized = cv2.resize(img, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
        return self._bgr2nv12(resized)

    def _bgr2nv12(self, bgr_img: np.ndarray) -> np.ndarray:
        """
        Convert BGR to NV12 using pre-allocated buffer.

        NV12 format:
        - Y plane: full resolution (height * width bytes)
        - UV plane: half resolution, interleaved (height/2 * width bytes)

        Args:
            bgr_img: BGR image resized to model input size

        Returns:
            NV12 buffer ready for BPU inference
        """
        # Convert BGR to YUV I420 (planar: Y...U...V...)
        yuv420p = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YUV_I420).reshape((self.area * 3 // 2,))

        # Copy Y plane directly to pre-allocated buffer
        self.nv12_buffer[:self.area] = yuv420p[:self.area]

        # Interleave UV into NV12 format (UVUVUV...)
        u_start = self.area
        v_start = self.area + (self.area // 4)
        self.nv12_buffer[self.area::2] = yuv420p[u_start:v_start]
        self.nv12_buffer[self.area + 1::2] = yuv420p[v_start:]

        return np.ascontiguousarray(self.nv12_buffer)

    def _postprocess(self, output: np.ndarray) -> Tuple[int, str, float, np.ndarray]:
        """
        Post-process BPU classifier output.
        
        Args:
            output: Model output tensor (logits or probabilities)
            
        Returns:
            Tuple of (class_id, class_name, confidence, probabilities)
        """
        # Flatten if needed
        probs = output.flatten()

        # Apply softmax if not already probabilities
        if probs.max() > 1.0 or probs.min() < 0.0:
            exp_scores = np.exp(probs - np.max(probs))
            probs = exp_scores / np.sum(exp_scores)

        # Ensure normalized
        probs_sum = np.sum(probs)
        if probs_sum > 0:
            probs = probs / probs_sum

        # Get top prediction
        class_id = int(np.argmax(probs))
        confidence = float(probs[class_id])
        class_name = self._class_names[class_id] if class_id < len(self._class_names) else "Unknown"
        
        return class_id, class_name, confidence, probs

    def classify(self, roi: np.ndarray) -> ClassificationResult:
        """
        Classify a single ROI image.
        
        Args:
            roi: BGR image crop of detected bread bag
            
        Returns:
            ClassificationResult
        """
        if self.model is None:
            logger.error("[BpuClassifier] Model not loaded!")
            return ClassificationResult(class_id=-1, class_name="Unknown", confidence=0.0)

        # Comprehensive input validation (prevents NV12 conversion of invalid images)
        error = self._validate_roi(roi)
        if error is not None:
            logger.error(f"[BpuClassifier] Invalid ROI: {error}")
            return ClassificationResult(class_id=-1, class_name="Unknown", confidence=0.0)

        try:
            # Preprocess (resize + BGR to NV12)
            input_data = self._preprocess(roi)

            # Run inference
            outputs = self.model[0].forward(input_data)

            # Get output array
            output_array = outputs[0].buffer

            # Post-process
            class_id, class_name, confidence, _ = self._postprocess(output_array)

            return ClassificationResult(
                class_id=class_id,
                class_name=class_name,
                confidence=confidence
            )

        except Exception as e:
            logger.error(f"[BpuClassifier] Classification error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return ClassificationResult(class_id=-1, class_name="Unknown", confidence=0.0)

    def classify_with_probs(self, roi: np.ndarray) -> Tuple[ClassificationResult, Dict[str, float]]:
        """
        Classify ROI and return full probability distribution.

        Required for multi-ROI voting and evidence accumulation.

        Args:
            roi: BGR image crop

        Returns:
            Tuple of (ClassificationResult, probability_dict)
        """
        if self.model is None:
            return ClassificationResult(class_id=-1, class_name="Unknown", confidence=0.0), {"Unknown": 1.0}

        # Comprehensive input validation
        error = self._validate_roi(roi)
        if error is not None:
            logger.error(f"[BpuClassifier] classify_with_probs invalid ROI: {error}")
            return ClassificationResult(class_id=-1, class_name="Unknown", confidence=0.0), {"Unknown": 1.0}

        try:
            # Preprocess
            input_data = self._preprocess(roi)

            # Run inference
            outputs = self.model[0].forward(input_data)
            output_array = outputs[0].buffer

            # Post-process
            class_id, class_name, confidence, probs = self._postprocess(output_array)

            # Build probability dictionary
            probs_dict = {}
            for i, name in enumerate(self._class_names):
                if i < len(probs):
                    probs_dict[name] = float(probs[i])

            result = ClassificationResult(
                class_id=class_id,
                class_name=class_name,
                confidence=confidence
            )

            return result, probs_dict

        except Exception as e:
            logger.error(f"[BpuClassifier] classify_with_probs error: {e}")
            return ClassificationResult(class_id=-1, class_name="Unknown", confidence=0.0), {"Unknown": 1.0}

    def classify_batch(self, rois: List[np.ndarray]) -> List[ClassificationResult]:
        """
        Classify multiple ROI images.
        
        Args:
            rois: List of BGR image crops
            
        Returns:
            List of ClassificationResult objects
        """
        return [self.classify(roi) for roi in rois]
    
    def cleanup(self):
        """Release BPU resources."""
        self.model = None
        logger.info("[BpuClassifier] Cleanup complete")
    
    @property
    def class_names(self) -> List[str]:
        """Return list of class names."""
        return self._class_names
