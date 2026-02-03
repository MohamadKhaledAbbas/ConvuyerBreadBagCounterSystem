"""
BPU-accelerated detector for RDK platform using Horizon .bin models.
"""

from typing import List, Tuple
import numpy as np

from src.detection.BaseDetection import BaseDetector, Detection
from src.utils.AppLogging import logger


try:
    from hobot_dnn import pyeasy_dnn as dnn
    HAS_BPU = True
except ImportError:
    HAS_BPU = False


class BpuDetector(BaseDetector):
    """
    BPU-accelerated YOLO detector for RDK platform.
    
    Uses Horizon BPU-optimized .bin models for efficient
    edge inference on RDK hardware.
    """
    
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.45,
        input_size: Tuple[int, int] = (640, 640)
    ):
        """
        Initialize BPU detector.
        
        Args:
            model_path: Path to .bin model file
            confidence_threshold: Minimum confidence for detections
            nms_threshold: NMS IoU threshold
            input_size: Model input size (width, height)
        """
        if not HAS_BPU:
            raise ImportError("hobot_dnn not available. BPU detector requires RDK platform.")
        
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        
        # Load model
        logger.info(f"[BpuDetector] Loading model: {model_path}")
        self.models = dnn.load(model_path)
        
        if not self.models:
            raise ValueError(f"Failed to load model: {model_path}")
        
        logger.info(f"[BpuDetector] Model loaded successfully")
    
    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for BPU inference."""
        import cv2
        
        # Resize to model input size
        resized = cv2.resize(frame, self.input_size)
        
        # BPU expects BGR HWC format (no normalization needed for quantized models)
        return resized
    
    def _postprocess(
        self,
        outputs: List[np.ndarray],
        original_size: Tuple[int, int]
    ) -> List[Detection]:
        """
        Post-process BPU inference outputs.
        
        Args:
            outputs: Model output tensors
            original_size: Original frame size (width, height)
            
        Returns:
            List of Detection objects
        """
        detections = []
        orig_w, orig_h = original_size
        input_w, input_h = self.input_size
        
        # Scale factors for mapping back to original size
        scale_x = orig_w / input_w
        scale_y = orig_h / input_h
        
        # Parse YOLO output format
        # Assuming YOLOv5/v8 style output: [batch, num_detections, 5+num_classes]
        # For single-class: [batch, num_detections, 6] -> [x_center, y_center, w, h, obj_conf, class_conf]
        
        output = outputs[0]
        if len(output.shape) == 3:
            output = output[0]  # Remove batch dimension
        
        boxes = []
        confidences = []
        
        for row in output:
            # Extract values based on YOLO format
            if len(row) >= 6:
                x_center, y_center, width, height = row[:4]
                obj_conf = row[4]
                class_conf = row[5] if len(row) > 5 else 1.0
                
                confidence = obj_conf * class_conf
                
                if confidence < self.confidence_threshold:
                    continue
                
                # Convert center format to corner format
                x1 = int((x_center - width / 2) * scale_x)
                y1 = int((y_center - height / 2) * scale_y)
                x2 = int((x_center + width / 2) * scale_x)
                y2 = int((y_center + height / 2) * scale_y)
                
                # Clamp to image bounds
                x1 = max(0, min(x1, orig_w - 1))
                y1 = max(0, min(y1, orig_h - 1))
                x2 = max(0, min(x2, orig_w - 1))
                y2 = max(0, min(y2, orig_h - 1))
                
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    confidences.append(float(confidence))
        
        # Apply NMS
        if boxes:
            import cv2
            indices = cv2.dnn.NMSBoxes(
                [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes],
                confidences,
                self.confidence_threshold,
                self.nms_threshold
            )
            
            if len(indices) > 0:
                indices = indices.flatten() if hasattr(indices, 'flatten') else indices
                for i in indices:
                    detections.append(Detection(
                        bbox=tuple(int(x) for x in boxes[i]),
                        confidence=float(confidences[i]),
                        class_id=0,
                        class_name="bread-bag"
                    ))
        
        return detections
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect bread bags in a frame using BPU.
        
        Args:
            frame: BGR image (numpy array)
            
        Returns:
            List of Detection objects
        """
        original_size = (frame.shape[1], frame.shape[0])
        
        # Preprocess
        input_data = self._preprocess(frame)
        
        # Run inference
        outputs = self.models[0].forward(input_data)
        
        # Extract numpy arrays from output
        output_arrays = [out.buffer for out in outputs]
        
        # Post-process
        detections = self._postprocess(output_arrays, original_size)
        
        return detections
    
    def cleanup(self):
        """Release BPU resources."""
        self.models = None
        logger.info("[BpuDetector] Cleanup complete")
