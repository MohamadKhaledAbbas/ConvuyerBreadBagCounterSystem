"""
BPU-accelerated detector for RDK platform using Horizon .bin models.

Based on the working implementation from BreadBagCounterSystem.
Optimized for YOLOv8 headless model output format.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.special import softmax

from src.detection.BaseDetection import BaseDetector, Detection
from src.utils.AppLogging import logger

# Import BPU Library
try:
    from hobot_dnn import pyeasy_dnn as dnn
    HAS_BPU = True
except ImportError:
    try:
        from hobot_dnn_rdkx5 import pyeasy_dnn as dnn
        HAS_BPU = True
    except ImportError:
        logger.warning("[BpuDetector] hobot_dnn not found. BPU detection unavailable.")
        dnn = None
        HAS_BPU = False


class BpuDetector(BaseDetector):
    """
    BPU-accelerated YOLO detector for RDK platform.
    
    Uses Horizon BPU-optimized .bin models for efficient
    edge inference on RDK hardware.

    Supports YOLOv8 headless output format with multi-stride detection.
    """
    
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.25,
        nms_threshold: float = 0.45,
        input_size: Tuple[int, int] = (640, 640),
        class_names: Optional[Dict[int, str]] = None
    ):
        """
        Initialize BPU detector.
        
        Args:
            model_path: Path to .bin model file
            confidence_threshold: Minimum confidence for detections
            nms_threshold: NMS IoU threshold
            input_size: Model input size (width, height)
            class_names: Optional dict mapping class_id to class_name
        """
        if not HAS_BPU:
            raise ImportError("hobot_dnn not available. BPU detector requires RDK platform.")
        
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.input_w, self.input_h = input_size

        # Class names (default to single class "bread-bag")
        self._class_names = class_names or {0: "bread-bag"}
        self.classes_num = len(self._class_names)

        # Pre-allocate NV12 buffer
        self.area = self.input_h * self.input_w
        self.nv12_buffer = np.zeros((self.area * 3 // 2,), dtype=np.uint8)

        # Load model
        logger.info(f"[BpuDetector] Loading model: {model_path}")
        self.quantize_model = dnn.load(model_path)

        if not self.quantize_model:
            raise ValueError(f"Failed to load model: {model_path}")
        
        # YOLOv8 headless configuration
        self.reg = 16  # DFL register count
        self.strides = [8, 16, 32]  # Multi-scale strides

        # Pre-calculate constants for postprocessing
        self.conf_thres_raw = -np.log(1 / self.confidence_threshold - 1)
        self.weights_static = np.array([i for i in range(self.reg)]).astype(np.float32)[np.newaxis, np.newaxis, :]

        # Generate anchor grids for each stride
        self.grids = []
        for stride in self.strides:
            grid_h, grid_w = self.input_h // stride, self.input_w // stride
            yv, xv = np.meshgrid(
                np.arange(0.5, grid_h + 0.5),
                np.arange(0.5, grid_w + 0.5),
                indexing='ij'
            )
            grid = np.stack([xv.flatten(), yv.flatten()], axis=1)
            self.grids.append(grid)

        logger.info(f"[BpuDetector] Model loaded, classes: {self.classes_num}")

    @property
    def class_names(self) -> Dict[int, str]:
        return self._class_names

    def _preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, float, float, int, int]:
        """
        Preprocess BGR image to NV12 format for BPU.

        Returns:
            Tuple of (nv12_buffer, x_scale, y_scale, x_shift, y_shift)
        """
        img_h, img_w = img.shape[:2]
        x_scale = min(1.0 * self.input_h / img_h, 1.0 * self.input_w / img_w)
        y_scale = x_scale

        new_w = int(img_w * x_scale)
        new_h = int(img_h * y_scale)

        x_shift = (self.input_w - new_w) // 2
        y_shift = (self.input_h - new_h) // 2

        # Resize with INTER_NEAREST (fast)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # Pad to input size
        padded = cv2.copyMakeBorder(
            resized, y_shift, self.input_h - new_h - y_shift,
            x_shift, self.input_w - new_w - x_shift,
            cv2.BORDER_CONSTANT, value=127
        )

        # Convert BGR to NV12
        yuv_i420 = cv2.cvtColor(padded, cv2.COLOR_BGR2YUV_I420)
        yuv_flat = yuv_i420.reshape(-1)

        # Copy Y plane
        self.nv12_buffer[:self.area] = yuv_flat[:self.area]

        # Interleave UV
        u_start = self.area
        v_start = self.area + (self.area // 4)
        self.nv12_buffer[self.area::2] = yuv_flat[u_start:v_start]
        self.nv12_buffer[self.area + 1::2] = yuv_flat[v_start:]

        return np.ascontiguousarray(self.nv12_buffer), x_scale, y_scale, x_shift, y_shift

    def _preprocess_nv12(
        self, nv12_data: np.ndarray, frame_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, float, float, int, int]:
        """
        Preprocess NV12 frame directly for BPU, avoiding BGR conversion.

        Resizes Y and UV planes separately, then pads to model input size.
        Eliminates the NV12→BGR→NV12 round-trip conversion.

        Args:
            nv12_data: Raw NV12 frame as 2D array (height*1.5, width)
            frame_size: Original (height, width) of the frame

        Returns:
            Tuple of (nv12_buffer, x_scale, y_scale, x_shift, y_shift)
        """
        orig_h, orig_w = frame_size

        x_scale = min(1.0 * self.input_h / orig_h, 1.0 * self.input_w / orig_w)
        y_scale = x_scale

        new_w = int(orig_w * x_scale) & ~1  # Round down to even for NV12 UV plane
        new_h = int(orig_h * y_scale) & ~1  # Round down to even for NV12 UV plane

        # Calculate padding with even alignment for UV plane compatibility
        total_pad_x = self.input_w - new_w
        total_pad_y = self.input_h - new_h
        x_shift = (total_pad_x // 2) & ~1
        y_shift = (total_pad_y // 2) & ~1
        right_pad = total_pad_x - x_shift
        bottom_pad = total_pad_y - y_shift

        # Split NV12 into Y and UV planes
        y_plane = nv12_data[:orig_h, :]           # (orig_h, orig_w)
        uv_plane = nv12_data[orig_h:, :]          # (orig_h//2, orig_w)

        # Resize Y plane (grayscale)
        y_resized = cv2.resize(y_plane, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # Resize UV plane: reshape interleaved UVUV to 2-channel, resize, reshape back
        uv_2ch = uv_plane.reshape(orig_h // 2, orig_w // 2, 2)
        uv_resized = cv2.resize(uv_2ch, (new_w // 2, new_h // 2), interpolation=cv2.INTER_NEAREST)
        uv_resized_flat = uv_resized.reshape(new_h // 2, new_w)

        # Pad Y plane (127 = mid-gray luminance)
        y_padded = cv2.copyMakeBorder(
            y_resized, y_shift, bottom_pad, x_shift, right_pad,
            cv2.BORDER_CONSTANT, value=127
        )

        # Pad UV plane (128 = neutral chroma for both U and V)
        uv_padded = cv2.copyMakeBorder(
            uv_resized_flat,
            y_shift // 2, bottom_pad // 2,
            x_shift, right_pad,
            cv2.BORDER_CONSTANT, value=128
        )

        # Write into pre-allocated NV12 buffer
        self.nv12_buffer[:self.area] = y_padded.reshape(-1)
        self.nv12_buffer[self.area:] = uv_padded.reshape(-1)

        return np.ascontiguousarray(self.nv12_buffer), x_scale, y_scale, x_shift, y_shift

    def _postprocess(
        self,
        outputs: List[np.ndarray],
        x_scale: float,
        y_scale: float,
        x_shift: int,
        y_shift: int,
        orig_shape: Tuple[int, int, int]
    ) -> List[Tuple[int, float, float, float, float, float]]:
        """
        Post-process YOLOv8 headless model outputs.

        Args:
            outputs: 6 output tensors (3 class + 3 bbox for each stride)
            x_scale, y_scale: Scale factors from preprocessing
            x_shift, y_shift: Padding offsets from preprocessing
            orig_shape: Original frame shape (height, width, channels)

        Returns:
            List of tuples (class_id, score, x1, y1, x2, y2)
        """
        # Parse outputs: 3 class tensors + 3 bbox tensors
        clses = [
            outputs[0].reshape(-1, self.classes_num),
            outputs[2].reshape(-1, self.classes_num),
            outputs[4].reshape(-1, self.classes_num)
        ]
        bboxes = [
            outputs[1].reshape(-1, self.reg * 4),
            outputs[3].reshape(-1, self.reg * 4),
            outputs[5].reshape(-1, self.reg * 4)
        ]

        dbboxes, ids, scores = [], [], []

        for cls, bbox, stride, grid in zip(clses, bboxes, self.strides, self.grids):
            max_scores = np.max(cls, axis=1)
            valid_mask = max_scores >= self.conf_thres_raw

            if not np.any(valid_mask):
                continue

            ids.append(np.argmax(cls[valid_mask, :], axis=1))
            scores.append(1 / (1 + np.exp(-max_scores[valid_mask])))

            # DFL distribution decoding
            pred_dist = softmax(bbox[valid_mask].reshape(-1, 4, self.reg), axis=2)
            ltrb = np.sum(pred_dist * self.weights_static, axis=2)

            grid_val = grid[valid_mask]
            x1y1 = grid_val - ltrb[:, 0:2]
            x2y2 = grid_val + ltrb[:, 2:4]
            dbboxes.append(np.hstack([x1y1, x2y2]) * stride)

        if not dbboxes:
            return []

        dbboxes = np.concatenate(dbboxes, axis=0)
        scores = np.concatenate(scores, axis=0)
        ids = np.concatenate(ids, axis=0)

        # NMS per class
        xywh = dbboxes.copy()
        xywh[:, 2:4] = xywh[:, 2:4] - xywh[:, 0:2]

        final_results = []
        for i in range(self.classes_num):
            mask = ids == i
            if not np.any(mask):
                continue

            indices = cv2.dnn.NMSBoxes(
                xywh[mask].tolist(),
                scores[mask].tolist(),
                self.confidence_threshold,
                self.nms_threshold
            )
            
            if len(indices) > 0:
                indices = indices.flatten()
                selected_boxes = dbboxes[mask][indices]
                selected_scores = scores[mask][indices]

                # Transform coordinates back to original image
                boxes_transformed = selected_boxes.copy()
                boxes_transformed[:, [0, 2]] = (boxes_transformed[:, [0, 2]] - x_shift) / x_scale
                boxes_transformed[:, [1, 3]] = (boxes_transformed[:, [1, 3]] - y_shift) / y_scale

                # Clip to image boundaries
                boxes_transformed[:, [0, 2]] = np.clip(boxes_transformed[:, [0, 2]], 0, orig_shape[1])
                boxes_transformed[:, [1, 3]] = np.clip(boxes_transformed[:, [1, 3]], 0, orig_shape[0])

                for j, score in enumerate(selected_scores):
                    x1, y1, x2, y2 = boxes_transformed[j]
                    final_results.append((i, float(score), float(x1), float(y1), float(x2), float(y2)))

        return final_results

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect objects in a frame using BPU.

        Args:
            frame: BGR image (numpy array)
            
        Returns:
            List of Detection objects
        """
        if self.quantize_model is None:
            return []

        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            logger.error("[BpuDetector] Invalid frame input")
            return []

        if len(frame.shape) != 3 or frame.shape[2] != 3:
            logger.error(f"[BpuDetector] Frame must be 3-channel BGR, got shape {frame.shape}")
            return []

        orig_shape = frame.shape

        # Preprocess
        input_data, x_scale, y_scale, x_shift, y_shift = self._preprocess(frame)

        # Run inference
        outputs = self.quantize_model[0].forward(input_data)
        output_arrays = [out.buffer for out in outputs]
        
        # Post-process
        results = self._postprocess(output_arrays, x_scale, y_scale, x_shift, y_shift, orig_shape)

        # Convert to Detection objects
        detections = []
        for class_id, score, x1, y1, x2, y2 in results:
            class_name = self._class_names.get(class_id, f"class_{class_id}")
            detections.append(Detection(
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                confidence=score,
                class_id=class_id,
                class_name=class_name
            ))

        return detections

    def detect_nv12(
        self, nv12_data: np.ndarray, frame_size: Tuple[int, int]
    ) -> List[Detection]:
        """
        Detect objects from a native NV12 frame, avoiding BGR conversion.

        This eliminates the NV12→BGR→NV12 round-trip that occurs when the
        frame source provides NV12 and the BPU model expects NV12.

        Args:
            nv12_data: Raw NV12 frame as 2D array (height*1.5, width)
            frame_size: (height, width) of the original frame

        Returns:
            List of Detection objects
        """
        if self.quantize_model is None:
            return []

        if nv12_data is None or not isinstance(nv12_data, np.ndarray) or nv12_data.size == 0:
            logger.error("[BpuDetector] Invalid NV12 data input")
            return []

        orig_h, orig_w = frame_size

        if nv12_data.shape[0] != orig_h * 3 // 2 or nv12_data.shape[1] != orig_w:
            logger.error(
                f"[BpuDetector] NV12 data shape {nv12_data.shape} does not match "
                f"frame_size {frame_size} (expected ({orig_h * 3 // 2}, {orig_w}))"
            )
            return []

        orig_shape = (orig_h, orig_w, 3)  # Virtual shape for postprocessing

        # Preprocess NV12 directly (no BGR conversion)
        input_data, x_scale, y_scale, x_shift, y_shift = self._preprocess_nv12(
            nv12_data, frame_size
        )

        # Run inference
        outputs = self.quantize_model[0].forward(input_data)
        output_arrays = [out.buffer for out in outputs]

        # Post-process
        results = self._postprocess(
            output_arrays, x_scale, y_scale, x_shift, y_shift, orig_shape
        )

        # Convert to Detection objects
        detections = []
        for class_id, score, x1, y1, x2, y2 in results:
            class_name = self._class_names.get(class_id, f"class_{class_id}")
            detections.append(Detection(
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                confidence=score,
                class_id=class_id,
                class_name=class_name
            ))

        return detections
    
    def cleanup(self):
        """Release BPU resources."""
        self.quantize_model = None
        logger.info("[BpuDetector] Cleanup complete")
