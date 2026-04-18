"""
QR Code Detector using OpenCV WeChatQRCode (CNN-based).

Uses the WeChat neural-network QR detector from opencv-contrib for robust
detection under perspective distortion, tilt, blur, and compression artefacts.

Input frames are resized to 640px width before detection for performance
(~4× faster on RDK); detected bounding boxes are scaled back to original
frame coordinates so callers are unaffected.

Container QR codes contain values 1-5 representing the container number.

Usage:
    detector = QRCodeDetector()
    result = detector.detect(frame)
    if result:
        qr_value, bbox, center = result
        print(f"Container {qr_value} at {center}")
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List
import os
import cv2
import numpy as np

import time

from src.utils.AppLogging import logger

# Default model directory relative to project root
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
_DEFAULT_MODEL_DIR = os.path.join(_PROJECT_ROOT, 'data', 'model', 'wechat_qrcode')

# Set QR_DEBUG=1 (or 2 for verbose every-frame) to enable per-pass timing logs.
# Set QR_DUMP_FRAME=/path/to/file.png to dump the next detect frame to disk.
_QR_DEBUG = int(os.environ.get('QR_DEBUG', '0') or '0')
_QR_DUMP_PATH = os.environ.get('QR_DUMP_FRAME', '').strip()


@dataclass
class QRDetection:
    """Result of a QR code detection."""
    value: str                          # Decoded QR code value
    bbox: np.ndarray                    # 4 corner points (4x2 array)
    center: Tuple[int, int]             # Center point (x, y)
    area: float                         # Bounding box area in pixels
    confidence: float = 1.0             # Detection confidence (always 1.0 for OpenCV)
    
    @property
    def qr_number(self) -> Optional[int]:
        """Get the QR code as an integer if it's a valid container number (1-5)."""
        try:
            num = int(self.value.strip())
            if 1 <= num <= 5:
                return num
        except (ValueError, AttributeError):
            pass
        return None


class QRCodeDetector:
    """
    QR code detector for container tracking.
    
    Engine: WeChatQRCode (CNN-based, handles perspective/tilt/blur).
    Frames are resized to ``detect_width`` (default 640) before detection
    for performance; bounding boxes are scaled back to original coordinates.
    """
    
    # Valid container QR code values
    VALID_QR_VALUES = {'1', '2', '3', '4', '5'}

    # Default width for the detection frame (px).  640 gives ~4× speedup
    # over 1280 on RDK while preserving QR decode accuracy.
    DEFAULT_DETECT_WIDTH = 640
    
    def __init__(self, model_dir: Optional[str] = None, engine: str = 'auto',
                 detect_width: int = DEFAULT_DETECT_WIDTH):
        """
        Initialize QR detector.
        
        Args:
            model_dir: Path to WeChatQRCode model files.
            engine: 'wechat' or 'auto' — both use WeChatQRCode CNN detector.
                    'legacy' is no longer supported and will raise an error.
            detect_width: Resize frames to this width before detection (0 = no resize).
        """
        self.last_detection: Optional[QRDetection] = None
        self.detection_count: int = 0
        self._frame_count: int = 0
        self._detect_width: int = max(0, detect_width)
        
        self._wechat: Optional[cv2.wechat_qrcode_WeChatQRCode] = None
        self._engine_name: str = "unknown"
        
        mdir = model_dir or _DEFAULT_MODEL_DIR

        if engine == 'legacy':
            raise ValueError(
                "engine='legacy' is no longer supported. "
                "Use 'wechat' or 'auto' (WeChatQRCode CNN detector)."
            )
        
        self._wechat = self._try_init_wechat(mdir)
        if self._wechat is None:
            raise RuntimeError(
                f"WeChatQRCode init failed (model_dir={mdir}). "
                f"Ensure opencv-contrib-python is installed and model files exist."
            )
        self._engine_name = "WeChatQRCode"
        
        logger.info(
            f"[QRCodeDetector] Initialized engine={self._engine_name} "
            f"detect_width={self._detect_width or 'native'}"
        )

    @property
    def engine_name(self) -> str:
        """Human-readable name of the active detector backend."""
        return self._engine_name
    
    @staticmethod
    def _try_init_wechat(model_dir: str) -> Optional[cv2.wechat_qrcode_WeChatQRCode]:
        """Try to initialize WeChatQRCode; return None on any failure."""
        if not hasattr(cv2, 'wechat_qrcode_WeChatQRCode'):
            logger.info("[QRCodeDetector] cv2.wechat_qrcode_WeChatQRCode not available "
                       "(install opencv-contrib-python)")
            return None
        
        files = {
            'detect_proto': os.path.join(model_dir, 'detect.prototxt'),
            'detect_model': os.path.join(model_dir, 'detect.caffemodel'),
            'sr_proto': os.path.join(model_dir, 'sr.prototxt'),
            'sr_model': os.path.join(model_dir, 'sr.caffemodel'),
        }
        
        missing = [name for name, path in files.items() if not os.path.isfile(path)]
        if missing:
            logger.info(f"[QRCodeDetector] WeChatQRCode model files missing: {missing} "
                       f"(looked in {model_dir})")
            return None
        
        try:
            wechat = cv2.wechat_qrcode_WeChatQRCode(
                files['detect_proto'],
                files['detect_model'],
                files['sr_proto'],
                files['sr_model'],
            )
            logger.info(f"[QRCodeDetector] WeChatQRCode models loaded from {model_dir}")
            return wechat
        except Exception as e:
            logger.warning(f"[QRCodeDetector] WeChatQRCode init failed: {e}")
            return None
    
    # ── WeChatQRCode engine ──────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> Optional[QRDetection]:
        """
        Detect and decode a single QR code in the given frame.
        
        Args:
            frame: BGR or grayscale image (numpy array)
        
        Returns:
            QRDetection if a valid container QR code is found, None otherwise
        """
        results = self.detect_all(frame)
        return results[0] if results else None

    def detect_all(self, frame: np.ndarray) -> List[QRDetection]:
        """
        Detect all QR codes in the frame using WeChatQRCode CNN detector.
        Frames wider than ``detect_width`` are resized down before detection;
        bounding boxes are scaled back to original coordinates.
        
        Args:
            frame: BGR or grayscale image
            
        Returns:
            List of QRDetection objects for all valid container QR codes
        """
        self._frame_count += 1
        
        if frame is None or frame.size == 0:
            return []

        # One-shot frame dump for offline profiling.
        global _QR_DUMP_PATH
        if _QR_DUMP_PATH:
            try:
                cv2.imwrite(_QR_DUMP_PATH, frame)
                logger.info(
                    f"[QR-DBG] Dumped detect frame to {_QR_DUMP_PATH} "
                    f"shape={frame.shape} dtype={frame.dtype}"
                )
            except Exception as e:
                logger.warning(f"[QR-DBG] Frame dump failed: {e}")
            _QR_DUMP_PATH = ''  # one-shot

        if _QR_DEBUG and (self._frame_count == 1 or _QR_DEBUG >= 2):
            logger.info(
                f"[QR-DBG] detect_all input: shape={frame.shape} dtype={frame.dtype} "
                f"contiguous={frame.flags['C_CONTIGUOUS']} engine={self._engine_name}"
            )

        # Resize to detect_width for faster CNN inference.
        # Bounding boxes are scaled back to original coordinates below.
        h_orig, w_orig = frame.shape[:2]
        scale = 1.0
        if self._detect_width and w_orig > self._detect_width:
            scale = w_orig / self._detect_width
            new_h = max(1, int(h_orig / scale))
            frame_det = cv2.resize(frame, (self._detect_width, new_h))
        else:
            frame_det = frame

        t0 = time.time()
        detections = self._detect_wechat(frame_det, scale)
        t_engine = (time.time() - t0) * 1000

        if _QR_DEBUG and (_QR_DEBUG >= 2 or self._frame_count % 20 == 1):
            logger.info(
                f"[QR-DBG] detect_all engine={self._engine_name} "
                f"total={t_engine:.1f}ms found={len(detections)}"
            )
        
        for d in detections:
            self.last_detection = d
            self.detection_count += 1
            if self.detection_count % 100 == 1:
                logger.debug(
                    f"[QRCodeDetector] Detection #{self.detection_count}: "
                    f"QR={d.qr_number}, center={d.center}, area={d.area:.0f}, "
                    f"engine={self._engine_name}"
                )
        
        return detections
    
    def _detect_wechat(self, frame: np.ndarray, scale: float = 1.0) -> List[QRDetection]:
        """Detect QR codes using WeChatQRCode CNN detector.
        
        Args:
            frame: (possibly resized) BGR image.
            scale: factor to multiply bbox coordinates by to map back to original resolution.
        """
        detections: List[QRDetection] = []
        try:
            t0 = time.time()
            texts, points = self._wechat.detectAndDecode(frame)
            t_call = (time.time() - t0) * 1000
            if _QR_DEBUG and (_QR_DEBUG >= 2 or self._frame_count % 20 == 1):
                logger.info(
                    f"[QR-DBG] wechat.detectAndDecode={t_call:.1f}ms "
                    f"texts={len(texts) if texts else 0} scale={scale:.2f}"
                )
            if texts and points is not None:
                for text, pts in zip(texts, points):
                    if not text:
                        continue
                    det = self._build_detection(text, pts, scale)
                    if det:
                        detections.append(det)
        except Exception as e:
            if self._frame_count % 1000 == 0:
                logger.warning(f"[QRCodeDetector] WeChatQRCode error: {e}")
        return detections
    
    @staticmethod
    def _build_detection(text: str, points: np.ndarray, scale: float = 1.0) -> Optional[QRDetection]:
        """Create a QRDetection from decoded text and raw points.
        
        Args:
            text: Decoded QR string.
            points: Corner points from the detector (in detection-frame coords).
            scale: Factor to multiply coords by to map back to original resolution.
        """
        pts = points.reshape(-1, 2)
        if scale != 1.0:
            pts = pts * scale
        pts = pts.astype(np.int32)
        if len(pts) != 4:
            return None
        center_x = int(np.mean(pts[:, 0]))
        center_y = int(np.mean(pts[:, 1]))
        area = cv2.contourArea(pts)
        det = QRDetection(
            value=text.strip(),
            bbox=pts,
            center=(center_x, center_y),
            area=area,
        )
        if det.qr_number is None:
            return None
        return det
    
    def draw_detection(
        self,
        frame: np.ndarray,
        detection: QRDetection,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw QR code detection visualization on a frame.
        
        Args:
            frame: Image to draw on (will be modified in place)
            detection: QRDetection to visualize
            color: BGR color tuple
            thickness: Line thickness
            
        Returns:
            Frame with visualization drawn
        """
        # Draw bounding polygon
        cv2.polylines(frame, [detection.bbox], True, color, thickness)
        
        # Draw center point
        cv2.circle(frame, detection.center, 5, color, -1)
        
        # Draw QR value label
        label = f"QR: {detection.value}"
        label_pos = (detection.center[0] - 30, detection.center[1] - 20)
        cv2.putText(
            frame, label, label_pos,
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
        )
        
        return frame
    
    def reset_stats(self) -> None:
        """Reset detection statistics."""
        self.detection_count = 0
        self._frame_count = 0
        self.last_detection = None
        logger.info("[QRCodeDetector] Statistics reset")
    
    def get_stats(self) -> dict:
        """Get detection statistics."""
        return {
            'engine': self._engine_name,
            'total_frames': self._frame_count,
            'total_detections': self.detection_count,
            'detection_rate': (
                self.detection_count / self._frame_count 
                if self._frame_count > 0 else 0.0
            ),
            'last_detection': (
                {
                    'value': self.last_detection.value,
                    'center': self.last_detection.center,
                    'qr_number': self.last_detection.qr_number,
                }
                if self.last_detection else None
            )
        }
