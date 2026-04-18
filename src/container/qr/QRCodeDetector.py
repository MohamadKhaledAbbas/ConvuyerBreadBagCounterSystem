"""
QR Code Detector using OpenCV WeChatQRCode (CNN-based).

Uses the WeChat neural-network QR detector from opencv-contrib for robust
detection under perspective distortion, tilt, blur, and compression artefacts.
Falls back to the classic cv2.QRCodeDetector when WeChatQRCode is unavailable.

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
    
    Primary engine: WeChatQRCode (CNN-based, handles perspective/tilt/blur).
    Fallback engine: cv2.QRCodeDetector (traditional, no model files needed).
    
    The detector is transparent to callers — all public methods return the
    same QRDetection objects regardless of which engine is active.
    """
    
    # Valid container QR code values
    VALID_QR_VALUES = {'1', '2', '3', '4', '5'}
    
    def __init__(self, model_dir: Optional[str] = None, engine: str = 'auto'):
        """
        Initialize QR detector.
        
        Args:
            model_dir: Path to WeChatQRCode model files.
            engine: 'wechat' — force CNN detector,
                    'legacy'  — force classic cv2.QRCodeDetector,
                    'auto'    — try WeChatQRCode first, fall back to legacy.
        """
        self.last_detection: Optional[QRDetection] = None
        self.detection_count: int = 0
        self._frame_count: int = 0
        
        self._wechat: Optional[cv2.wechat_qrcode_WeChatQRCode] = None
        self._legacy: Optional[cv2.QRCodeDetector] = None
        self._engine_name: str = "unknown"
        
        mdir = model_dir or _DEFAULT_MODEL_DIR
        
        if engine == 'legacy':
            # Explicit legacy — skip WeChatQRCode entirely
            self._legacy = cv2.QRCodeDetector()
            self._has_multi = hasattr(self._legacy, 'detectAndDecodeMulti')
            self._engine_name = "cv2.QRCodeDetector"
        elif engine == 'wechat':
            self._wechat = self._try_init_wechat(mdir)
            if self._wechat is None:
                raise RuntimeError(
                    f"engine='wechat' requested but WeChatQRCode init failed "
                    f"(model_dir={mdir})"
                )
            self._engine_name = "WeChatQRCode"
        else:
            # auto: try WeChatQRCode, fall back to legacy
            self._wechat = self._try_init_wechat(mdir)
            if self._wechat is not None:
                self._engine_name = "WeChatQRCode"
            else:
                self._legacy = cv2.QRCodeDetector()
                self._has_multi = hasattr(self._legacy, 'detectAndDecodeMulti')
                self._engine_name = "cv2.QRCodeDetector"
                logger.warning(
                    "[QRCodeDetector] WeChatQRCode unavailable — using legacy "
                    "cv2.QRCodeDetector (reduced perspective tolerance)"
                )
        
        logger.info(f"[QRCodeDetector] Initialized with engine={self._engine_name}")

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
    
    @staticmethod
    def _preprocess(frame: np.ndarray) -> np.ndarray:
        """Convert to grayscale and apply adaptive threshold to sharpen QR modules."""
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        # Adaptive threshold recovers sharp black/white QR modules
        # even through MJPG / lossy codec compression artifacts.
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, blockSize=51, C=10
        )
    
    def detect(self, frame: np.ndarray) -> Optional[QRDetection]:
        """
        Detect and decode a single QR code in the given frame.
        
        Tries raw frame first, then falls back to preprocessed version
        for better accuracy on compressed video frames.
        
        Args:
            frame: BGR or grayscale image (numpy array)
        
        Returns:
            QRDetection if a valid container QR code is found, None otherwise
        """
        results = self.detect_all(frame)
        return results[0] if results else None
    
    def detect_all(self, frame: np.ndarray) -> List[QRDetection]:
        """
        Detect all QR codes in the frame.
        
        Uses WeChatQRCode (CNN) when available, otherwise falls back to
        the legacy detector with preprocessing variants.
        
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

        t0 = time.time()
        if self._wechat is not None:
            detections = self._detect_wechat(frame)
        else:
            detections = self._detect_legacy(frame)
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
    
    # ── WeChatQRCode engine ──────────────────────────────────────────────
    
    def _detect_wechat(self, frame: np.ndarray) -> List[QRDetection]:
        """Detect QR codes using WeChatQRCode CNN detector."""
        detections: List[QRDetection] = []
        try:
            t0 = time.time()
            texts, points = self._wechat.detectAndDecode(frame)
            t_call = (time.time() - t0) * 1000
            if _QR_DEBUG and (_QR_DEBUG >= 2 or self._frame_count % 20 == 1):
                logger.info(
                    f"[QR-DBG] wechat.detectAndDecode={t_call:.1f}ms "
                    f"texts={len(texts) if texts else 0}"
                )
            if texts and points is not None:
                for text, pts in zip(texts, points):
                    if not text:
                        continue
                    det = self._build_detection(text, pts)
                    if det:
                        detections.append(det)
        except Exception as e:
            if self._frame_count % 1000 == 0:
                logger.warning(f"[QRCodeDetector] WeChatQRCode error: {e}")
        return detections
    
    # ── Legacy engine (fallback) ─────────────────────────────────────────
    
    def _detect_legacy(self, frame: np.ndarray) -> List[QRDetection]:
        """Detect using cv2.QRCodeDetector with preprocessing fallbacks."""
        t0 = time.time()
        raw_dets = self._detect_multi_legacy(frame)
        t_raw = (time.time() - t0) * 1000

        seen_values = {d.value for d in raw_dets}
        merged = list(raw_dets)

        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        t_pp_passes = []
        for label, preprocess_fn in (
            ('adaptive', lambda g: cv2.adaptiveThreshold(
                g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, blockSize=51, C=10)),
            ('otsu', lambda g: cv2.threshold(
                g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
        ):
            try:
                tp0 = time.time()
                pp = preprocess_fn(gray)
                t_pp = (time.time() - tp0) * 1000
                td0 = time.time()
                pp_dets = self._detect_multi_legacy(pp)
                t_det = (time.time() - td0) * 1000
                t_pp_passes.append((label, t_pp, t_det, len(pp_dets)))
                for d in pp_dets:
                    if d.value not in seen_values:
                        merged.append(d)
                        seen_values.add(d.value)
            except Exception:
                pass

        if _QR_DEBUG and (_QR_DEBUG >= 2 or self._frame_count % 20 == 1):
            extra = ' '.join(
                f"{lbl}=pp{pp:.0f}+det{det:.0f}ms({n})"
                for lbl, pp, det, n in t_pp_passes
            )
            logger.info(
                f"[QR-DBG] legacy raw_pass={t_raw:.0f}ms({len(raw_dets)}) {extra}"
            )

        return merged
    
    def _detect_multi_legacy(self, image: np.ndarray) -> List[QRDetection]:
        """Run legacy detection on a single image."""
        detections: List[QRDetection] = []
        try:
            if self._has_multi:
                retval, decoded_texts, points, _ = self._legacy.detectAndDecodeMulti(image)
                if retval and decoded_texts is not None:
                    for i, text in enumerate(decoded_texts):
                        if not text:
                            continue
                        det = self._build_detection(text, points[i])
                        if det:
                            detections.append(det)
            else:
                decoded_text, points, _ = self._legacy.detectAndDecode(image)
                if decoded_text and points is not None:
                    det = self._build_detection(decoded_text, points)
                    if det:
                        detections.append(det)
        except Exception as e:
            if self._frame_count % 1000 == 0:
                logger.warning(f"[QRCodeDetector] Legacy detection error: {e}")
        return detections
    
    @staticmethod
    def _build_detection(text: str, points: np.ndarray) -> Optional[QRDetection]:
        """Create a QRDetection from decoded text and raw points."""
        pts = points.reshape(-1, 2).astype(np.int32)
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
