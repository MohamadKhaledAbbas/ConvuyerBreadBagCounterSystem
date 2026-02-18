"""
Application settings for ConvuyerBreadBagCounterSystem.

Platform-aware configuration with model paths and runtime settings.
"""

import hashlib
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional

from src.utils.platform import IS_RDK


def _parse_bool_env(key: str, default: bool) -> bool:
    """Parse boolean environment variable."""
    val = os.getenv(key, str(default)).lower()
    return val in ('true', '1', 'yes', 'on')


@dataclass
class ModelInfo:
    """Model version tracking."""
    path: str
    version: str
    loaded_at: datetime = field(default_factory=datetime.now)
    checksum: Optional[str] = None
    
    def compute_checksum(self):
        """Compute MD5 checksum of model file."""
        if os.path.exists(self.path):
            try:
                with open(self.path, 'rb') as f:
                    self.checksum = hashlib.md5(f.read()).hexdigest()[:8]
            except Exception:
                self.checksum = None


@dataclass
class AppConfig:
    """
    Application configuration for ConvuyerBreadBagCounterSystem.
    """
    
    APP_VERSION: str = "18-02-2026_v2.4.2"
    
    # Video source for testing
    video_path: str = os.getenv("VIDEO_PATH", "D:\\Recordings\\2026_02_05\\2026_02_16\\output_2026-02-15_19-59-34.h264")
    
    # Platform-specific model paths
    detection_model: str = os.getenv(
        "DETECTION_MODEL",
        "data/model/yolo_nano_detect_v13_bayese_640x640_nv12.bin" if IS_RDK
        else "data/model/yolo_nano_detect_v13.pt"
    )
    classification_model: str = os.getenv(
        "CLASS_MODEL",
        "data/model/yolo_small_classify_v16_bayese_256x256_nv12.bin" if IS_RDK
        else "data/model/yolo_small_classify_v16.pt"
    )
    
    # Database path
    if IS_RDK:
        db_path: str = os.getenv("DB_PATH", "/home/sunrise/ConvuyerBreadCounting/data/db/bag_events.db")
    else:
        db_path: str = os.getenv("DB_PATH", "data/db/bag_events.db")
    
    # Recording directory
    recording_dir: str = os.getenv("RECORDING_DIR", "data/recordings")
    
    # Model versions
    detection_model_version: str = os.getenv("DETECTION_MODEL_VERSION", "v1.0")
    classification_model_version: str = os.getenv("CLASS_MODEL_VERSION", "v1.0")
    
    # Testing mode for OpenCV frame source
    opencv_testing_mode: bool = field(default_factory=lambda: _parse_bool_env("OPENCV_TESTING_MODE", False))
    
    # Frame source configuration for performance tuning
    frame_queue_size: int = int(os.getenv("FRAME_QUEUE_SIZE", "30"))  # Bounded queue prevents memory overflow
    frame_target_fps: Optional[float] = None  # None = use source FPS

    # Classifier class names - bread bag types
    classifier_classes: Dict[int, str] = None
    
    # Detector class names - just bread-bag for convuyer
    detector_classes: Dict[int, str] = None
    
    # Model info objects
    detection_model_info: Optional[ModelInfo] = None
    classification_model_info: Optional[ModelInfo] = None
    
    def __post_init__(self):
        if self.classifier_classes is None:
            self.classifier_classes = {
                0: 'Black_Orange',
                1: 'Blue_Yellow',
                2: 'Bran',
                3: 'Brown_Orange',
                4: 'Green_Yellow',
                5: 'Purple_Yellow',
                6: 'Red_Yellow',
                7: 'Rejected',
                8: 'Wheatberry'
            }
        
        if self.detector_classes is None:
            # For convuyer, we only detect bread bags (single class)
            self.detector_classes = {
                0: 'bread-bag'
            }
        
        # Initialize model info
        self.detection_model_info = ModelInfo(
            path=self.detection_model,
            version=self.detection_model_version,
            loaded_at=datetime.now()
        )
        self.detection_model_info.compute_checksum()
        
        self.classification_model_info = ModelInfo(
            path=self.classification_model,
            version=self.classification_model_version,
            loaded_at=datetime.now()
        )
        self.classification_model_info.compute_checksum()
    
    def get_model_versions(self) -> Dict[str, Any]:
        """Return model version info for logging."""
        return {
            "detection": {
                "version": self.detection_model_version,
                "path": self.detection_model,
                "checksum": self.detection_model_info.checksum if self.detection_model_info else None,
            },
            "classification": {
                "version": self.classification_model_version,
                "path": self.classification_model,
                "checksum": self.classification_model_info.checksum if self.classification_model_info else None,
            }
        }
    
    def log_configuration(self):
        """Log current configuration."""
        from src.utils.AppLogging import logger
        logger.info(f"[Config] App Version: {self.APP_VERSION}")
        logger.info(f"[Config] Detection Model: {self.detection_model}")
        logger.info(f"[Config] Classification Model: {self.classification_model}")
        logger.info(f"[Config] Database: {self.db_path}")


# Global config instance
config = AppConfig()
