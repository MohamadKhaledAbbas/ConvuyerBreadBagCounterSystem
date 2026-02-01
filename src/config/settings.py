"""
Application settings for ConveyerBreadBagCounterSystem.

Platform-aware configuration with model paths and runtime settings.
"""

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional
import hashlib

from src.utils.platform import IS_RDK, IS_WINDOWS


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
    Application configuration for ConveyerBreadBagCounterSystem.
    """
    
    APP_VERSION: str = "2.0.0"
    
    # Video source for testing
    video_path: str = os.getenv("VIDEO_PATH", "data/test_video.mp4")
    
    # Platform-specific model paths
    detection_model: str = os.getenv(
        "DETECTION_MODEL",
        "data/model/detect_conveyer_640x640_nv12.bin" if IS_RDK
        else "data/model/detect_conveyer.pt"
    )
    classification_model: str = os.getenv(
        "CLASS_MODEL",
        "data/model/classify_bread_224x224_nv12.bin" if IS_RDK
        else "data/model/classify_bread.pt"
    )
    
    # Database path
    if IS_RDK:
        db_path: str = os.getenv("DB_PATH", "/home/sunrise/ConveyerCounting/data/db/bag_events.db")
    else:
        db_path: str = os.getenv("DB_PATH", "data/db/bag_events.db")
    
    # Recording directory
    recording_dir: str = os.getenv("RECORDING_DIR", "data/recordings")
    
    # Model versions
    detection_model_version: str = os.getenv("DETECTION_MODEL_VERSION", "v1.0")
    classification_model_version: str = os.getenv("CLASS_MODEL_VERSION", "v1.0")
    
    # Testing mode for OpenCV frame source
    opencv_testing_mode: bool = field(default_factory=lambda: _parse_bool_env("OPENCV_TESTING_MODE", False))
    
    # Classifier class names - bread bag types
    classifier_classes: Dict[int, str] = None
    
    # Detector class names - just bread-bag for conveyer
    detector_classes: Dict[int, str] = None
    
    # Model info objects
    detection_model_info: Optional[ModelInfo] = None
    classification_model_info: Optional[ModelInfo] = None
    
    def __post_init__(self):
        if self.classifier_classes is None:
            self.classifier_classes = {
                0: 'Blue_Yellow',
                1: 'Bran',
                2: 'Brown_Orange_Family',
                3: 'Green_Yellow',
                4: 'Red_Yellow',
                5: 'Rejected',
                6: 'Wheatberry'
            }
        
        if self.detector_classes is None:
            # For conveyer, we only detect bread bags (single class)
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
