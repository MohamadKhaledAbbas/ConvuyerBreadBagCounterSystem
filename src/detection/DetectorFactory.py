"""
Detector factory for creating platform-appropriate detectors.
"""

from typing import Optional

from src.config.settings import AppConfig
from src.detection.BaseDetection import BaseDetector
from src.utils.AppLogging import logger
from src.utils.platform import IS_RDK


class DetectorFactory:
    """
    Factory for creating detectors based on platform.
    """
    
    @staticmethod
    def create(
        config: Optional[AppConfig] = None,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.5
    ) -> BaseDetector:
        """
        Create a platform-appropriate detector.
        
        Args:
            config: Application configuration (optional)
            model_path: Override model path (optional)
            confidence_threshold: Minimum detection confidence
            
        Returns:
            BaseDetector instance (BPU on RDK, Ultralytics on Windows)
        """
        if config is None:
            config = AppConfig()
        
        if model_path is None:
            model_path = config.detection_model
        
        if IS_RDK:
            logger.info("[DetectorFactory] Creating BPU detector for RDK platform")
            from src.detection.BpuDetector import BpuDetector
            return BpuDetector(
                model_path=model_path,
                confidence_threshold=confidence_threshold
            )
        else:
            logger.info("[DetectorFactory] Creating Ultralytics detector")
            from src.detection.UltralyticsDetector import UltralyticsDetector
            return UltralyticsDetector(
                model_path=model_path,
                confidence_threshold=confidence_threshold
            )
