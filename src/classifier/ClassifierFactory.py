"""
Classifier factory for creating platform-appropriate classifiers.
"""

from typing import List, Optional

from src.classifier.BaseClassifier import BaseClassifier
from src.config.settings import AppConfig
from src.utils.AppLogging import logger
from src.utils.platform import IS_RDK


class ClassifierFactory:
    """
    Factory for creating classifiers based on platform.
    """
    
    @staticmethod
    def create(
        config: Optional[AppConfig] = None,
        model_path: Optional[str] = None,
        classes: Optional[List[str]] = None
    ) -> BaseClassifier:
        """
        Create a platform-appropriate classifier.
        
        Args:
            config: Application configuration (optional)
            model_path: Override model path (optional)
            classes: Override class names (optional)
            
        Returns:
            BaseClassifier instance (BPU on RDK, Ultralytics on Windows)
        """
        if config is None:
            config = AppConfig()
        
        if model_path is None:
            model_path = config.classification_model
        
        if classes is None:
            classes = config.classifier_classes
        
        if IS_RDK:
            logger.info("[ClassifierFactory] Creating BPU classifier for RDK platform")
            from src.classifier.BpuClassifier import BpuClassifier
            return BpuClassifier(
                model_path=model_path,
                classes=classes
            )
        else:
            logger.info("[ClassifierFactory] Creating Ultralytics classifier")
            from src.classifier.UltralyticsClassifier import UltralyticsClassifier
            return UltralyticsClassifier(
                model_path=model_path,
                classes=classes
            )
