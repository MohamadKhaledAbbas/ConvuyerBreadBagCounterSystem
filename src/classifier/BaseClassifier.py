"""
Base classifier interface for bread bag type classification.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np


@dataclass
class ClassificationResult:
    """Single classification result."""
    class_id: int
    class_name: str
    confidence: float
    
    def __repr__(self):
        return f"ClassificationResult({self.class_name}: {self.confidence:.3f})"


@dataclass
class EvidenceAccumulator:
    """
    Accumulates classification evidence over multiple frames.
    
    Used to build confidence in classification by collecting
    multiple predictions during object's track lifetime.
    """
    class_counts: Dict[str, int] = field(default_factory=dict)
    class_confidences: Dict[str, List[float]] = field(default_factory=dict)
    total_samples: int = 0
    
    def add_evidence(self, class_name: str, confidence: float):
        """Add a classification prediction as evidence."""
        self.total_samples += 1
        
        if class_name not in self.class_counts:
            self.class_counts[class_name] = 0
            self.class_confidences[class_name] = []
        
        self.class_counts[class_name] += 1
        self.class_confidences[class_name].append(confidence)
    
    def get_top_prediction(self) -> Optional[Tuple[str, float, float]]:
        """
        Get the most likely class based on accumulated evidence.
        
        Returns:
            Tuple of (class_name, average_confidence, vote_ratio) or None
        """
        if not self.class_counts:
            return None
        
        # Find class with most votes
        top_class = max(self.class_counts, key=self.class_counts.get)
        vote_count = self.class_counts[top_class]
        vote_ratio = vote_count / self.total_samples
        
        # Calculate average confidence for top class
        confidences = self.class_confidences[top_class]
        avg_confidence = sum(confidences) / len(confidences)
        
        return (top_class, avg_confidence, vote_ratio)
    
    def get_all_candidates(self) -> List[Tuple[str, int, float]]:
        """
        Get all candidate classes ranked by vote count.
        
        Returns:
            List of (class_name, vote_count, avg_confidence)
        """
        candidates = []
        for class_name, count in sorted(
            self.class_counts.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            avg_conf = sum(self.class_confidences[class_name]) / len(self.class_confidences[class_name])
            candidates.append((class_name, count, avg_conf))
        return candidates
    
    def clear(self):
        """Reset accumulated evidence."""
        self.class_counts.clear()
        self.class_confidences.clear()
        self.total_samples = 0


class BaseClassifier(ABC):
    """
    Abstract base class for bread bag classifiers.
    
    Classifies ROI images into bread bag types:
    - Various bread types (configurable)
    - Rejected class for non-bread items
    """
    
    @abstractmethod
    def classify(self, roi: np.ndarray) -> ClassificationResult:
        """
        Classify a single ROI image.
        
        Args:
            roi: BGR image crop of detected bread bag
            
        Returns:
            ClassificationResult with class and confidence
        """
        pass
    
    @abstractmethod
    def classify_batch(self, rois: List[np.ndarray]) -> List[ClassificationResult]:
        """
        Classify multiple ROI images.
        
        Args:
            rois: List of BGR image crops
            
        Returns:
            List of ClassificationResult objects
        """
        pass
    
    @abstractmethod
    def cleanup(self):
        """Release model resources."""
        pass
    
    @property
    @abstractmethod
    def class_names(self) -> List[str]:
        """Return list of class names."""
        pass
