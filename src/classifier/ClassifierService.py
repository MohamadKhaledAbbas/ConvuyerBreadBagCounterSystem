"""
Classification service that coordinates detection ROIs with classification
and accumulates evidence over a track's lifetime.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np
import cv2
import time

from src.classifier.BaseClassifier import BaseClassifier, ClassificationResult, EvidenceAccumulator
from src.utils.AppLogging import logger
from src.utils.Utils import compute_sharpness, compute_brightness


@dataclass
class ROIQualityConfig:
    """Configuration for ROI quality filtering."""
    min_size: int = 50  # Minimum width/height in pixels
    max_size: int = 500  # Maximum width/height
    min_sharpness: float = 10.0  # Laplacian variance threshold
    min_brightness: float = 30.0  # Mean brightness threshold
    max_brightness: float = 230.0  # Max brightness threshold
    min_aspect_ratio: float = 0.3  # Min width/height ratio
    max_aspect_ratio: float = 3.0  # Max width/height ratio


@dataclass
class TrackClassificationState:
    """Classification state for a tracked object."""
    track_id: int
    evidence: EvidenceAccumulator = field(default_factory=EvidenceAccumulator)
    best_roi: Optional[np.ndarray] = None
    best_roi_quality: float = 0.0
    roi_count: int = 0
    created_at: float = field(default_factory=time.time)
    
    def add_roi(
        self,
        roi: np.ndarray,
        classification: ClassificationResult,
        quality_score: float
    ):
        """Add a classified ROI to the track's evidence."""
        self.roi_count += 1
        self.evidence.add_evidence(classification.class_name, classification.confidence)
        
        # Keep the best quality ROI
        if quality_score > self.best_roi_quality:
            self.best_roi = roi.copy()
            self.best_roi_quality = quality_score


class ClassifierService:
    """
    Coordinates ROI classification with evidence accumulation.
    
    For each tracked object:
    1. Collect ROI crops during track lifetime
    2. Filter by quality (sharpness, brightness, size)
    3. Classify qualifying ROIs
    4. Accumulate evidence to determine final class
    """
    
    def __init__(
        self,
        classifier: BaseClassifier,
        quality_config: Optional[ROIQualityConfig] = None,
        min_evidence_samples: int = 3,
        min_vote_ratio: float = 0.5,
        min_confidence: float = 0.5
    ):
        """
        Initialize classifier service.
        
        Args:
            classifier: Base classifier instance
            quality_config: ROI quality filtering config
            min_evidence_samples: Minimum samples before final decision
            min_vote_ratio: Minimum vote ratio for confident decision
            min_confidence: Minimum confidence for final decision
        """
        self.classifier = classifier
        self.quality_config = quality_config or ROIQualityConfig()
        self.min_evidence_samples = min_evidence_samples
        self.min_vote_ratio = min_vote_ratio
        self.min_confidence = min_confidence
        
        # Track ID -> ClassificationState mapping
        self.track_states: Dict[int, TrackClassificationState] = {}
        
        logger.info(
            f"[ClassifierService] Initialized with min_evidence={min_evidence_samples}, "
            f"min_vote_ratio={min_vote_ratio}, min_confidence={min_confidence}"
        )
    
    def _compute_roi_quality(self, roi: np.ndarray) -> Tuple[float, bool, str]:
        """
        Compute quality score for an ROI.
        
        Returns:
            Tuple of (quality_score, is_valid, rejection_reason)
        """
        h, w = roi.shape[:2]
        cfg = self.quality_config
        
        # Size check
        if w < cfg.min_size or h < cfg.min_size:
            return (0.0, False, "too_small")
        if w > cfg.max_size or h > cfg.max_size:
            return (0.0, False, "too_large")
        
        # Aspect ratio check
        aspect = w / h
        if aspect < cfg.min_aspect_ratio or aspect > cfg.max_aspect_ratio:
            return (0.0, False, "bad_aspect_ratio")
        
        # Compute sharpness
        sharpness = compute_sharpness(roi)
        if sharpness < cfg.min_sharpness:
            return (0.0, False, "too_blurry")
        
        # Compute brightness
        brightness = compute_brightness(roi)
        if brightness < cfg.min_brightness:
            return (0.0, False, "too_dark")
        if brightness > cfg.max_brightness:
            return (0.0, False, "too_bright")
        
        # Compute overall quality score
        # Normalize each factor to 0-1 range
        size_score = min(w, h) / cfg.max_size
        sharpness_score = min(sharpness / 100.0, 1.0)
        brightness_score = 1.0 - abs(brightness - 127) / 127  # Center around middle gray
        
        quality = (size_score * 0.3 + sharpness_score * 0.5 + brightness_score * 0.2)
        
        return (quality, True, "ok")
    
    def process_detection(
        self,
        track_id: int,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> Optional[ClassificationResult]:
        """
        Process a detection ROI for classification.
        
        Args:
            track_id: Track identifier
            frame: Full frame image
            bbox: Bounding box (x1, y1, x2, y2)
            
        Returns:
            ClassificationResult if ROI was classified, None if rejected
        """
        # Get or create track state
        if track_id not in self.track_states:
            self.track_states[track_id] = TrackClassificationState(track_id=track_id)
        
        state = self.track_states[track_id]
        
        # Extract ROI with padding
        x1, y1, x2, y2 = bbox
        pad = 5
        h, w = frame.shape[:2]
        
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        
        roi = frame[y1:y2, x1:x2].copy()
        
        if roi.size == 0:
            return None
        
        # Check ROI quality
        quality, is_valid, reason = self._compute_roi_quality(roi)
        
        if not is_valid:
            logger.debug(f"[ClassifierService] Track {track_id} ROI rejected: {reason}")
            return None
        
        # Classify the ROI
        classification = self.classifier.classify(roi)
        
        # Add to evidence
        state.add_roi(roi, classification, quality)
        
        logger.debug(
            f"[ClassifierService] Track {track_id}: {classification.class_name} "
            f"({classification.confidence:.2f}), samples={state.roi_count}"
        )
        
        return classification
    
    def get_final_classification(
        self,
        track_id: int
    ) -> Optional[Tuple[str, float, float, np.ndarray, List[Tuple[str, int, float]]]]:
        """
        Get final classification for a completed track.
        
        Called when a track ends (object leaves frame).
        
        Args:
            track_id: Track identifier
            
        Returns:
            Tuple of (class_name, confidence, vote_ratio, best_roi, all_candidates)
            or None if insufficient evidence
        """
        if track_id not in self.track_states:
            return None
        
        state = self.track_states[track_id]
        
        # Check minimum evidence
        if state.evidence.total_samples < self.min_evidence_samples:
            logger.warning(
                f"[ClassifierService] Track {track_id} insufficient evidence: "
                f"{state.evidence.total_samples}/{self.min_evidence_samples}"
            )
            # Still return what we have with low confidence indicator
            top = state.evidence.get_top_prediction()
            if top:
                class_name, avg_conf, vote_ratio = top
                candidates = state.evidence.get_all_candidates()
                return (
                    class_name,
                    avg_conf * 0.5,  # Penalize low evidence
                    vote_ratio,
                    state.best_roi,
                    candidates
                )
            return None
        
        # Get top prediction
        top = state.evidence.get_top_prediction()
        
        if top is None:
            return None
        
        class_name, avg_conf, vote_ratio = top
        candidates = state.evidence.get_all_candidates()
        
        # Check confidence thresholds
        if vote_ratio < self.min_vote_ratio:
            logger.info(
                f"[ClassifierService] Track {track_id} low vote ratio: "
                f"{vote_ratio:.2f} < {self.min_vote_ratio}"
            )
        
        if avg_conf < self.min_confidence:
            logger.info(
                f"[ClassifierService] Track {track_id} low confidence: "
                f"{avg_conf:.2f} < {self.min_confidence}"
            )
        
        return (class_name, avg_conf, vote_ratio, state.best_roi, candidates)
    
    def remove_track(self, track_id: int) -> Optional[TrackClassificationState]:
        """
        Remove and return track state.
        
        Args:
            track_id: Track identifier
            
        Returns:
            TrackClassificationState if existed
        """
        return self.track_states.pop(track_id, None)
    
    def get_track_state(self, track_id: int) -> Optional[TrackClassificationState]:
        """Get current state for a track."""
        return self.track_states.get(track_id)
    
    def cleanup(self):
        """Clean up resources."""
        self.track_states.clear()
        self.classifier.cleanup()
        logger.info("[ClassifierService] Cleanup complete")
