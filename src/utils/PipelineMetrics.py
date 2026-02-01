"""
Pipeline metrics for monitoring system performance.
"""

import time
import threading
from typing import Dict, Any, List
from dataclasses import dataclass, field
from collections import deque
import statistics

from src.utils.AppLogging import logger


@dataclass
class DetectionMetrics:
    """Metrics for detection stage."""
    total_frames: int = 0
    total_detections: int = 0
    filtered_low_confidence: int = 0
    avg_processing_time_ms: float = 0.0
    processing_times: deque = field(default_factory=lambda: deque(maxlen=100))


@dataclass
class TrackingMetrics:
    """Metrics for tracking stage."""
    tracks_created: int = 0
    tracks_completed: int = 0
    tracks_expired: int = 0
    avg_track_duration_frames: float = 0.0
    track_durations: deque = field(default_factory=lambda: deque(maxlen=100))


@dataclass
class ClassificationMetrics:
    """Metrics for classification stage."""
    total_classified: int = 0
    classifications_by_label: Dict[str, int] = field(default_factory=dict)
    avg_confidence: float = 0.0
    avg_candidates_per_track: float = 0.0
    confidences: deque = field(default_factory=lambda: deque(maxlen=100))
    candidates_counts: deque = field(default_factory=lambda: deque(maxlen=100))


@dataclass
class QualityMetrics:
    """ROI quality metrics."""
    total_rois_collected: int = 0
    rois_rejected_size: int = 0
    rois_rejected_brightness: int = 0
    rois_rejected_sharpness: int = 0
    avg_sharpness: float = 0.0
    sharpness_values: deque = field(default_factory=lambda: deque(maxlen=100))


class PipelineMetrics:
    """
    Centralized metrics collection for the counting pipeline.
    
    Thread-safe metrics collection and periodic logging.
    """
    
    def __init__(self, log_interval_seconds: float = 30.0):
        self.detection = DetectionMetrics()
        self.tracking = TrackingMetrics()
        self.classification = ClassificationMetrics()
        self.quality = QualityMetrics()
        
        self._lock = threading.Lock()
        self._last_log_time = time.time()
        self._log_interval = log_interval_seconds
    
    def record_detection(self, detections_count: int, processing_time_ms: float):
        """Record detection frame processing."""
        with self._lock:
            self.detection.total_frames += 1
            self.detection.total_detections += detections_count
            self.detection.processing_times.append(processing_time_ms)
            if self.detection.processing_times:
                self.detection.avg_processing_time_ms = statistics.mean(
                    self.detection.processing_times
                )
        self._maybe_log()
    
    def record_detection_filtered(self, reason: str = "low_confidence"):
        """Record filtered detection."""
        with self._lock:
            if reason == "low_confidence":
                self.detection.filtered_low_confidence += 1
    
    def record_track_created(self):
        """Record new track creation."""
        with self._lock:
            self.tracking.tracks_created += 1
    
    def record_track_completed(self, duration_frames: int):
        """Record track completion."""
        with self._lock:
            self.tracking.tracks_completed += 1
            self.tracking.track_durations.append(duration_frames)
            if self.tracking.track_durations:
                self.tracking.avg_track_duration_frames = statistics.mean(
                    self.tracking.track_durations
                )
    
    def record_track_expired(self):
        """Record track expiration (insufficient data)."""
        with self._lock:
            self.tracking.tracks_expired += 1
    
    def record_classification(self, label: str, confidence: float, candidates_count: int):
        """Record classification result."""
        with self._lock:
            self.classification.total_classified += 1
            self.classification.classifications_by_label[label] = (
                self.classification.classifications_by_label.get(label, 0) + 1
            )
            self.classification.confidences.append(confidence)
            self.classification.candidates_counts.append(candidates_count)
            
            if self.classification.confidences:
                self.classification.avg_confidence = statistics.mean(
                    self.classification.confidences
                )
            if self.classification.candidates_counts:
                self.classification.avg_candidates_per_track = statistics.mean(
                    self.classification.candidates_counts
                )
    
    def record_roi_collected(self, sharpness: float):
        """Record ROI collection."""
        with self._lock:
            self.quality.total_rois_collected += 1
            self.quality.sharpness_values.append(sharpness)
            if self.quality.sharpness_values:
                self.quality.avg_sharpness = statistics.mean(
                    self.quality.sharpness_values
                )
    
    def record_roi_rejected(self, reason: str):
        """Record ROI rejection."""
        with self._lock:
            if reason == "size":
                self.quality.rois_rejected_size += 1
            elif reason == "brightness":
                self.quality.rois_rejected_brightness += 1
            elif reason == "sharpness":
                self.quality.rois_rejected_sharpness += 1
    
    def _maybe_log(self):
        """Log metrics if interval has passed."""
        current_time = time.time()
        if current_time - self._last_log_time >= self._log_interval:
            self._log_metrics()
            self._last_log_time = current_time
    
    def _log_metrics(self):
        """Log current metrics summary."""
        with self._lock:
            logger.info(
                f"[PipelineMetrics] Detection: frames={self.detection.total_frames}, "
                f"detections={self.detection.total_detections}, "
                f"avg_time={self.detection.avg_processing_time_ms:.1f}ms"
            )
            logger.info(
                f"[PipelineMetrics] Tracking: created={self.tracking.tracks_created}, "
                f"completed={self.tracking.tracks_completed}, "
                f"expired={self.tracking.tracks_expired}, "
                f"avg_duration={self.tracking.avg_track_duration_frames:.1f}frames"
            )
            logger.info(
                f"[PipelineMetrics] Classification: total={self.classification.total_classified}, "
                f"avg_conf={self.classification.avg_confidence:.2f}, "
                f"avg_candidates={self.classification.avg_candidates_per_track:.1f}"
            )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        with self._lock:
            return {
                "detection": {
                    "total_frames": self.detection.total_frames,
                    "total_detections": self.detection.total_detections,
                    "avg_processing_time_ms": self.detection.avg_processing_time_ms,
                },
                "tracking": {
                    "tracks_created": self.tracking.tracks_created,
                    "tracks_completed": self.tracking.tracks_completed,
                    "tracks_expired": self.tracking.tracks_expired,
                    "avg_duration_frames": self.tracking.avg_track_duration_frames,
                },
                "classification": {
                    "total_classified": self.classification.total_classified,
                    "by_label": dict(self.classification.classifications_by_label),
                    "avg_confidence": self.classification.avg_confidence,
                },
                "quality": {
                    "total_rois": self.quality.total_rois_collected,
                    "avg_sharpness": self.quality.avg_sharpness,
                }
            }


# Global metrics instance
pipeline_metrics = PipelineMetrics()
