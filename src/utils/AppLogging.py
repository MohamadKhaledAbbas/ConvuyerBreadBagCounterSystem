"""
Centralized logging configuration for ConveyerBreadBagCounterSystem.

Provides both standard logging and structured logging for metrics.
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path


def setup_logging(log_dir: str = "data/logs") -> logging.Logger:
    """Setup application logging with file and console handlers."""
    # Create log directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate timestamped log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"conveyer_counter_{timestamp}.log")
    
    # Create logger
    logger = logging.getLogger("ConveyerBreadBagCounter")
    logger.setLevel(logging.DEBUG)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # File handler - detailed logs
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    
    # Console handler - info and above
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


class StructuredLogger:
    """Structured logging for pipeline metrics and events."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def _log_structured(self, event_type: str, level: int, **kwargs):
        """Log a structured event."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            **kwargs
        }
        self.logger.log(level, json.dumps(event))
    
    def track_created(self, track_id: int, bbox: List[float], frame_index: int, confidence: float):
        """Log track creation."""
        self._log_structured(
            "track_created",
            logging.INFO,
            track_id=track_id,
            bbox=bbox,
            frame_index=frame_index,
            confidence=confidence
        )
    
    def track_updated(self, track_id: int, bbox: List[float], frame_index: int, roi_count: int):
        """Log track update."""
        self._log_structured(
            "track_updated",
            logging.DEBUG,
            track_id=track_id,
            bbox=bbox,
            frame_index=frame_index,
            roi_count=roi_count
        )
    
    def track_completed(self, track_id: int, frame_index: int, roi_count: int, duration_frames: int):
        """Log track completion (object left frame)."""
        self._log_structured(
            "track_completed",
            logging.INFO,
            track_id=track_id,
            frame_index=frame_index,
            roi_count=roi_count,
            duration_frames=duration_frames
        )
    
    def classification_result(self, track_id: int, label: str, confidence: float, 
                              candidates_count: int, metadata: Optional[Dict] = None):
        """Log classification result."""
        self._log_structured(
            "classification_result",
            logging.INFO,
            track_id=track_id,
            label=label,
            confidence=confidence,
            candidates_count=candidates_count,
            metadata=metadata or {}
        )
    
    def classification_candidate(self, track_id: int, candidate_idx: int, label: str,
                                  confidence: float, sharpness: float, frame_index: int,
                                  relative_time: float = 0.0, contribution: float = 0.0,
                                  bbox: Optional[List] = None):
        """Log individual classification candidate."""
        self._log_structured(
            "classification_candidate",
            logging.DEBUG,
            track_id=track_id,
            candidate_idx=candidate_idx,
            label=label,
            confidence=confidence,
            sharpness=sharpness,
            frame_index=frame_index,
            relative_time=relative_time,
            contribution=contribution,
            bbox=bbox
        )
    
    def roi_rejected(self, track_id: int, reason: str, **kwargs):
        """Log ROI rejection."""
        self._log_structured(
            "roi_rejected",
            logging.DEBUG,
            track_id=track_id,
            reason=reason,
            **kwargs
        )
    
    def pipeline_error(self, component: str, operation: str, error_type: str,
                       error_message: str, affected_ids: Optional[List] = None,
                       context: Optional[Dict] = None):
        """Log pipeline error."""
        self._log_structured(
            "pipeline_error",
            logging.ERROR,
            component=component,
            operation=operation,
            error_type=error_type,
            error_message=error_message,
            affected_ids=affected_ids,
            context=context or {}
        )
    
    def smoothing_applied(self, track_id: int, original_label: str, smoothed_label: str,
                          confidence: float, reason: str):
        """Log bidirectional smoothing application."""
        self._log_structured(
            "smoothing_applied",
            logging.INFO,
            track_id=track_id,
            original_label=original_label,
            smoothed_label=smoothed_label,
            confidence=confidence,
            reason=reason
        )


# Global logger instances
logger = setup_logging()
structured_logger = StructuredLogger(logger)


def get_log_file_paths() -> Dict[str, str]:
    """Get paths to current log files."""
    log_dir = "data/logs"
    if not os.path.exists(log_dir):
        return {}
    
    files = sorted(Path(log_dir).glob("conveyer_counter_*.log"), reverse=True)
    if files:
        return {"main_log": str(files[0])}
    return {}
