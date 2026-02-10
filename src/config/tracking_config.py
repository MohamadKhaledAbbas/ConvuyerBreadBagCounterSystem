"""
Tracking configuration for ConveyerBreadBagCounterSystem.

Simplified configuration for conveyer-based tracking.
Unlike the chaotic table environment, conveyer tracking is simpler:
- Objects appear on one side
- Move linearly across the frame
- Disappear on the other side

This allows for simpler tracking logic without complex state machines.
"""

import os
from dataclasses import dataclass

from src.utils.platform import IS_RDK


def _parse_bool_env(key: str, default: bool) -> bool:
    """Parse boolean environment variable."""
    val = os.getenv(key, str(default)).lower()
    return val in ('true', '1', 'yes', 'on')


def _parse_float_env(key: str, default: float) -> float:
    """Parse float environment variable."""
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


def _parse_int_env(key: str, default: int) -> int:
    """Parse integer environment variable."""
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def _parse_str_env(key: str, default: str) -> str:
    """Parse string environment variable."""
    return os.getenv(key, default)


@dataclass
class TrackingConfig:
    """
    Configuration for conveyer-based bread bag tracking and classification.
    
    Simplified from the original BreadBagCounterSystem which handled:
    - Complex table environments with workers
    - Open/closing/closed state detection
    - Event-centric tracking with centroid association
    
    For conveyer systems, we use simpler logic:
    - Single detection class (bread-bag)
    - Linear motion tracking
    - ROI collection during track lifetime
    - Classification after track ends (object leaves frame)
    """
    
    # ==========================================================================
    # Target Performance
    # ==========================================================================
    
    target_fps: float = _parse_float_env("TARGET_FPS", 25.0)
    """Target frames per second for processing."""
    
    # ==========================================================================
    # Detection Thresholds
    # ==========================================================================
    
    min_detection_confidence: float = _parse_float_env("MIN_DETECTION_CONFIDENCE", 0.4)
    """Minimum confidence threshold for detections."""
    
    min_confidence_new_track: float = _parse_float_env("MIN_CONFIDENCE_NEW_TRACK", 0.7)
    """
    Minimum confidence required to create a NEW track.
    This is higher than min_detection_confidence to prevent creating tracks from noise.
    Existing tracks can still be updated with lower confidence detections.
    """

    # ==========================================================================
    # Tracking Parameters
    # ==========================================================================
    
    iou_threshold: float = _parse_float_env("IOU_THRESHOLD", 0.3)
    """IoU threshold for track association."""
    
    max_frames_without_detection: int = _parse_int_env("MAX_FRAMES_WITHOUT_DETECTION", 15)
    """
    Maximum frames a track can survive without detection.
    After this, the track is considered complete (object left frame).
    """
    
    min_track_duration_frames: int = _parse_int_env("MIN_TRACK_DURATION_FRAMES", 5)
    """
    Minimum frames a track must exist for classification.
    Tracks shorter than this are discarded as noise.
    """
    
    max_active_tracks: int = _parse_int_env("MAX_ACTIVE_TRACKS", 20)
    """Maximum concurrent tracks to prevent memory issues."""
    
    exit_margin_pixels: int = _parse_int_env("EXIT_MARGIN_PIXELS", 20)
    """
    Margin from frame edge to consider a track as exiting.
    Objects within this margin are considered to be leaving the frame.
    """

    # ==========================================================================
    # Enhanced Tracking Parameters (Multi-criteria matching)
    # ==========================================================================

    use_multi_criteria_matching: bool = _parse_bool_env("USE_MULTI_CRITERIA_MATCHING", True)
    """Enable multi-criteria cost function (IoU + centroid + motion + size)."""

    use_second_stage_matching: bool = _parse_bool_env("USE_SECOND_STAGE_MATCHING", True)
    """Enable second-stage centroid-based matching for missed tracks."""

    velocity_smoothing_alpha: float = _parse_float_env("VELOCITY_SMOOTHING_ALPHA", 0.3)
    """
    Exponential moving average alpha for velocity smoothing.
    Higher values (0.5-0.8) = more responsive, lower values (0.1-0.3) = more stable.
    """

    second_stage_max_distance: float = _parse_float_env("SECOND_STAGE_MAX_DISTANCE", 150.0)
    """Maximum centroid distance (pixels) for second-stage matching."""

    second_stage_threshold: float = _parse_float_env("SECOND_STAGE_THRESHOLD", 0.8)
    """Cost threshold for second-stage matching (0-1, lower = stricter)."""

    # ==========================================================================
    # Travel Path Validation (Bottom-to-Top)
    # ==========================================================================

    require_full_travel: bool = _parse_bool_env("REQUIRE_FULL_TRAVEL", True)
    """
    Require tracks to travel from entry zone (bottom) to exit zone (top).
    Tracks appearing mid-frame or not reaching the top are ignored.
    """

    entry_zone_ratio: float = _parse_float_env("ENTRY_ZONE_RATIO", 0.25)
    """
    Bottom fraction of frame considered as entry zone (0.25 = bottom 25%).
    Tracks must first appear within this zone to be considered valid.
    """

    exit_zone_ratio: float = _parse_float_env("EXIT_ZONE_RATIO", 0.15)
    """
    Top fraction of frame considered as exit zone (0.15 = top 15%).
    Tracks must exit through this zone to be considered valid.
    """

    # ==========================================================================
    # ROI Collection Parameters
    # ==========================================================================
    
    max_rois_per_track: int = _parse_int_env("MAX_ROIS_PER_TRACK", 15)
    """Maximum ROI candidates to collect per track."""
    
    top_k_candidates: int = _parse_int_env("TOP_K_CANDIDATES", 8)
    """Number of best ROIs to use for classification."""
    
    min_roi_size: int = _parse_int_env("MIN_ROI_SIZE", 40)
    """Minimum ROI dimension (width or height) in pixels."""
    
    max_roi_size: int = _parse_int_env("MAX_ROI_SIZE", 500)
    """Maximum ROI dimension in pixels."""
    
    min_roi_aspect_ratio: float = _parse_float_env("MIN_ROI_ASPECT_RATIO", 0.4)
    """Minimum aspect ratio (width/height) for valid ROI."""
    
    max_roi_aspect_ratio: float = _parse_float_env("MAX_ROI_ASPECT_RATIO", 2.5)
    """Maximum aspect ratio for valid ROI."""
    
    min_sharpness: float = _parse_float_env("MIN_SHARPNESS", 100.0)
    """Minimum Laplacian variance for ROI sharpness."""
    
    min_mean_brightness: float = _parse_float_env("MIN_MEAN_BRIGHTNESS", 50.0)
    """Minimum mean brightness for valid ROI."""
    
    max_mean_brightness: float = _parse_float_env("MAX_MEAN_BRIGHTNESS", 200.0)
    """Maximum mean brightness for valid ROI."""
    
    roi_padding_ratio: float = _parse_float_env("ROI_PADDING_RATIO", 0.05)
    """Padding ratio to add around detected bounding box for ROI."""
    
    # Temporal/Distance weighting (earlier ROIs = closer to camera = better quality)
    enable_temporal_weighting: bool = _parse_bool_env("ENABLE_TEMPORAL_WEIGHTING", True)
    """Enable temporal weighting - earlier ROIs get higher quality scores."""

    temporal_decay_rate: float = _parse_float_env("TEMPORAL_DECAY_RATE", 0.15)
    """
    Temporal decay rate for ROI quality (0-1).
    0 = no decay, 1 = strong decay.
    0.15 means 15% quality reduction from first to last ROI.
    """

    # ==========================================================================
    # Classification Parameters
    # ==========================================================================
    
    min_candidates_for_classification: int = _parse_int_env("MIN_CANDIDATES_FOR_CLASSIFICATION", 2)
    """Minimum ROI candidates required for classification."""
    
    min_total_evidence_score: float = _parse_float_env("MIN_TOTAL_EVIDENCE_SCORE", 1.0)
    """Minimum evidence score for classification acceptance."""
    
    evidence_ratio_threshold: float = _parse_float_env("EVIDENCE_RATIO_THRESHOLD", 1.5)
    """Minimum ratio of winner to runner-up score."""
    
    high_confidence_threshold: float = _parse_float_env("HIGH_CONFIDENCE_THRESHOLD", 0.85)
    """Threshold above which classification is considered high confidence."""
    
    reject_labels: str = _parse_str_env("REJECT_LABELS", "Rejected,Unknown")
    """Comma-separated list of labels to filter from voting."""
    
    # ==========================================================================
    # Bidirectional Smoothing Parameters
    # ==========================================================================
    
    bidirectional_smoothing_enabled: bool = _parse_bool_env("BIDIRECTIONAL_SMOOTHING_ENABLED", True)
    """Enable bidirectional context-aware smoothing."""
    
    bidirectional_buffer_size: int = _parse_int_env("BIDIRECTIONAL_BUFFER_SIZE", 7)
    """
    Buffer size for bidirectional smoothing (should be odd for symmetry).
    This is the STABILIZER that corrects misclassifications.
    - Classifications are shown immediately as TENTATIVE
    - After smoothing, they become CONFIRMED and persisted to DB
    - UI shows both states clearly
    """

    bidirectional_confidence_threshold: float = _parse_float_env("BIDIRECTIONAL_CONFIDENCE_THRESHOLD", 0.90)
    """Confidence threshold above which smoothing is bypassed."""
    
    bidirectional_context_agreement_ratio: float = _parse_float_env("BIDIRECTIONAL_CONTEXT_AGREEMENT_RATIO", 0.8)
    """Fraction of context items that must agree for override."""
    
    bidirectional_uncertain_override_ratio: float = _parse_float_env("BIDIRECTIONAL_UNCERTAIN_OVERRIDE_RATIO", 0.5)
    """Relaxed ratio for overriding Uncertain/Unknown labels."""
    
    bidirectional_batch_transition_protection: bool = _parse_bool_env("BIDIRECTIONAL_BATCH_TRANSITION_PROTECTION", True)
    """Protect batch transitions from incorrect smoothing."""
    
    bidirectional_inactivity_timeout_ms: float = _parse_float_env("BIDIRECTIONAL_INACTIVITY_TIMEOUT_MS", 300_000.0)
    """Timeout for flushing buffer on inactivity."""
    
    # ==========================================================================
    # Evidence Accumulation
    # ==========================================================================
    
    evidence_accumulation_enabled: bool = _parse_bool_env("EVIDENCE_ACCUMULATION_ENABLED", True)
    """Use trust-weighted log-evidence accumulation."""
    
    sharpness_weight_scale: float = _parse_float_env("SHARPNESS_WEIGHT_SCALE", 100.0)
    """Scale factor for sharpness in trust calculation."""
    
    # ==========================================================================
    # Spool Configuration (from original system)
    # ==========================================================================
    
    spool_dir: str = _parse_str_env(
        "SPOOL_DIR",
        "/home/sunrise/ConveyerCounting/data/spool" if IS_RDK else "data/spool"
    )
    """Directory for spool segment files."""
    
    spool_segment_duration: float = _parse_float_env("SPOOL_SEGMENT_DURATION", 5.0)
    """Target segment duration in seconds."""
    
    spool_max_segment_duration: float = _parse_float_env("SPOOL_MAX_SEGMENT_DURATION", 10.0)
    """Maximum segment duration before forced rotation."""
    
    spool_retention_seconds: float = _parse_float_env("SPOOL_RETENTION_SECONDS", 180.0)
    """Maximum age of spool segments before deletion."""
    
    spool_recorder_queue_size: int = _parse_int_env("SPOOL_RECORDER_QUEUE_SIZE", 100)
    """Frame queue size for spool recorder."""
    
    spool_recorder_stats_interval: float = _parse_float_env("SPOOL_RECORDER_STATS_INTERVAL", 10.0)
    """Interval for recorder statistics logging."""
    
    spool_processor_target_fps: float = _parse_float_env("SPOOL_PROCESSOR_TARGET_FPS", 30.0)
    """Target FPS for spool processor."""
    
    # ==========================================================================
    # ROI Saving for Debug/Analysis
    # ==========================================================================
    
    save_all_rois: bool = _parse_bool_env("SAVE_ALL_ROIS", True)
    """Save all ROI candidates for analysis."""
    
    save_roi_candidates: bool = _parse_bool_env("SAVE_ROI_CANDIDATES", True)
    """Save ROI candidates with metadata."""
    
    roi_candidates_dir: str = _parse_str_env("ROI_CANDIDATES_DIR", "data/roi_candidates")
    """Directory for saved ROI candidates."""
    
    save_rois_by_class: bool = _parse_bool_env("SAVE_ROIS_BY_CLASS", True)
    """Organize saved ROIs by classification result in subdirectories."""

    @property
    def reject_label_set(self) -> set:
        """Get reject labels as a set."""
        return set(label.strip() for label in self.reject_labels.split(",") if label.strip())


# Global tracking config instance
tracking_config = TrackingConfig()
