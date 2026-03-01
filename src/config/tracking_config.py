"""
Tracking configuration for ConvuyerBreadBagCounterSystem.

Simplified configuration for convuyer-based tracking.
Unlike the chaotic table environment, convuyer tracking is simpler:
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
    Configuration for convuyer-based bread bag tracking and classification.
    
    Simplified from the original BreadBagCounterSystem which handled:
    - Complex table environments with workers
    - Open/closing/closed state detection
    - Event-centric tracking with centroid association
    
    For convuyer systems, we use simpler logic:
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
    # Conveyor ROI Zone (Detection Region of Interest)
    # ==========================================================================
    # Restricts detection to the conveyor belt area only.  Detections whose
    # center falls outside this zone are dropped BEFORE they reach the tracker,
    # eliminating ~80% of false-positive lost tracks caused by objects on the
    # table edges, operator hands, etc.
    #
    # The camera is fixed, so the conveyor boundaries are constant.
    # Values are in PIXELS for the native frame resolution (typically 1280×720).
    # Set via env vars or the /settings/conveyor-roi endpoint.
    # ==========================================================================

    conveyor_roi_enabled: bool = _parse_bool_env("CONVEYOR_ROI_ENABLED", True)
    """
    Enable conveyor ROI zone filtering.
    When True, detections outside the defined zone are dropped before tracking.
    Default: False (disabled until the user configures the zone boundaries).
    """

    conveyor_roi_x_min: int = _parse_int_env("CONVEYOR_ROI_X_MIN", 370)
    """
    Left boundary of the conveyor zone (pixels from left edge).
    Detections with center_x < this value are dropped.
    Default: 200 px (suitable for 1280-wide frame with conveyor in center).
    """

    conveyor_roi_x_max: int = _parse_int_env("CONVEYOR_ROI_X_MAX", 930)
    """
    Right boundary of the conveyor zone (pixels from left edge).
    Detections with center_x > this value are dropped.
    Default: 1080 px (suitable for 1280-wide frame with conveyor in center).
    """

    conveyor_roi_y_min: int = _parse_int_env("CONVEYOR_ROI_Y_MIN", 0)
    """
    Top boundary of the conveyor zone (pixels from top edge).
    Default: 0 (full height — conveyor typically fills the vertical range).
    """

    conveyor_roi_y_max: int = _parse_int_env("CONVEYOR_ROI_Y_MAX", 720)
    """
    Bottom boundary of the conveyor zone (pixels from top edge).
    Default: 720 (full height for 720p frame).
    """

    conveyor_roi_show_overlay: bool = _parse_bool_env("CONVEYOR_ROI_SHOW_OVERLAY", True)
    """
    Draw the ROI zone boundaries on the annotated frame for debugging.
    Shows a subtle dashed rectangle and shades the excluded areas.
    Only visible when conveyor_roi_enabled is True and display is active.
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
    
    exit_margin_pixels: int = _parse_int_env("EXIT_MARGIN_PIXELS", 30)
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
    # Travel Path Validation (Time-based + Exit Direction)
    # ==========================================================================

    require_full_travel: bool = _parse_bool_env("REQUIRE_FULL_TRAVEL", True)
    """
    Require tracks to meet minimum travel time and exit from the top.
    Tracks that don't meet the time threshold or exit from bottom are invalid.
    """

    min_travel_duration_seconds: float = _parse_float_env("MIN_TRAVEL_DURATION_SECONDS", 2.0)
    """
    Base minimum time (seconds) a track must be visible to be considered valid.
    This uses ADAPTIVE scaling based on where the track first appeared:
    - Tracks appearing at bottom: require 100% of this duration (e.g., 2.0s)
    - Tracks appearing mid-frame: require 50% of this duration (e.g., 1.0s)
    - Tracks appearing near top: require 30% of this duration (e.g., 0.6s)
    
    This adaptive approach prevents valid bags from being rejected due to:
    - Late detection (detector missed them initially)
    - Partial occlusion at entry
    - Lower confidence at bottom of frame
    
    Typical conveyor speeds: objects take 2-3 seconds to traverse full frame.
    """

    exit_zone_ratio: float = _parse_float_env("EXIT_ZONE_RATIO", 0.15)
    """
    Top fraction of frame considered as exit zone (0.15 = top 15%).
    Tracks must exit through this zone to be considered valid.
    """

    bottom_exit_zone_ratio: float = _parse_float_env("BOTTOM_EXIT_ZONE_RATIO", 0.15)
    """
    Bottom fraction of frame considered as invalid exit zone (0.15 = bottom 15%).
    Tracks exiting from this zone are marked invalid (wrong direction).
    """

    # ==========================================================================
    # Ghost Track Recovery (Occlusion Handling)
    # ==========================================================================

    ghost_track_max_age_seconds: float = _parse_float_env("GHOST_TRACK_MAX_AGE_SECONDS", 4.0)
    """
    Maximum time (seconds) to hold a ghost track before finalizing as lost.
    Ghost tracks are lost tracks held in a buffer for possible re-association
    when occluded bags reappear.
    """

    ghost_track_x_tolerance_pixels: float = _parse_float_env("GHOST_TRACK_X_TOLERANCE_PIXELS", 80.0)
    """
    X-axis tolerance (pixels) for ghost track re-association.
    Bags primarily move vertically on conveyor, but camera perspective
    causes slight horizontal shift (~50-80px over full frame travel).
    """

    ghost_track_max_y_gap_ratio: float = _parse_float_env("GHOST_TRACK_MAX_Y_GAP_RATIO", 0.2)
    """
    Maximum Y gap as fraction of frame height for ghost re-association.
    Detection must be within this distance of the ghost's predicted Y position.
    Default increased to 0.3 (30%) to handle fast-moving bags on conveyor.
    """

    # --------------------------------------------------------------------------
    # Ghost Exit Validation (promote near-top ghosts to completed)
    # --------------------------------------------------------------------------

    ghost_exit_validation_enabled: bool = _parse_bool_env("GHOST_EXIT_VALIDATION_ENABLED", True)
    """
    When a ghost track expires, validate whether it would have reached the top
    exit zone based on predicted conveyor trajectory. If yes, promote to
    track_completed instead of track_lost.

    This prevents silent undercounting of bags that lose detection near the top
    of the frame but clearly would have exited.
    """

    ghost_exit_near_top_ratio: float = _parse_float_env("GHOST_EXIT_NEAR_TOP_RATIO", 0.35)
    """
    Maximum Y ratio (from top) for a ghost's last real position to qualify for
    exit validation. The last detected position must be in the upper portion
    of the frame (y <= frame_height * ratio).
    Default: 0.35 = top 35% of frame. For 720p this is y <= 252.
    """

    ghost_exit_min_travel_ratio: float = _parse_float_env("GHOST_EXIT_MIN_TRAVEL_RATIO", 0.40)
    """
    Minimum vertical travel as fraction of frame height required for ghost exit
    validation. The track must have traveled at least this ratio downward-to-upward
    (entry_y - last_y) / frame_height.
    Default: 0.40 = at least 40% of frame traveled. For 720p = 288px.
    """

    ghost_exit_min_hits: int = _parse_int_env("GHOST_EXIT_MIN_HITS", 5)
    """
    Minimum detection hits required for a ghost track to qualify for exit
    validation. Prevents promoting tracks with too little evidence.
    """

    ghost_exit_predicted_top_ratio: float = _parse_float_env("GHOST_EXIT_PREDICTED_TOP_RATIO", 0.20)
    """
    Maximum Y ratio (from top) for the conveyor-predicted position to qualify.
    The predicted position (using conveyor velocity over elapsed time) must be
    in the top portion of the frame (pred_y <= frame_height * ratio).
    Default: 0.20 = top 20% of frame. For 720p this is y <= 144.
    """

    # ==========================================================================
    # Shadow / Merge Detection
    # ==========================================================================

    merge_bbox_growth_threshold: float = _parse_float_env("MERGE_BBOX_GROWTH_THRESHOLD", 1.4)
    """
    Bbox width growth ratio that triggers merge check (1.4 = 40% growth).
    When a surviving track's bbox grows by this ratio, check if it absorbed a neighbor.
    """

    merge_spatial_tolerance_pixels: float = _parse_float_env("MERGE_SPATIAL_TOLERANCE_PIXELS", 50.0)
    """
    Maximum X gap (pixels) between two tracks to consider them adjacent for merge detection.
    """

    merge_y_tolerance_pixels: float = _parse_float_env("MERGE_Y_TOLERANCE_PIXELS", 30.0)
    """
    Maximum Y difference (pixels) between two tracks at merge time.
    """

    # ==========================================================================
    # Entry Type Classification (Diagnostics Only)
    # ==========================================================================

    bottom_entry_zone_ratio: float = _parse_float_env("BOTTOM_ENTRY_ZONE_RATIO", 0.4)
    """
    Bottom fraction of frame considered as bottom_entry zone (0.4 = bottom 40%).
    Tracks created below this line are classified as bottom_entry.
    """

    thrown_entry_min_velocity: float = _parse_float_env("THROWN_ENTRY_MIN_VELOCITY", 15.0)
    """
    Minimum velocity (px/frame) to classify a mid-frame entry as thrown_entry.
    Entries above this threshold are bags thrown onto the conveyor.
    """

    thrown_entry_detection_frames: int = _parse_int_env("THROWN_ENTRY_DETECTION_FRAMES", 5)
    """
    Number of frames to measure initial velocity for thrown_entry classification.
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

    # ROI Diversity Controls (ensure different poses/positions)
    roi_min_frame_spacing: int = _parse_int_env("ROI_MIN_FRAME_SPACING", 5)
    """
    Minimum frames between ROI collections for same track.
    Prevents collecting nearly identical ROIs from consecutive frames.
    Default: 3 frames (at 25 FPS = ~120ms spacing).
    """

    roi_min_position_change: float = _parse_float_env("ROI_MIN_POSITION_CHANGE", 50.0)
    """
    Minimum centroid movement (pixels) required between ROI collections.
    Ensures ROIs are captured at different positions/poses along the conveyor.
    Default: 20 pixels.
    """

    # Gradual Position Penalty (Y-axis based quality adjustment)
    enable_gradual_position_penalty: bool = _parse_bool_env("ENABLE_GRADUAL_POSITION_PENALTY", True)
    """
    Enable gradual position penalty instead of binary upper/lower half penalty.
    Smoother quality degradation from bottom (best) to top (worst).
    """

    position_penalty_start_ratio: float = _parse_float_env("POSITION_PENALTY_START_RATIO", 0.3)
    """
    Y-axis ratio where position penalty starts (0.0=top, 1.0=bottom).
    Default: 0.3 (top 30% of frame) - no penalty below this line.
    Relaxed from 0.5 (center) to allow more of the frame to be penalty-free.
    """

    position_penalty_max_ratio: float = _parse_float_env("POSITION_PENALTY_MAX_RATIO", 0.10)
    """
    Y-axis ratio where maximum penalty is applied (0.0=top, 1.0=bottom).
    Default: 0.10 (top 10% of frame) - full penalty at or above this line.
    """

    position_penalty_min_multiplier: float = _parse_float_env("POSITION_PENALTY_MIN_MULTIPLIER", 0.3)
    """
    Minimum quality multiplier at top of frame (0-1).
    Default: 0.3 (70% quality reduction at top).
    """

    # Brightness Quality Factor (shadow protection)
    optimal_brightness: float = _parse_float_env("OPTIMAL_BRIGHTNESS", 120.0)
    """
    Optimal brightness value for color-based classification (0-255).
    ROIs closer to this value get higher quality scores.
    Default: 120.0 (mid-range, ideal for accurate color representation).
    """

    brightness_penalty_weight: float = _parse_float_env("BRIGHTNESS_PENALTY_WEIGHT", 0.4)
    """
    How strongly brightness deviation penalizes quality score (0-1).
    0.0 = no penalty (brightness only does hard min/max gate).
    1.0 = maximum penalty (ROI at min/max brightness gets 0 quality).
    Default: 0.4 (moderate penalty - shadow ROIs get ~60% quality reduction).
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
    # Run-Length State Machine Parameters
    # ==========================================================================

    smoothing_algorithm: str = _parse_str_env("SMOOTHING_ALGORITHM", "run_length")
    """
    Smoothing algorithm to use.
    'run_length' (default) — RunLengthStateMachine (batch-aware, state-machine approach).
    'window'               — BidirectionalSmoother (legacy sliding-window approach).
    """

    run_length_min_run: int = _parse_int_env("RUN_LENGTH_MIN_RUN", 5)
    """
    Minimum consecutive same-class items required to establish/confirm a batch.
    Controls how quickly the state machine locks onto a batch identity.
    Lower values = faster response; higher values = more stability.
    """

    run_length_max_blip: int = _parse_int_env("RUN_LENGTH_MAX_BLIP", 3)
    """
    Maximum consecutive non-matching items treated as noise (blip).
    If a run of different-class items is shorter than this and then reverts
    to the original batch class, the items are absorbed (overridden) rather
    than treated as a real batch transition.
    """

    run_length_transition_confirm_count: int = _parse_int_env(
        "RUN_LENGTH_TRANSITION_CONFIRM_COUNT", 5
    )
    """
    Number of consecutive new-class items required to confirm a batch transition.
    Until this threshold is reached the state machine stays in TRANSITION and
    holds the items before deciding.
    """

    # ==========================================================================
    # Bidirectional Smoothing Parameters (legacy — used when smoothing_algorithm='window')
    # ==========================================================================
    
    bidirectional_smoothing_enabled: bool = _parse_bool_env("BIDIRECTIONAL_SMOOTHING_ENABLED", True)
    """Enable bidirectional context-aware smoothing."""
    
    bidirectional_buffer_size: int = _parse_int_env("BIDIRECTIONAL_BUFFER_SIZE", 21)
    """
    Buffer size for bidirectional smoothing (must be odd for center-based analysis).
    Default is 21 (10 past + 1 center + 10 future) for center-based context smoothing.
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
    
    bidirectional_inactivity_timeout_ms: float = _parse_float_env("BIDIRECTIONAL_INACTIVITY_TIMEOUT_MS", (30 * 60 * 1000))
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
        "/home/sunrise/ConvuyerBreadCounting/data/spool" if IS_RDK else "data/spool"
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
    
    save_all_rois: bool = _parse_bool_env("SAVE_ALL_ROIS", False)
    """Save all ROI candidates for analysis."""
    
    save_roi_candidates: bool = _parse_bool_env("SAVE_ROI_CANDIDATES", False)
    """Save ROI candidates with metadata."""
    
    roi_candidates_dir: str = _parse_str_env("ROI_CANDIDATES_DIR", "data/roi_candidates")
    """Directory for saved ROI candidates."""
    
    save_rois_by_class: bool = _parse_bool_env("SAVE_ROIS_BY_CLASS", False)
    """Organize saved ROIs by classification result in subdirectories."""

    save_classified_rois: bool = _parse_bool_env("SAVE_CLASSIFIED_ROIS", True)
    """
    Save only the ROIs that are actually used for classification (voting).
    This is narrower than save_all_rois and save_roi_candidates.
    Only the top-K ROIs selected for classification voting are saved.
    """

    classified_rois_dir: str = _parse_str_env("CLASSIFIED_ROIS_DIR", "data/classified_rois")
    """Directory for saving ROIs used in classification."""

    classified_rois_retention_hours: float = _parse_float_env("CLASSIFIED_ROIS_RETENTION_HOURS", 72.0)
    """
    Maximum age (hours) for classified ROI files before automatic deletion.
    Set to 0 to disable time-based retention (not recommended for production).
    Default: 24 hours.
    """

    classified_rois_max_count: int = _parse_int_env("CLASSIFIED_ROIS_MAX_COUNT", 40_000)
    """
    Maximum number of classified ROI files to retain.
    When exceeded, oldest files are deleted first.
    Set to 0 to disable count-based retention.
    Default: 10000 files.
    """

    classified_rois_purge_interval_minutes: float = _parse_float_env("CLASSIFIED_ROIS_PURGE_INTERVAL_MINUTES", 60.0)
    """
    How often to run the purge check (in minutes).
    Default: 60 minutes.
    """

    # ── Lost track snapshot settings ──────────────────────────────────────

    lost_snapshots_dir: str = _parse_str_env("LOST_SNAPSHOTS_DIR", "data/spool/lost_snapshots")
    """Directory for saving frame snapshots when tracks are lost."""

    lost_snapshots_retention_hours: float = _parse_float_env("LOST_SNAPSHOTS_RETENTION_HOURS", 24.0)
    """
    Maximum age (hours) for lost track snapshot files before automatic deletion.
    Default: 24 hours (1 day) to avoid filling SD card.
    """

    lost_snapshots_max_count: int = _parse_int_env("LOST_SNAPSHOTS_MAX_COUNT", 6000)
    """
    Maximum number of lost track snapshot files to retain.
    When exceeded, oldest files are deleted first.
    Default: 6000 files (~600 events × 10 frames each).
    """

    # ── Evidence ring buffer settings ──────────────────────────────────────

    evidence_buffer_size: int = _parse_int_env("EVIDENCE_BUFFER_SIZE", 10)
    """
    Number of annotated frames to keep in the evidence ring buffer.
    When a track is lost/invalid, ALL buffered frames are saved as a
    filmstrip for post-mortem review.
    Default: 10 frames (5 seconds at 0.5 s sample interval).
    This gives ~1 s of context BEFORE the ghost phase starts (ghost
    timeout = 4 s), so the bag is visible while still actively tracked.
    Memory cost: 10 × (640×360×3) ≈ 6.6 MB.
    """

    evidence_sample_interval: float = _parse_float_env("EVIDENCE_SAMPLE_INTERVAL", 0.5)
    """
    Time in seconds between evidence frame samples.
    Lower values give denser coverage but more memory/disk usage.
    Default: 0.5 s → 10 frames covers 5.0 s (ghost timeout + 1 s pre-ghost context).
    """

    @property
    def reject_label_set(self) -> set:
        """Get reject labels as a set."""
        return set(label.strip() for label in self.reject_labels.split(",") if label.strip())


# Global tracking config instance
tracking_config = TrackingConfig()
