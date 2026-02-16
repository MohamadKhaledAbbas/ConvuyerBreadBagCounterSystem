"""
Bidirectional sliding window smoother.

Ported from v1 BreadBagCounterSystem - this logic is kept as it handles
low confidence classification scenarios by allowing window-level corrections.

When a sliding window has low confidence classifications, this smoother can:
1. Look at the overall window composition (using all items for context)
2. Override individual low-confidence predictions
3. Apply majority voting within confidence-weighted windows

Sliding Window Approach:
- Accumulate items until we have window_size (default 7) items
- Once full, use ALL items for context analysis
- Confirm/pop ONLY the oldest item (index 0)
- Keep remaining items in buffer for next iteration
- This gives each confirmed item full bidirectional context
"""

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Optional

from src.utils.AppLogging import logger


@dataclass
class ClassificationRecord:
    """Record of a single classification event."""
    track_id: int
    class_name: str
    confidence: float
    vote_ratio: float
    timestamp: float
    window_position: Optional[int] = None  # Position when confirmed
    smoothed: bool = False
    original_class: Optional[str] = None
    non_rejected_rois: int = 0  # Number of non-rejected ROIs (trustworthiness indicator)

class BidirectionalSmoother:
    """
    Sliding window classification smoother.

    Key features:
    1. Accumulates classifications into a sliding window
    2. Identifies low-confidence predictions using full window context
    3. Applies bidirectional smoothing based on surrounding items
    4. Confirms oldest item when window is full, keeps rest for context

    Sliding Window Logic:
    - Window size is 7 by default (3 items before + target + 3 after)
    - When window is full, analyze ALL 7 items for context
    - Confirm ONLY the oldest item (index 0) - it now has full context
    - Keep remaining 6 items in buffer
    - When new item arrives (7 again), confirm next oldest

    Smoothing rules:
    - If the oldest item has low confidence and differs from dominant class,
      it may be overridden based on window context
    - Confidence threshold determines what counts as "low"
    - Vote ratio threshold determines window dominance
    """

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        vote_ratio_threshold: float = 0.6,
        window_size: int = 21,
        window_timeout_seconds: float = 30.0,
        min_window_dominance: float = 0.7,
        min_trusted_rois: int = 3,
        warmup_smoothing_enabled: bool = True
    ):
        """
        Initialize smoother.

        Args:
            confidence_threshold: Below this, classification is considered uncertain
            vote_ratio_threshold: Below this, evidence ratio is considered weak
            window_size: Sliding window size (default 21 = 10 before + center + 10 after, must be odd)
            window_timeout_seconds: Force flush window after this duration of inactivity
            min_window_dominance: Min ratio for a class to be considered dominant
            min_trusted_rois: Min non-rejected ROIs needed to trust classification (default 3)
            warmup_smoothing_enabled: Enable in-place smoothing during warmup phase
        """
        # Validate window size
        if window_size < 3:
            raise ValueError("window_size must be >= 3")
        if window_size % 2 == 0:
            raise ValueError("window_size must be odd for center-based analysis")
        
        self.confidence_threshold = confidence_threshold
        self.vote_ratio_threshold = vote_ratio_threshold
        self.window_size = window_size
        self.window_timeout_seconds = window_timeout_seconds
        self.min_window_dominance = min_window_dominance
        self.min_trusted_rois = min_trusted_rois
        self.warmup_smoothing_enabled = warmup_smoothing_enabled

        # Sliding window buffer
        self.window_buffer: List[ClassificationRecord] = []
        self._last_activity_time: float = time.time()

        # Confirmed records history
        self.confirmed_records: List[ClassificationRecord] = []
        self._confirm_count = 0

        # Smoothing statistics
        self.total_records = 0
        self.smoothed_records = 0

        logger.info(
            f"[BidirectionalSmoother] Initialized with confidence_threshold={confidence_threshold}, "
            f"window_size={window_size} (center-based analysis mode), "
            f"warmup_smoothing_enabled={warmup_smoothing_enabled}"
        )

    @property
    def center_index(self) -> int:
        """Get the center index of the window (e.g., 21 // 2 = 10)."""
        return self.window_size // 2

    def _is_low_confidence(self, record: ClassificationRecord) -> bool:
        """
        Check if a record has low confidence.

        A record is considered low confidence if:
        1. It's 'Rejected' (always low confidence)
        2. Confidence or vote_ratio is below threshold

        Note: non_rejected_rois is checked separately during add_classification()
        to determine if we should even trust the classification enough to add it.
        """
        # 'Rejected' is always low confidence
        if record.class_name == 'Rejected':
            return True

        # Standard confidence checks (this is what batch smoothing is for!)
        return (
            record.confidence < self.confidence_threshold or
            record.vote_ratio < self.vote_ratio_threshold
        )

    def _get_window_distribution(self) -> Dict[str, int]:
        """Get count of each class in current window."""
        dist = defaultdict(int)
        for rec in self.window_buffer:
            dist[rec.class_name] += 1
        return dict(dist)

    def _get_confidence_weighted_distribution(self) -> Dict[str, float]:
        """Get confidence-weighted class distribution in current window."""
        dist = defaultdict(float)
        for rec in self.window_buffer:
            dist[rec.class_name] += rec.confidence
        return dict(dist)

    def _get_dominant_class(self, records: List[ClassificationRecord]) -> Optional[str]:
        """
        Get dominant class from a subset of records (exclude Rejected).
        
        Args:
            records: List of classification records to analyze
            
        Returns:
            Class with highest confidence weight, or None if empty or all Rejected
        """
        if not records:
            return None
        
        # Calculate confidence-weighted distribution (exclude Rejected)
        dist = defaultdict(float)
        for rec in records:
            if rec.class_name != 'Rejected':
                dist[rec.class_name] += rec.confidence
        
        if not dist:
            return None
        
        return max(dist, key=dist.get)

    def _create_smoothed_record(
        self,
        original: ClassificationRecord,
        new_class: str,
        reason: str
    ) -> ClassificationRecord:
        """
        Create a smoothed record from an original record.
        
        Args:
            original: Original classification record
            new_class: New class name to assign
            reason: Reason for smoothing (for logging)
            
        Returns:
            New smoothed ClassificationRecord
        """
        smoothed = ClassificationRecord(
            track_id=original.track_id,
            class_name=new_class,
            confidence=original.confidence,
            vote_ratio=original.vote_ratio,
            timestamp=original.timestamp,
            window_position=self._confirm_count,
            smoothed=True,
            original_class=original.class_name,
            non_rejected_rois=original.non_rejected_rois
        )
        self.smoothed_records += 1
        logger.info(
            f"[SMOOTHING] T{original.track_id} SMOOTHED | "
            f"{original.class_name}->{new_class} conf={original.confidence:.3f} reason={reason}"
        )
        return smoothed

    def _analyze_batch_context(self) -> Dict:
        """
        Analyze the batch context of the current window centered on center_index.
        
        Returns:
            Dict with keys:
                - dominant_class: Most common class (exclude Rejected)
                - dominance_ratio: Ratio of dominant class (0.0-1.0)
                - in_transition: Whether past/future have different dominants
                - past_dominant: Dominant in past half (0 to center_index-1)
                - future_dominant: Dominant in future half (center_index+1 to end)
        """
        if not self.window_buffer:
            return {
                'dominant_class': None,
                'dominance_ratio': 0.0,
                'in_transition': False,
                'past_dominant': None,
                'future_dominant': None
            }
        
        # Get confidence-weighted distribution (exclude Rejected)
        weighted_dist = self._get_confidence_weighted_distribution()
        weighted_dist_no_rejected = {k: v for k, v in weighted_dist.items() if k != 'Rejected'}
        
        # Calculate dominant class and ratio
        if weighted_dist_no_rejected:
            total_weight = sum(weighted_dist_no_rejected.values())
            dominant_class = max(weighted_dist_no_rejected, key=weighted_dist_no_rejected.get)
            dominance_ratio = weighted_dist_no_rejected[dominant_class] / total_weight if total_weight > 0 else 0.0
        else:
            # All items are Rejected - fallback
            dominant_class = None
            dominance_ratio = 0.0
            logger.info("[SMOOTHING] all_rejected_window | using_fallback=Unknown")
        
        # Split window into past and future halves
        center = self.center_index
        past_records = self.window_buffer[:center] if center > 0 else []
        future_records = self.window_buffer[center+1:] if center+1 < len(self.window_buffer) else []
        
        # Get dominant class for each half (minimum 2 items for meaningful analysis)
        past_dominant = self._get_dominant_class(past_records) if len(past_records) >= 2 else None
        future_dominant = self._get_dominant_class(future_records) if len(future_records) >= 2 else None
        
        # Detect transition (both halves must be valid and different)
        in_transition = (
            past_dominant is not None and 
            future_dominant is not None and 
            past_dominant != future_dominant
        )
        
        logger.info(
            f"[SMOOTHING] CONTEXT_ANALYSIS | dominant={dominant_class} "
            f"dominance={dominance_ratio:.2f} in_transition={in_transition} "
            f"past_dominant={past_dominant} future_dominant={future_dominant}"
        )
        
        return {
            'dominant_class': dominant_class,
            'dominance_ratio': dominance_ratio,
            'in_transition': in_transition,
            'past_dominant': past_dominant,
            'future_dominant': future_dominant
        }

    def _warmup_smooth_all(self):
        """
        Apply in-place smoothing to the warmup buffer.
        
        During warmup, clean up Rejected/outlier classifications IN-PLACE
        to prevent bad context from polluting the center analysis.
        
        This does NOT confirm items, just updates them in buffer.
        """
        if not self.window_buffer:
            return
        
        # Get confidence-weighted distribution (exclude Rejected)
        weighted_dist = self._get_confidence_weighted_distribution()
        weighted_dist_no_rejected = {k: v for k, v in weighted_dist.items() if k != 'Rejected'}
        
        # If all items are Rejected, use Unknown as fallback
        if not weighted_dist_no_rejected:
            # Smooth all Rejected to Unknown
            for i, record in enumerate(self.window_buffer):
                if record.class_name == 'Rejected':
                    logger.info(
                        f"[SMOOTHING] WARMUP_SMOOTH | T{record.track_id} "
                        f"Rejected->Unknown buffer_pos={i} reason=all_rejected_fallback"
                    )
                    # Update in-place
                    self.window_buffer[i] = ClassificationRecord(
                        track_id=record.track_id,
                        class_name='Unknown',
                        confidence=record.confidence,
                        vote_ratio=record.vote_ratio,
                        timestamp=record.timestamp,
                        window_position=None,  # Not confirmed yet
                        smoothed=True,
                        original_class='Rejected',
                        non_rejected_rois=record.non_rejected_rois
                    )
            return
        
        # Calculate dominant class and outlier threshold
        total_weight = sum(weighted_dist_no_rejected.values())
        dominant_class = max(weighted_dist_no_rejected, key=weighted_dist_no_rejected.get)
        dominance_ratio = weighted_dist_no_rejected[dominant_class] / total_weight if total_weight > 0 else 0.0
        
        # Get class counts for outlier detection
        class_dist = self._get_window_distribution()
        outlier_threshold = max(1, int(len(self.window_buffer) * 0.2))
        
        # Smooth items in-place
        for i, record in enumerate(self.window_buffer):
            original_class = record.class_name
            should_smooth = False
            reason = ""
            
            # Rule 1: Always smooth Rejected
            if record.class_name == 'Rejected':
                should_smooth = True
                reason = 'rejected_to_dominant'
            # Rule 2: Smooth outliers if dominant class is strong
            elif (dominance_ratio >= self.min_window_dominance and 
                  record.class_name != dominant_class):
                class_count = class_dist.get(record.class_name, 0)
                if class_count <= outlier_threshold:
                    should_smooth = True
                    reason = 'outlier_in_warmup'
            
            if should_smooth:
                logger.info(
                    f"[SMOOTHING] WARMUP_SMOOTH | T{record.track_id} "
                    f"{original_class}->{dominant_class} buffer_pos={i} reason={reason}"
                )
                # Update in-place
                self.window_buffer[i] = ClassificationRecord(
                    track_id=record.track_id,
                    class_name=dominant_class,
                    confidence=record.confidence,
                    vote_ratio=record.vote_ratio,
                    timestamp=record.timestamp,
                    window_position=None,  # Not confirmed yet
                    smoothed=True,
                    original_class=original_class,
                    non_rejected_rois=record.non_rejected_rois
                )

    def _smooth_oldest_with_context(
        self,
        oldest_record: ClassificationRecord,
        batch_context: Dict
    ) -> ClassificationRecord:
        """
        Apply smoothing to the oldest record using batch context from center analysis.
        
        Args:
            oldest_record: The oldest record in the window
            batch_context: Context dict from _analyze_batch_context()
            
        Returns:
            Smoothed or original record
        """
        dominant_class = batch_context['dominant_class']
        dominance_ratio = batch_context['dominance_ratio']
        in_transition = batch_context['in_transition']
        past_dominant = batch_context['past_dominant']
        
        # Rule 1: Always Smooth Rejected
        if oldest_record.class_name == 'Rejected':
            if dominance_ratio >= self.min_window_dominance and dominant_class is not None:
                target_class = dominant_class
                reason = 'rejected_to_dominant'
            else:
                target_class = 'Unknown'
                reason = 'rejected_to_unknown'
            return self._create_smoothed_record(oldest_record, target_class, reason)
        
        # Rule 2: Handle Batch Transitions
        if in_transition:
            # Oldest is in "past" context - check if it matches past batch
            if oldest_record.class_name == past_dominant:
                # Valid transition item - don't smooth
                logger.info(
                    f"[SMOOTHING] T{oldest_record.track_id} | "
                    f"action=NO_SMOOTHING reason=valid_transition_item"
                )
                oldest_record.window_position = self._confirm_count
                return oldest_record
            else:
                # Outlier during transition - smooth to past dominant
                if past_dominant is not None:
                    return self._create_smoothed_record(
                        oldest_record, past_dominant, 'transition_outlier'
                    )
        
        # Rule 3: Check Dominance
        if dominance_ratio < self.min_window_dominance or dominant_class is None:
            # No clear dominant - don't smooth
            logger.info(
                f"[SMOOTHING] T{oldest_record.track_id} | "
                f"action=NO_SMOOTHING reason=no_clear_dominant"
            )
            oldest_record.window_position = self._confirm_count
            return oldest_record
        
        # Rule 4: Smooth Outliers in Stable Batch
        if oldest_record.class_name != dominant_class:
            class_dist = self._get_window_distribution()
            class_count = class_dist.get(oldest_record.class_name, 0)
            outlier_threshold = max(1, int(len(self.window_buffer) * 0.2))
            
            if class_count <= outlier_threshold:
                reason = (
                    "outlier_low_conf" if self._is_low_confidence(oldest_record)
                    else "outlier_batch_override"
                )
                return self._create_smoothed_record(oldest_record, dominant_class, reason)
        
        # No smoothing needed
        oldest_record.window_position = self._confirm_count
        return oldest_record

    def _confirm_oldest_using_center_analysis(self) -> ClassificationRecord:
        """
        Confirm the oldest item using center-based batch context analysis.
        
        This method:
        1. Analyzes the center position to understand batch context
        2. Uses that context to determine if oldest should be smoothed
        3. Confirms and pops the oldest item (position 0)
        
        Returns:
            The confirmed (potentially smoothed) record
        """
        if not self.window_buffer:
            raise ValueError("Cannot confirm from empty window")
        
        # Analyze batch context centered on center_index
        batch_context = self._analyze_batch_context()
        
        # Get oldest record
        oldest_record = self.window_buffer[0]
        
        # Apply smoothing using center context
        confirmed = self._smooth_oldest_with_context(oldest_record, batch_context)
        
        # Remove oldest from buffer
        self.window_buffer.pop(0)
        self._confirm_count += 1
        
        # Store in history
        self.confirmed_records.append(confirmed)
        
        # Limit history
        if len(self.confirmed_records) > 100:
            self.confirmed_records = self.confirmed_records[-100:]
        
        logger.info(
            f"[SMOOTHING] T{confirmed.track_id} CONFIRMED | "
            f"class={confirmed.class_name} smoothed={confirmed.smoothed} "
            f"remaining_in_window={len(self.window_buffer)}"
        )
        
        return confirmed

    def _apply_smoothing_to_oldest(self) -> ClassificationRecord:
        """
        Apply smoothing to the oldest item in the window using full window context.

        Special handling for 'Rejected' class:
        - 'Rejected' is ALWAYS overridden (we never want to miss counting a bag)
        - If there's a dominant class in window, override to that class
        - If no dominant class, override to 'Unknown' (still counts as a bag)

        Returns:
            The (potentially smoothed) oldest record
        """
        if not self.window_buffer:
            raise ValueError("Cannot apply smoothing to empty window")

        oldest_record = self.window_buffer[0]

        # Need at least 2 items for meaningful smoothing
        if len(self.window_buffer) < 2:
            # Special case: even with small window, convert Rejected to Unknown
            if oldest_record.class_name == 'Rejected':
                smoothed = ClassificationRecord(
                    track_id=oldest_record.track_id,
                    class_name='Unknown',
                    confidence=oldest_record.confidence,
                    vote_ratio=oldest_record.vote_ratio,
                    timestamp=oldest_record.timestamp,
                    window_position=self._confirm_count,
                    smoothed=True,
                    original_class='Rejected',
                    non_rejected_rois=oldest_record.non_rejected_rois
                )
                self.smoothed_records += 1
                logger.info(
                    f"[SMOOTHING] T{oldest_record.track_id} SMOOTHED | "
                    f"Rejected->Unknown reason=small_window_fallback"
                )
                return smoothed

            logger.info(
                f"[SMOOTHING] T{oldest_record.track_id} | action=NO_SMOOTHING "
                f"reason=window_too_small size={len(self.window_buffer)}"
            )
            return oldest_record

        # Find dominant class using full window context (excluding 'Rejected' from consideration)
        weighted_dist = self._get_confidence_weighted_distribution()
        # Remove 'Rejected' from distribution for dominant class calculation
        weighted_dist_no_rejected = {k: v for k, v in weighted_dist.items() if k != 'Rejected'}
        total_weight = sum(weighted_dist_no_rejected.values())

        window_classes = [r.class_name for r in self.window_buffer]

        # Handle case where all items are Rejected
        if total_weight == 0:
            if oldest_record.class_name == 'Rejected':
                # All items are Rejected - convert to Unknown
                smoothed = ClassificationRecord(
                    track_id=oldest_record.track_id,
                    class_name='Unknown',
                    confidence=oldest_record.confidence,
                    vote_ratio=oldest_record.vote_ratio,
                    timestamp=oldest_record.timestamp,
                    window_position=self._confirm_count,
                    smoothed=True,
                    original_class='Rejected',
                    non_rejected_rois=oldest_record.non_rejected_rois
                )
                self.smoothed_records += 1
                logger.info(
                    f"[SMOOTHING] T{oldest_record.track_id} SMOOTHED | "
                    f"Rejected->Unknown reason=all_rejected_fallback"
                )
                return smoothed

            logger.info(
                f"[SMOOTHING] T{oldest_record.track_id} | action=NO_SMOOTHING "
                f"reason=zero_weight"
            )
            return oldest_record

        dominant_class = max(weighted_dist_no_rejected, key=weighted_dist_no_rejected.get)
        dominance_ratio = weighted_dist_no_rejected[dominant_class] / total_weight

        logger.info(
            f"[SMOOTHING] WINDOW_ANALYSIS | size={len(self.window_buffer)} "
            f"classes={window_classes} dominant={dominant_class} "
            f"dominance={dominance_ratio:.2f}"
        )

        # Special handling for 'Rejected' - ALWAYS override
        if oldest_record.class_name == 'Rejected':
            if dominance_ratio >= self.min_window_dominance:
                # Override to dominant class
                target_class = dominant_class
                reason = 'rejected_to_dominant'
            else:
                # No clear dominant class - use Unknown so we still count it
                target_class = 'Unknown'
                reason = 'rejected_to_unknown_no_dominant'

            smoothed = ClassificationRecord(
                track_id=oldest_record.track_id,
                class_name=target_class,
                confidence=oldest_record.confidence,
                vote_ratio=oldest_record.vote_ratio,
                timestamp=oldest_record.timestamp,
                window_position=self._confirm_count,
                smoothed=True,
                original_class='Rejected',
                non_rejected_rois=oldest_record.non_rejected_rois
            )
            self.smoothed_records += 1
            logger.info(
                f"[SMOOTHING] T{oldest_record.track_id} SMOOTHED | "
                f"Rejected->{target_class} dominance={dominance_ratio:.2f} "
                f"reason={reason}"
            )
            return smoothed

        # For non-Rejected classes: apply standard smoothing logic
        # Check if there's a clearly dominant class
        if dominance_ratio < self.min_window_dominance:
            logger.info(
                f"[SMOOTHING] T{oldest_record.track_id} | action=NO_SMOOTHING "
                f"reason=no_dominant_class best={dominant_class} ratio={dominance_ratio:.2f}"
            )
            oldest_record.window_position = self._confirm_count
            return oldest_record

        # Check if oldest record should be smoothed (outlier in dominant batch)
        # On a conveyor, bags come in monotype batches. A small minority of
        # different-class items against strong dominance is almost certainly
        # misclassification, even with high confidence.
        # Examples:
        #   window=7:  [B,B,B,X(1.00),B,B,B] → X smoothed (1 outlier)
        #   window=15: [B,B,B,B,B,B,B,B,B,X,B,X,B,B,B] → X smoothed (2 outliers)
        if oldest_record.class_name != dominant_class:
            class_dist = self._get_window_distribution()
            class_count = class_dist.get(oldest_record.class_name, 0)
            # Allow smoothing when the non-dominant class is a small minority
            # (up to ~20% of window size, minimum 1).  This handles both small
            # windows (size 7 → threshold 1) and larger ones (size 15 → threshold 3).
            outlier_threshold = max(1, int(len(self.window_buffer) * 0.2))

            if class_count <= outlier_threshold:
                reason = (
                    "outlier_low_conf" if self._is_low_confidence(oldest_record)
                    else "outlier_batch_override"
                )
                smoothed = ClassificationRecord(
                    track_id=oldest_record.track_id,
                    class_name=dominant_class,
                    confidence=oldest_record.confidence,
                    vote_ratio=oldest_record.vote_ratio,
                    timestamp=oldest_record.timestamp,
                    window_position=self._confirm_count,
                    smoothed=True,
                    original_class=oldest_record.class_name,
                    non_rejected_rois=oldest_record.non_rejected_rois
                )
                self.smoothed_records += 1
                logger.info(
                    f"[SMOOTHING] T{oldest_record.track_id} SMOOTHED | "
                    f"{oldest_record.class_name}->{dominant_class} "
                    f"conf={oldest_record.confidence:.3f} dominance={dominance_ratio:.2f} "
                    f"class_count={class_count}/{len(self.window_buffer)} "
                    f"threshold={outlier_threshold} reason={reason}"
                )
                return smoothed

        # No smoothing needed
        oldest_record.window_position = self._confirm_count
        return oldest_record

    def _confirm_oldest(self) -> ClassificationRecord:
        """
        Confirm and remove the oldest item from the window.

        Uses full window context for smoothing decision, but only confirms/pops
        the oldest item.

        Returns:
            The confirmed (potentially smoothed) record
        """
        # Apply smoothing using full window context
        confirmed = self._apply_smoothing_to_oldest()

        # Remove oldest from buffer
        self.window_buffer.pop(0)
        self._confirm_count += 1

        # Store in history
        self.confirmed_records.append(confirmed)

        # Limit history
        if len(self.confirmed_records) > 100:
            self.confirmed_records = self.confirmed_records[-100:]

        logger.info(
            f"[SMOOTHING] T{confirmed.track_id} CONFIRMED | "
            f"class={confirmed.class_name} smoothed={confirmed.smoothed} "
            f"remaining_in_window={len(self.window_buffer)}"
        )

        return confirmed

    def add_classification(
        self,
        track_id: int,
        class_name: str,
        confidence: float,
        vote_ratio: float,
        non_rejected_rois: int = 0
    ) -> Optional[ClassificationRecord]:
        """
        Add a classification to the sliding window.

        Two-phase processing:
        - Phase 1 (Warmup): Accumulate items until window is full (items 1-20 for window_size=21)
        - Phase 2 (Steady State): Confirm oldest using center analysis (items 21+)

        Note: 'Rejected' class is now INCLUDED in the window.

        Items with < min_trusted_rois are already converted to 'Rejected' by the caller
        before reaching this method. The smoother then applies confidence-based smoothing
        to override low-confidence items (including 'Rejected') based on window context.

        Args:
            track_id: Track identifier
            class_name: Predicted class
            confidence: Classification confidence
            vote_ratio: Vote ratio from evidence accumulation
            non_rejected_rois: Number of non-rejected ROIs (for logging/debugging)

        Returns:
            A confirmed record if window was full, None otherwise
        """
        # Create record
        record = ClassificationRecord(
            track_id=track_id,
            class_name=class_name,
            confidence=confidence,
            vote_ratio=vote_ratio,
            timestamp=time.time(),
            non_rejected_rois=non_rejected_rois
        )

        # Add to sliding window
        self.window_buffer.append(record)
        self._last_activity_time = time.time()
        self.total_records += 1

        # Phase 1: Warmup - accumulate until window is full
        if len(self.window_buffer) < self.window_size:
            # Optional warmup smoothing to clean bad context
            if self.warmup_smoothing_enabled:
                self._warmup_smooth_all()
            
            logger.debug(
                f"[SMOOTHING] WARMUP | buffer={len(self.window_buffer)}/{self.window_size} "
                f"status=accumulating"
            )
            return None
        
        # Phase 2: Steady state - confirm oldest using center analysis
        logger.info(
            f"[SMOOTHING] STEADY_STATE | buffer={len(self.window_buffer)} "
            f"confirming_oldest_with_center_analysis"
        )
        return self._confirm_oldest_using_center_analysis()

    def check_timeout(self) -> List[ClassificationRecord]:
        """
        Check for timeout and flush remaining items if needed.

        Call this periodically to handle cases where no new items arrive
        but we have pending items in the window.

        Returns:
            List of confirmed records if timeout occurred, empty list otherwise
        """
        if not self.window_buffer:
            return []

        elapsed = time.time() - self._last_activity_time
        if elapsed < self.window_timeout_seconds:
            return []

        logger.info(
            f"[SMOOTHING] TIMEOUT | elapsed={elapsed:.1f}s "
            f"pending={len(self.window_buffer)}"
        )

        return self.flush_remaining()

    def flush_remaining(self) -> List[ClassificationRecord]:
        """
        Flush all remaining items in the window with partial context.

        Called on timeout or cleanup. Confirms items one by one,
        each using whatever context remains in the window.

        Returns:
            List of all confirmed records
        """
        confirmed = []

        while self.window_buffer:
            # For partial context (buffer < window_size), still use center analysis
            # but log that it's using partial context
            if len(self.window_buffer) < self.window_size:
                logger.info(
                    f"[SMOOTHING] FLUSH | buffer={len(self.window_buffer)} context=partial"
                )
            
            # Use center analysis even with partial window
            if self.window_buffer:
                batch_context = self._analyze_batch_context()
                oldest_record = self.window_buffer[0]
                record = self._smooth_oldest_with_context(oldest_record, batch_context)
                
                # Remove oldest from buffer
                self.window_buffer.pop(0)
                self._confirm_count += 1
                
                # Store in history
                self.confirmed_records.append(record)
                
                # Limit history
                if len(self.confirmed_records) > 100:
                    self.confirmed_records = self.confirmed_records[-100:]
                
                logger.info(
                    f"[SMOOTHING] T{record.track_id} CONFIRMED | "
                    f"class={record.class_name} smoothed={record.smoothed} "
                    f"remaining_in_window={len(self.window_buffer)}"
                )
                
                confirmed.append(record)

        if confirmed:
            logger.info(
                f"[SMOOTHING] FLUSHED | confirmed={len(confirmed)} records"
            )

        return confirmed

    def finalize_batch(self) -> List[ClassificationRecord]:
        """
        Alias for flush_remaining() for backward compatibility.

        Returns:
            List of all confirmed records
        """
        return self.flush_remaining()

    def get_pending_records(self) -> List[ClassificationRecord]:
        """Get records in current sliding window (not yet confirmed)."""
        return self.window_buffer.copy()

    def get_statistics(self) -> Dict[str, any]:
        """Get smoothing statistics."""
        return {
            'total_records': self.total_records,
            'smoothed_records': self.smoothed_records,
            'smoothing_rate': (
                self.smoothed_records / self.total_records
                if self.total_records > 0 else 0.0
            ),
            'confirmed_count': self._confirm_count,
            'pending_in_window': len(self.window_buffer)
        }

    def get_dominant_class(self) -> Optional[str]:
        """Get the dominant (most frequent) class in the current window, excluding Rejected."""
        if not self.window_buffer:
            return None
        dist: Dict[str, int] = {}
        for rec in self.window_buffer:
            if rec.class_name != 'Rejected':
                dist[rec.class_name] = dist.get(rec.class_name, 0) + 1
        if not dist:
            return None
        return max(dist, key=dist.get)

    def get_pending_summary(self) -> Dict[str, int]:
        """Get counts of pending items in the smoothing window grouped by class."""
        summary: Dict[str, int] = {}
        for record in self.window_buffer:
            summary[record.class_name] = summary.get(record.class_name, 0) + 1
        return summary

    def cleanup(self):
        """Clean up resources."""
        if self.window_buffer:
            # Flush remaining items
            self.flush_remaining()

        self.confirmed_records.clear()
        logger.info(
            f"[BidirectionalSmoother] Cleanup complete, "
            f"smoothed {self.smoothed_records}/{self.total_records} records"
        )
