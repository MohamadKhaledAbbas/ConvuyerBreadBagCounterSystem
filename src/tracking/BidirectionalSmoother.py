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
        window_size: int = 7,
        window_timeout_seconds: float = 30.0,
        min_window_dominance: float = 0.7,
        min_trusted_rois: int = 3
    ):
        """
        Initialize smoother.

        Args:
            confidence_threshold: Below this, classification is considered uncertain
            vote_ratio_threshold: Below this, evidence ratio is considered weak
            window_size: Sliding window size (default 7 = 3 before + target + 3 after)
            window_timeout_seconds: Force flush window after this duration of inactivity
            min_window_dominance: Min ratio for a class to be considered dominant
            min_trusted_rois: Min non-rejected ROIs needed to trust classification (default 3)
        """
        self.confidence_threshold = confidence_threshold
        self.vote_ratio_threshold = vote_ratio_threshold
        self.window_size = window_size
        self.window_timeout_seconds = window_timeout_seconds
        self.min_window_dominance = min_window_dominance
        self.min_trusted_rois = min_trusted_rois

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
            f"window_size={window_size} (sliding window mode)"
        )

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
        self.total_records += 1
        self._last_activity_time = time.time()

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

        logger.info(
            f"[SMOOTHING] T{track_id} ADDED_TO_WINDOW | "
            f"class={class_name} conf={confidence:.3f} rois={non_rejected_rois} "
            f"window_size={len(self.window_buffer)}/{self.window_size}"
        )

        # Check if window is full - confirm oldest item
        if len(self.window_buffer) >= self.window_size:
            return self._confirm_oldest()

        return None

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
        Flush all remaining items in the window.

        Called on timeout or cleanup. Confirms items one by one,
        each using whatever context remains in the window.

        Returns:
            List of all confirmed records
        """
        confirmed = []

        while self.window_buffer:
            record = self._confirm_oldest()
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
