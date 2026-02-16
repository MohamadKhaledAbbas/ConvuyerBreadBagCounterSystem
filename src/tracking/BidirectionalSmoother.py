"""
Forward-context sliding window smoother.

Uses forward-context smoothing for classification correction on conveyor belt
bread bag counting systems. Each item is confirmed using context from items
that arrive AFTER it, providing reliable smoothing without bidirectional complexity.

Smoothing Rules:
1. Rejected is ALWAYS smoothed (to forward dominant or Unknown)
2. Outliers in stable forward context are smoothed to dominant
3. Items at batch boundaries (matching recent confirmed batch) are preserved
4. No dominant in forward context → keep original classification

Forward Context Approach:
- Accumulate items until we have window_size (default 21) items
- When full, analyze items AFTER the oldest as forward context
- Smooth oldest if needed, then confirm and remove it
- Keep remaining items in buffer for next iteration
- Smart batch boundary handling using confirmed history
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
    Forward-context sliding window classification smoother.

    Key features:
    1. Accumulates classifications into a sliding window
    2. Uses forward context (items after oldest) for smoothing decisions
    3. Always smooths Rejected class to prevent missed counts
    4. Smart batch boundary handling using confirmed history
    5. Confirms oldest item when window is full, keeps rest for context

    Forward Context Logic:
    - Accumulate window_size items (default 21)
    - When full, analyze items 1..N-1 as forward context for item 0
    - Smooth item 0 if needed, then confirm and remove it
    - Keep remaining items; when new item arrives, repeat

    Smoothing rules:
    - Rejected is ALWAYS overridden (we never want to miss counting a bag)
    - If forward context has a dominant class, override outliers to that class
    - At batch boundaries, preserve items matching recent confirmed batch
    - If no dominant class, override Rejected to 'Unknown' (still counts)
    """

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        vote_ratio_threshold: float = 0.6,
        window_size: int = 21,
        window_timeout_seconds: float = 30.0,
        min_window_dominance: float = 0.7,
        min_trusted_rois: int = 3,
        warmup_smoothing_enabled: bool = True  # Deprecated, kept for backward compatibility
    ):
        """
        Initialize smoother.

        Args:
            confidence_threshold: Below this, classification is considered uncertain
            vote_ratio_threshold: Below this, evidence ratio is considered weak
            window_size: Sliding window size (default 21, must be >= 3)
            window_timeout_seconds: Force flush window after this duration of inactivity
            min_window_dominance: Min ratio for a class to be considered dominant
            min_trusted_rois: Min non-rejected ROIs needed to trust classification (default 3)
            warmup_smoothing_enabled: Deprecated, kept for backward compatibility (ignored)
        """
        # Validate window size
        if window_size < 3:
            raise ValueError("window_size must be >= 3")
        
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
            f"window_size={window_size} (forward-context smoothing mode)"
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

    def _get_last_confirmed_non_rejected_class(self) -> Optional[str]:
        """Get the class of the last confirmed non-Rejected/non-Unknown record."""
        for record in reversed(self.confirmed_records):
            if record.class_name not in ('Rejected', 'Unknown'):
                return record.class_name
        return None

    def _is_at_batch_boundary(self, record: ClassificationRecord) -> bool:
        """
        Check if a record is at a batch boundary (last item of previous batch).

        Uses confirmed history: if recent confirmed items match the record's class,
        the record is likely the tail end of the same batch and should be preserved
        even if forward context has shifted to a new batch.

        Args:
            record: The record to check

        Returns:
            True if record appears to be at a batch boundary
        """
        if not self.confirmed_records:
            return False

        lookback = min(5, len(self.confirmed_records))
        recent = self.confirmed_records[-lookback:]
        recent_classes = [r.class_name for r in recent
                         if r.class_name not in ('Rejected', 'Unknown')]

        if not recent_classes:
            return False

        match_count = sum(1 for c in recent_classes if c == record.class_name)
        return match_count >= len(recent_classes) * 0.5

    def _smooth_oldest_forward(self) -> ClassificationRecord:
        """
        Apply smoothing to the oldest record using forward context.

        Forward context = all items after the oldest in the window buffer.

        Rules (in priority order):
        1. Rejected is ALWAYS smoothed (to forward dominant, confirmed history, or Unknown)
        2. If no meaningful forward context → keep original (unless Rejected)
        3. If no clear dominance in forward context → keep original
        4. If oldest matches forward dominant → keep (no smoothing needed)
        5. If at batch boundary (matches recent confirmed batch) → preserve
        6. If outlier in strong forward context → smooth to dominant
        7. Otherwise → keep original

        Returns:
            The (potentially smoothed) oldest record
        """
        if not self.window_buffer:
            raise ValueError("Cannot smooth from empty window")

        oldest = self.window_buffer[0]
        forward = self.window_buffer[1:]

        # Get forward context distribution (exclude Rejected)
        forward_non_rejected = [r for r in forward if r.class_name != 'Rejected']

        # Calculate forward dominant class and dominance ratio
        if forward_non_rejected:
            weighted_dist = defaultdict(float)
            for r in forward_non_rejected:
                weighted_dist[r.class_name] += r.confidence
            total_weight = sum(weighted_dist.values())
            forward_dominant = max(weighted_dist, key=weighted_dist.get) if weighted_dist else None
            dominance_ratio = (
                weighted_dist.get(forward_dominant, 0) / total_weight
                if total_weight > 0 else 0.0
            )
        else:
            forward_dominant = None
            dominance_ratio = 0.0

        logger.info(
            f"[SMOOTHING] FORWARD_CONTEXT | oldest=T{oldest.track_id}:{oldest.class_name} "
            f"forward_size={len(forward)} forward_dominant={forward_dominant} "
            f"dominance={dominance_ratio:.2f}"
        )

        # RULE 1: Always smooth Rejected
        if oldest.class_name == 'Rejected':
            if forward_dominant and dominance_ratio >= self.min_window_dominance:
                return self._create_smoothed_record(
                    oldest, forward_dominant, 'rejected_to_forward_dominant'
                )
            elif forward_dominant:
                # Some forward context but weak dominance - use best guess
                return self._create_smoothed_record(
                    oldest, forward_dominant, 'rejected_to_best_guess'
                )
            else:
                # No forward context or all Rejected - use confirmed history or Unknown
                fallback = self._get_last_confirmed_non_rejected_class()
                target = fallback if fallback else 'Unknown'
                reason = 'rejected_to_confirmed_history' if fallback else 'rejected_to_unknown'
                return self._create_smoothed_record(oldest, target, reason)

        # RULE 2: No meaningful forward context → keep original
        if not forward_non_rejected or not forward_dominant:
            logger.info(
                f"[SMOOTHING] T{oldest.track_id} | action=NO_SMOOTHING "
                f"reason=no_forward_context"
            )
            oldest.window_position = self._confirm_count
            return oldest

        # RULE 3: No clear dominance in forward context → keep original
        if dominance_ratio < self.min_window_dominance:
            logger.info(
                f"[SMOOTHING] T{oldest.track_id} | action=NO_SMOOTHING "
                f"reason=no_forward_dominant ratio={dominance_ratio:.2f}"
            )
            oldest.window_position = self._confirm_count
            return oldest

        # RULE 4: Oldest matches forward dominant → no smoothing needed
        if oldest.class_name == forward_dominant:
            oldest.window_position = self._confirm_count
            return oldest

        # RULE 5: Batch boundary check - preserve end-of-batch items
        if self._is_at_batch_boundary(oldest):
            logger.info(
                f"[SMOOTHING] T{oldest.track_id} | action=NO_SMOOTHING "
                f"reason=batch_boundary class={oldest.class_name}"
            )
            oldest.window_position = self._confirm_count
            return oldest

        # RULE 6: Outlier detection - smooth if record is rare in forward context
        forward_class_count = sum(1 for r in forward if r.class_name == oldest.class_name)
        outlier_threshold = max(1, int(len(forward) * 0.2))

        if forward_class_count <= outlier_threshold:
            reason = (
                "outlier_low_conf" if self._is_low_confidence(oldest)
                else "outlier_forward_smooth"
            )
            return self._create_smoothed_record(oldest, forward_dominant, reason)

        # RULE 7: Not an outlier - significant presence in forward context, keep original
        logger.info(
            f"[SMOOTHING] T{oldest.track_id} | action=NO_SMOOTHING "
            f"reason=significant_forward_presence "
            f"class_count={forward_class_count}/{len(forward)}"
        )
        oldest.window_position = self._confirm_count
        return oldest

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

        Processing:
        - Phase 1 (Accumulation): Collect items until window is full
        - Phase 2 (Steady State): Confirm oldest using forward-context smoothing

        Note: 'Rejected' class is included in the window and will always be
        smoothed when confirmed.

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

        # Phase 1: Accumulate until window is full
        if len(self.window_buffer) < self.window_size:
            logger.debug(
                f"[SMOOTHING] ACCUMULATING | buffer={len(self.window_buffer)}/{self.window_size} "
                f"status=waiting_for_context"
            )
            return None

        # Phase 2: Steady state - confirm oldest using forward context
        return self._confirm_oldest_forward()

    def _confirm_oldest_forward(self) -> ClassificationRecord:
        """
        Confirm the oldest item using forward-context smoothing.

        This method:
        1. Applies forward-context smoothing to the oldest item
        2. Removes the oldest item from the buffer
        3. Stores the confirmed record in history

        Returns:
            The confirmed (potentially smoothed) record
        """
        if not self.window_buffer:
            raise ValueError("Cannot confirm from empty window")

        # Apply forward-context smoothing
        confirmed = self._smooth_oldest_forward()

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
        Flush all remaining items in the window with partial forward context.

        Called on timeout or cleanup. Confirms items one by one using
        forward-context smoothing with whatever context remains.

        Returns:
            List of all confirmed records
        """
        confirmed = []

        while self.window_buffer:
            if len(self.window_buffer) < self.window_size:
                logger.info(
                    f"[SMOOTHING] FLUSH | buffer={len(self.window_buffer)} context=partial"
                )

            # Use forward-context smoothing even with partial window
            record = self._smooth_oldest_forward()

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
