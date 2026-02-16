"""
Three-phase deferred confirmation smoother.

This implementation eliminates the warmup smoothing feedback loop by:
1. Accumulating raw classifications without smoothing during warmup
2. Deferring confirmation of the first center_index items (they had no future context)
3. Releasing deferred items once batch identity is established via batch lock

Key Design Rules:
- No warmup smoothing (raw data preserved during accumulation)
- Always use original_class for distribution calculations
- Exclude both Rejected AND Unknown from dominance
- Override ALL outliers once batch is locked
- Conservative transition detection (≥70% dominance in EACH half)
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
    Three-phase deferred confirmation smoother.
    
    Phase 1 — Accumulate (items 1 to window_size-1):
        Store raw classifications with NO smoothing and NO confirmation
        
    Phase 2 — Steady State (items window_size onward):
        Window is full. Center analysis confirms the oldest item.
        BUT: the first center_index items are DEFERRED (held in _deferred_records)
        because they were the "past" half with no future context when they entered.
        
    Phase 3 — Batch Lock (after batch_lock_count steady-state confirmations):
        By now we've seen enough items to KNOW what the batch is with certainty.
        Release all deferred items using the established batch identity.
        ALL outliers are overridden to match the batch class.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        vote_ratio_threshold: float = 0.6,
        window_size: int = 21,
        window_timeout_seconds: float = 30.0,
        min_window_dominance: float = 0.7,
        min_trusted_rois: int = 3,
        batch_lock_count: int = 10
    ):
        """
        Initialize three-phase smoother.

        Args:
            confidence_threshold: Below this, classification is considered uncertain
            vote_ratio_threshold: Below this, evidence ratio is considered weak
            window_size: Sliding window size (must be odd, default 21)
            window_timeout_seconds: Force flush window after this duration of inactivity
            min_window_dominance: Min ratio for a class to be considered dominant
            min_trusted_rois: Min non-rejected ROIs needed to trust classification
            batch_lock_count: Number of steady-state confirmations before batch lock (default 10)
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
        self.batch_lock_count = batch_lock_count

        # Sliding window buffer
        self.window_buffer: List[ClassificationRecord] = []
        self._last_activity_time: float = time.time()

        # Deferred records (first center_index items that had no future context)
        self._deferred_records: List[ClassificationRecord] = []
        
        # Steady state tracking
        self._steady_state_confirms: int = 0
        self._batch_locked: bool = False

        # Confirmed records history
        self.confirmed_records: List[ClassificationRecord] = []
        self._confirm_count = 0

        # Smoothing statistics
        self.total_records = 0
        self.smoothed_records = 0

        logger.info(
            f"[BidirectionalSmoother] Initialized with confidence_threshold={confidence_threshold}, "
            f"window_size={window_size}, batch_lock_count={batch_lock_count} "
            f"(THREE-PHASE DEFERRED CONFIRMATION MODE)"
        )

    @property
    def center_index(self) -> int:
        """Get the center index of the window (e.g., 21 // 2 = 10)."""
        return self.window_size // 2

    def _get_original_class(self, record: ClassificationRecord) -> str:
        """Get the original class of a record (before any smoothing)."""
        return record.original_class if record.original_class is not None else record.class_name

    def _is_low_confidence(self, record: ClassificationRecord) -> bool:
        """
        Check if a record has low confidence.

        A record is considered low confidence if:
        1. It's 'Rejected' (always low confidence)
        2. Confidence or vote_ratio is below threshold
        """
        if record.class_name == 'Rejected':
            return True

        return (
            record.confidence < self.confidence_threshold or
            record.vote_ratio < self.vote_ratio_threshold
        )

    def _get_distribution_from_records(
        self, 
        records: List[ClassificationRecord], 
        use_original: bool = True,
        exclude_classes: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Get confidence-weighted class distribution from records.
        
        Args:
            records: List of classification records
            use_original: If True, use original_class; if False, use current class_name
            exclude_classes: Classes to exclude from distribution (e.g., ['Rejected', 'Unknown'])
            
        Returns:
            Dict mapping class names to confidence weights
        """
        if exclude_classes is None:
            exclude_classes = ['Rejected', 'Unknown']
            
        dist = defaultdict(float)
        for rec in records:
            class_name = self._get_original_class(rec) if use_original else rec.class_name
            if class_name not in exclude_classes:
                dist[class_name] += rec.confidence
        return dict(dist)

    def _get_dominant_class_from_records(
        self, 
        records: List[ClassificationRecord],
        use_original: bool = True
    ) -> Optional[str]:
        """
        Get dominant class from records (exclude Rejected and Unknown).
        
        Args:
            records: List of classification records
            use_original: If True, use original_class; if False, use current class_name
            
        Returns:
            Class with highest confidence weight, or None if empty
        """
        if not records:
            return None
        
        dist = self._get_distribution_from_records(records, use_original=use_original)
        
        if not dist:
            return None
        
        return max(dist, key=dist.get)

    def _get_batch_identity(self) -> Dict:
        """
        Analyze confirmed records to determine batch identity.
        
        Returns:
            Dict with keys:
                - dominant_class: Most common class in confirmed records
                - dominance_ratio: Ratio of dominant class (0.0-1.0)
                - total_confirmed: Number of confirmed records analyzed
        """
        if not self.confirmed_records:
            return {
                'dominant_class': None,
                'dominance_ratio': 0.0,
                'total_confirmed': 0
            }
        
        # Use original_class for batch identity to avoid feedback loops
        dist = self._get_distribution_from_records(self.confirmed_records, use_original=True)
        
        if not dist:
            return {
                'dominant_class': None,
                'dominance_ratio': 0.0,
                'total_confirmed': len(self.confirmed_records)
            }
        
        total_weight = sum(dist.values())
        dominant_class = max(dist, key=dist.get)
        dominance_ratio = dist[dominant_class] / total_weight if total_weight > 0 else 0.0
        
        return {
            'dominant_class': dominant_class,
            'dominance_ratio': dominance_ratio,
            'total_confirmed': len(self.confirmed_records)
        }

    def _analyze_window_context(self) -> Dict:
        """
        Analyze the batch context of the current window centered on center_index.
        
        Returns:
            Dict with keys:
                - dominant_class: Most common class (exclude Rejected and Unknown)
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
        
        # Get distribution using original_class (exclude Rejected and Unknown)
        dist = self._get_distribution_from_records(self.window_buffer, use_original=True)
        
        # Calculate dominant class and ratio
        if dist:
            total_weight = sum(dist.values())
            dominant_class = max(dist, key=dist.get)
            dominance_ratio = dist[dominant_class] / total_weight if total_weight > 0 else 0.0
        else:
            dominant_class = None
            dominance_ratio = 0.0
            logger.info("[SMOOTHING] all_rejected_or_unknown_window | using_fallback=None")
        
        # Split window into past and future halves
        center = self.center_index
        past_records = self.window_buffer[:center] if center > 0 else []
        future_records = self.window_buffer[center+1:] if center+1 < len(self.window_buffer) else []
        
        # Get dominant class for each half (use original_class)
        past_dominant = None
        future_dominant = None
        
        if len(past_records) >= 2:
            past_dist = self._get_distribution_from_records(past_records, use_original=True)
            if past_dist:
                past_total = sum(past_dist.values())
                past_dominant = max(past_dist, key=past_dist.get)
                past_ratio = past_dist[past_dominant] / past_total if past_total > 0 else 0.0
                # Require ≥70% dominance for valid half
                if past_ratio < 0.7:
                    past_dominant = None
        
        if len(future_records) >= 2:
            future_dist = self._get_distribution_from_records(future_records, use_original=True)
            if future_dist:
                future_total = sum(future_dist.values())
                future_dominant = max(future_dist, key=future_dist.get)
                future_ratio = future_dist[future_dominant] / future_total if future_total > 0 else 0.0
                # Require ≥70% dominance for valid half
                if future_ratio < 0.7:
                    future_dominant = None
        
        # Conservative transition detection: both halves must be valid and different
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
            original_class=self._get_original_class(original),
            non_rejected_rois=original.non_rejected_rois
        )
        self.smoothed_records += 1
        logger.info(
            f"[SMOOTHING] T{original.track_id} SMOOTHED | "
            f"{self._get_original_class(original)}->{new_class} conf={original.confidence:.3f} reason={reason}"
        )
        return smoothed

    def _smooth_record_with_context(
        self,
        record: ClassificationRecord,
        context: Dict
    ) -> ClassificationRecord:
        """
        Apply smoothing to a record using window context.
        
        After batch lock: ALL outliers are overridden to dominant class when 
        dominance ≥ min_window_dominance, not just low-confidence ones.
        
        Args:
            record: The record to potentially smooth
            context: Context dict from _analyze_window_context()
            
        Returns:
            Smoothed or original record
        """
        original_class = self._get_original_class(record)
        dominant_class = context['dominant_class']
        dominance_ratio = context['dominance_ratio']
        in_transition = context['in_transition']
        past_dominant = context['past_dominant']
        
        # Rule 1: Always Smooth Rejected
        if original_class == 'Rejected':
            if dominance_ratio >= self.min_window_dominance and dominant_class is not None:
                target_class = dominant_class
                reason = 'rejected_to_dominant'
            else:
                target_class = 'Unknown'
                reason = 'rejected_to_unknown'
            return self._create_smoothed_record(record, target_class, reason)
        
        # Rule 2: Handle Batch Transitions (conservative)
        if in_transition:
            # Record is in "past" context - check if it matches past batch
            if original_class == past_dominant:
                # Valid transition item - don't smooth
                logger.info(
                    f"[SMOOTHING] T{record.track_id} | "
                    f"action=NO_SMOOTHING reason=valid_transition_item"
                )
                record.window_position = self._confirm_count
                return record
            else:
                # Outlier during transition - smooth to past dominant
                if past_dominant is not None:
                    return self._create_smoothed_record(
                        record, past_dominant, 'transition_outlier'
                    )
        
        # Rule 3: Check Dominance
        if dominance_ratio < self.min_window_dominance or dominant_class is None:
            # No clear dominant - don't smooth
            logger.info(
                f"[SMOOTHING] T{record.track_id} | "
                f"action=NO_SMOOTHING reason=no_clear_dominant"
            )
            record.window_position = self._confirm_count
            return record
        
        # Rule 4: Smooth Outliers
        # After batch lock: ALL outliers are smoothed when batch dominance is strong
        if original_class != dominant_class:
            if self._batch_locked:
                # Batch locked: override ALL outliers regardless of confidence
                return self._create_smoothed_record(
                    record, dominant_class, 'batch_locked_override'
                )
            else:
                # Before batch lock: only smooth low-confidence outliers
                if self._is_low_confidence(record):
                    return self._create_smoothed_record(
                        record, dominant_class, 'outlier_low_conf'
                    )
        
        # No smoothing needed
        record.window_position = self._confirm_count
        return record

    def _resolve_deferred_records(self) -> List[ClassificationRecord]:
        """
        Resolve all deferred records using confirmed records as batch identity evidence.
        
        ALL outliers (including high-confidence) are overridden to batch class when 
        batch dominance ≥ min_window_dominance.
        
        Returns:
            List of resolved (confirmed) deferred records
        """
        if not self._deferred_records:
            return []
        
        # Get batch identity from confirmed records
        batch_identity = self._get_batch_identity()
        dominant_class = batch_identity['dominant_class']
        dominance_ratio = batch_identity['dominance_ratio']
        
        logger.info(
            f"[SMOOTHING] RESOLVING_DEFERRED | count={len(self._deferred_records)} "
            f"batch_class={dominant_class} dominance={dominance_ratio:.2f}"
        )
        
        resolved = []
        for record in self._deferred_records:
            original_class = self._get_original_class(record)
            
            # Rule 1: Always smooth Rejected to batch class or Unknown
            if original_class == 'Rejected':
                if dominance_ratio >= self.min_window_dominance and dominant_class is not None:
                    target_class = dominant_class
                    reason = 'deferred_rejected_to_batch'
                else:
                    target_class = 'Unknown'
                    reason = 'deferred_rejected_to_unknown'
                resolved_record = self._create_smoothed_record(record, target_class, reason)
            
            # Rule 2: Override ALL outliers when batch dominance is strong
            elif (dominance_ratio >= self.min_window_dominance and 
                  dominant_class is not None and 
                  original_class != dominant_class):
                resolved_record = self._create_smoothed_record(
                    record, dominant_class, 'deferred_batch_override'
                )
            
            # Rule 3: Keep matching class as-is
            else:
                logger.info(
                    f"[SMOOTHING] T{record.track_id} | "
                    f"action=NO_SMOOTHING reason=deferred_matches_batch"
                )
                record.window_position = self._confirm_count
                resolved_record = record
            
            self._confirm_count += 1
            self.confirmed_records.append(resolved_record)
            resolved.append(resolved_record)
        
        # Clear deferred records
        self._deferred_records.clear()
        
        logger.info(
            f"[SMOOTHING] DEFERRED_RESOLVED | count={len(resolved)}"
        )
        
        return resolved

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
        context = self._analyze_window_context()
        
        # Get oldest record
        oldest_record = self.window_buffer[0]
        
        # Apply smoothing using context
        confirmed = self._smooth_record_with_context(oldest_record, context)
        
        # Remove oldest from buffer
        self.window_buffer.pop(0)
        self._confirm_count += 1
        self._steady_state_confirms += 1
        
        # Store in history
        self.confirmed_records.append(confirmed)
        
        # Limit history
        if len(self.confirmed_records) > 100:
            self.confirmed_records = self.confirmed_records[-100:]
        
        logger.info(
            f"[SMOOTHING] T{confirmed.track_id} CONFIRMED | "
            f"class={confirmed.class_name} smoothed={confirmed.smoothed} "
            f"remaining_in_window={len(self.window_buffer)} "
            f"steady_state_confirms={self._steady_state_confirms}"
        )
        
        return confirmed

    def add_classification(
        self,
        track_id: int,
        class_name: str,
        confidence: float,
        vote_ratio: float,
        non_rejected_rois: int = 0
    ) -> Optional[List[ClassificationRecord]]:
        """
        Add a classification to the three-phase smoother.

        Three-phase processing:
        - Phase 1 (Accumulate): Items 1 to window_size-1, no smoothing, no confirmation
        - Phase 2 (Steady State): Items window_size onward, confirm oldest (defer first center_index)
        - Phase 3 (Batch Lock): After batch_lock_count confirmations, release deferred records

        Args:
            track_id: Track identifier
            class_name: Predicted class
            confidence: Classification confidence
            vote_ratio: Vote ratio from evidence accumulation
            non_rejected_rois: Number of non-rejected ROIs (for logging/debugging)

        Returns:
            A list of confirmed records when:
            - A single record is confirmed in steady state (returns list of 1)
            - Deferred records are released at batch lock (returns list of N deferred + 1 current)
            None during accumulation phase
        """
        # Create record with NO smoothing
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

        # Phase 1: Accumulate - no smoothing, no confirmation
        if len(self.window_buffer) < self.window_size:
            logger.debug(
                f"[SMOOTHING] ACCUMULATE | buffer={len(self.window_buffer)}/{self.window_size} "
                f"status=raw_accumulation (no smoothing)"
            )
            return None
        
        # Phase 2: Steady State - confirm oldest
        logger.info(
            f"[SMOOTHING] STEADY_STATE | buffer={len(self.window_buffer)} "
            f"confirming_oldest_with_center_analysis"
        )
        
        # Check if we should defer this confirmation
        if self._steady_state_confirms < self.center_index:
            # Defer the oldest record (it had no future context when it entered)
            oldest_record = self.window_buffer[0]
            self.window_buffer.pop(0)
            self._deferred_records.append(oldest_record)
            self._steady_state_confirms += 1
            
            logger.info(
                f"[SMOOTHING] T{oldest_record.track_id} DEFERRED | "
                f"deferred_count={len(self._deferred_records)} "
                f"reason=no_future_context_on_entry "
                f"steady_state_confirms={self._steady_state_confirms}"
            )
            return None
        
        # Confirm oldest using center analysis
        confirmed = self._confirm_oldest_using_center_analysis()
        
        # Check for batch lock
        if not self._batch_locked and self._steady_state_confirms >= self.center_index + self.batch_lock_count:
            self._batch_locked = True
            logger.info(
                f"[SMOOTHING] BATCH_LOCKED | steady_state_confirms={self._steady_state_confirms} "
                f"releasing_deferred={len(self._deferred_records)}"
            )
            
            # Resolve deferred records
            resolved = self._resolve_deferred_records()
            
            # Return all resolved + current confirmation
            return resolved + [confirmed]
        
        # Return single confirmation
        return [confirmed]

    def check_timeout(self) -> List[ClassificationRecord]:
        """
        Check for timeout and flush remaining items if needed.

        Call this periodically to handle cases where no new items arrive
        but we have pending items in the window.

        Returns:
            List of confirmed records if timeout occurred, empty list otherwise
        """
        if not self.window_buffer and not self._deferred_records:
            return []

        elapsed = time.time() - self._last_activity_time
        if elapsed < self.window_timeout_seconds:
            return []

        logger.info(
            f"[SMOOTHING] TIMEOUT | elapsed={elapsed:.1f}s "
            f"pending={len(self.window_buffer)} deferred={len(self._deferred_records)}"
        )

        return self.flush_remaining()

    def flush_remaining(self) -> List[ClassificationRecord]:
        """
        Flush all remaining items (deferred and window buffer) with partial context.

        Called on timeout or cleanup. First resolves deferred records, then confirms 
        items in window buffer one by one.

        Returns:
            List of all confirmed records
        """
        confirmed = []

        # First, resolve any deferred records
        if self._deferred_records:
            logger.info(
                f"[SMOOTHING] FLUSH_DEFERRED | count={len(self._deferred_records)}"
            )
            resolved = self._resolve_deferred_records()
            confirmed.extend(resolved)

        # Then flush window buffer
        while self.window_buffer:
            # For partial context (buffer < window_size), still use center analysis
            if len(self.window_buffer) < self.window_size:
                logger.info(
                    f"[SMOOTHING] FLUSH | buffer={len(self.window_buffer)} context=partial"
                )
            
            # Use center analysis even with partial window
            if self.window_buffer:
                context = self._analyze_window_context()
                oldest_record = self.window_buffer[0]
                record = self._smooth_record_with_context(oldest_record, context)
                
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
        """
        Get all pending records (deferred + window buffer).
        
        Returns:
            List of all records not yet confirmed
        """
        return self._deferred_records + self.window_buffer

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
            'pending_in_window': len(self.window_buffer),
            'deferred_count': len(self._deferred_records),
            'batch_locked': self._batch_locked,
            'steady_state_confirms': self._steady_state_confirms
        }

    def get_dominant_class(self) -> Optional[str]:
        """
        Get the dominant class from all pending records (deferred + window).
        
        Uses original_class and excludes Rejected and Unknown.
        """
        all_pending = self._deferred_records + self.window_buffer
        if not all_pending:
            return None
        
        return self._get_dominant_class_from_records(all_pending, use_original=True)

    def get_pending_summary(self) -> Dict[str, int]:
        """
        Get counts of pending items grouped by class.
        
        Includes both deferred and window buffer items.
        """
        summary: Dict[str, int] = {}
        all_pending = self._deferred_records + self.window_buffer
        for record in all_pending:
            summary[record.class_name] = summary.get(record.class_name, 0) + 1
        return summary

    def cleanup(self):
        """Clean up resources."""
        if self.window_buffer or self._deferred_records:
            # Flush remaining items
            self.flush_remaining()

        self.confirmed_records.clear()
        logger.info(
            f"[BidirectionalSmoother] Cleanup complete, "
            f"smoothed {self.smoothed_records}/{self.total_records} records"
        )
