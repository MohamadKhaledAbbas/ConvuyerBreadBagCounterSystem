"""
Run-Length State Machine for classification smoothing.

Replaces the BidirectionalSmoother with a state-machine approach that:
1. Tracks consecutive classification runs (consecutive items of the same class)
2. Uses run-length thresholds to confirm batch identity
3. Detects genuine batch transitions vs. transient blips (false positives)
4. Provides rich decision metadata for debugging

States:
- ACCUMULATING:    No batch confirmed yet. Items held until a qualifying run appears.
- CONFIRMED_BATCH: Batch identity established. Items confirmed as they arrive.
- TRANSITION:      Potential batch change detected. Evaluating new run.

Key parameters (all configurable via env vars via tracking_config):
- min_run_length:           Consecutive same-class items to establish a batch (default 5)
- max_blip:                 Max consecutive non-matching items absorbed as noise (default 3)
- transition_confirm_count: Items of new class needed to confirm a batch change (default 5)

Interface:
    Same as BidirectionalSmoother — add_classification() → Optional[ClassificationRecord]
    Drop-in replacement with no changes needed at call sites.
"""

import time
from collections import deque
from typing import List, Dict, Optional, Any

from src.tracking.BidirectionalSmoother import ClassificationRecord
from src.utils.AppLogging import logger


class RunLengthStateMachine:
    """
    Run-length state machine for batch-aware classification smoothing.

    Decision flow:
      ACCUMULATING  →  Buffer items until min_run_length consecutive same-class items
                       → CONFIRMED_BATCH (batch identity locked)
      CONFIRMED_BATCH → Confirm items as they arrive; Rejected overridden to batch class
                       → TRANSITION when a different class is seen
      TRANSITION     → Evaluate whether the class change is a real transition or a blip
                       → CONFIRMED_BATCH (new class) when run >= transition_confirm_count
                       → CONFIRMED_BATCH (old class) when class reverts within max_blip items

    Drop-in replacement for BidirectionalSmoother:
      - Same add_classification() → Optional[ClassificationRecord] interface
      - Same flush_remaining(), check_timeout(), get_statistics(),
        get_pending_summary(), get_dominant_class() API
      - Exposes extra properties: state, confirmed_batch_class,
        current_run_class, current_run_length, transition_history,
        last_decision_reason
    """

    ACCUMULATING    = 'ACCUMULATING'
    CONFIRMED_BATCH = 'CONFIRMED_BATCH'
    TRANSITION      = 'TRANSITION'

    # History size limits
    _MAX_CONFIRMED_RECORDS = 100
    _MAX_TRANSITION_HISTORY = 20

    def __init__(
        self,
        min_run_length: int = 5,
        max_blip: int = 3,
        transition_confirm_count: int = 5,
        window_timeout_seconds: float = 30.0,
    ):
        """
        Initialise the state machine.

        Args:
            min_run_length:           Consecutive same-class items needed to lock a batch.
            max_blip:                 Maximum consecutive non-matching items treated as noise.
            transition_confirm_count: Items of the new class needed to confirm a batch change.
            window_timeout_seconds:   Flush pending items after this period of inactivity.
        """
        self.min_run_length          = min_run_length
        self.max_blip                = max_blip
        self.transition_confirm_count = transition_confirm_count
        self.window_timeout_seconds  = window_timeout_seconds

        # Compatibility shim: BidirectionalSmoother exposes window_size
        self.window_size = min_run_length

        # ── State ────────────────────────────────────────────────────────────
        self._state: str = self.ACCUMULATING
        self._confirmed_batch_class: Optional[str] = None

        # ACCUMULATING phase buffers
        self._accum_buffer: List[ClassificationRecord] = []   # pre-run items
        self._current_run:  List[ClassificationRecord] = []   # current same-class streak
        self._current_run_class: Optional[str] = None

        # TRANSITION phase buffer
        self._transition_buffer: List[ClassificationRecord] = []
        self._transition_candidate_class: Optional[str] = None

        # Confirmed-but-not-yet-returned items (emitted one per add_classification call)
        self._output_queue: deque = deque()

        # ── History & statistics ─────────────────────────────────────────────
        self.confirmed_records:  List[ClassificationRecord]  = []
        self.transition_history: List[Dict[str, Any]]        = []

        self.total_records    = 0
        self.smoothed_records = 0
        self._confirm_count   = 0

        # ── Timing ───────────────────────────────────────────────────────────
        self._last_activity_time  = time.time()
        self._last_decision_reason: str = ''

        logger.info(
            f"[RunLengthStateMachine] Initialized | "
            f"min_run={min_run_length} max_blip={max_blip} "
            f"transition_confirm={transition_confirm_count}"
        )

    # ── Public read-only properties ───────────────────────────────────────────

    @property
    def state(self) -> str:
        """Current state machine state string."""
        return self._state

    @property
    def confirmed_batch_class(self) -> Optional[str]:
        """Locked batch class, or None while still ACCUMULATING."""
        return self._confirmed_batch_class

    @property
    def current_run_length(self) -> int:
        """Length of the active run (transition buffer when in TRANSITION state)."""
        if self._state == self.TRANSITION:
            return len(self._transition_buffer)
        return len(self._current_run)

    @property
    def current_run_class(self) -> Optional[str]:
        """Class of the active run (candidate class when in TRANSITION state)."""
        if self._state == self.TRANSITION:
            return self._transition_candidate_class
        return self._current_run_class

    @property
    def last_decision_reason(self) -> str:
        """Human-readable reason for the most recent smoothing decision."""
        return self._last_decision_reason

    # ── Core public interface (matches BidirectionalSmoother) ─────────────────

    def add_classification(
        self,
        track_id: int,
        class_name: str,
        confidence: float,
        vote_ratio: float,
        non_rejected_rois: int = 0
    ) -> Optional[ClassificationRecord]:
        """
        Submit a new classification to the state machine.

        Returns one confirmed ClassificationRecord when one is ready, or None
        while items are still accumulating / under evaluation.

        Args:
            track_id:         Track identifier.
            class_name:       Predicted class.
            confidence:       Classification confidence [0, 1].
            vote_ratio:       Vote ratio from evidence accumulation.
            non_rejected_rois: Number of non-rejected ROIs (for metadata only).

        Returns:
            A confirmed ClassificationRecord, or None.
        """
        record = ClassificationRecord(
            track_id=track_id,
            class_name=class_name,
            confidence=confidence,
            vote_ratio=vote_ratio,
            timestamp=time.time(),
            non_rejected_rois=non_rejected_rois
        )
        self.total_records += 1
        self._last_activity_time = time.time()

        # Route through state-machine handlers (may populate _output_queue)
        self._process_item(record)

        # Emit exactly one confirmed record per call
        return self._dequeue_one()

    def check_timeout(self) -> List[ClassificationRecord]:
        """
        Flush pending items if the inactivity timeout has elapsed.

        Returns:
            List of confirmed records on timeout; empty list otherwise.
        """
        if not self._has_pending():
            return []
        elapsed = time.time() - self._last_activity_time
        if elapsed < self.window_timeout_seconds:
            return []
        logger.info(
            f"[RunLengthStateMachine] TIMEOUT | elapsed={elapsed:.1f}s "
            f"pending={self._pending_count()}"
        )
        return self.flush_remaining()

    def flush_remaining(self) -> List[ClassificationRecord]:
        """
        Flush all pending items immediately (on timeout or cleanup).

        Items that cannot be confirmed normally are overridden to the confirmed
        batch class (or 'Unknown' if no batch has been established).

        Returns:
            List of all confirmed records produced by the flush.
        """
        batch_class = self._confirmed_batch_class

        # Flush accumulation pre-run buffer
        for item in self._accum_buffer:
            if item.class_name == 'Rejected' or (batch_class and item.class_name != batch_class):
                target = batch_class or 'Unknown'
                item = self._create_smoothed(item, target, 'flush_accum')
            self._output_queue.append(item)
        self._accum_buffer = []

        # Flush the current accumulation run
        if self._state == self.ACCUMULATING and self._current_run:
            if len(self._current_run) >= self.min_run_length:
                batch_class = self._current_run_class
                self._confirmed_batch_class = batch_class
            for item in self._current_run:
                if batch_class and item.class_name != batch_class:
                    item = self._create_smoothed(item, batch_class, 'flush_short_run')
                elif not batch_class and item.class_name == 'Rejected':
                    item = self._create_smoothed(item, 'Unknown', 'flush_rejected')
                self._output_queue.append(item)
            self._current_run = []
            self._current_run_class = None

        # Flush transition buffer (absorb to confirmed batch class)
        for item in self._transition_buffer:
            if batch_class and item.class_name != batch_class:
                item = self._create_smoothed(item, batch_class, 'flush_transition')
            self._output_queue.append(item)
        self._transition_buffer = []
        self._transition_candidate_class = None

        # Drain output queue and collect results
        confirmed: List[ClassificationRecord] = []
        while self._output_queue:
            rec = self._dequeue_one()
            if rec:
                confirmed.append(rec)

        self._state = self.ACCUMULATING
        if confirmed:
            logger.info(f"[RunLengthStateMachine] FLUSHED | confirmed={len(confirmed)}")
        return confirmed

    def finalize_batch(self) -> List[ClassificationRecord]:
        """Alias for flush_remaining() — backward compatibility."""
        return self.flush_remaining()

    def get_pending_records(self) -> List[ClassificationRecord]:
        """Return all records not yet confirmed (safe copy)."""
        return (
            list(self._accum_buffer) +
            list(self._current_run) +
            list(self._transition_buffer) +
            list(self._output_queue)
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Return smoothing statistics (same keys as BidirectionalSmoother)."""
        pending = self._pending_count()
        return {
            'total_records':    self.total_records,
            'smoothed_records': self.smoothed_records,
            'smoothing_rate': (
                self.smoothed_records / self.total_records
                if self.total_records > 0 else 0.0
            ),
            'confirmed_count':       self._confirm_count,
            'pending_in_window':     pending,
            # Extra fields exposed by the state machine
            'state':                 self._state,
            'confirmed_batch_class': self._confirmed_batch_class,
            'current_run_class':     self.current_run_class,
            'current_run_length':    self.current_run_length,
        }

    def get_dominant_class(self) -> Optional[str]:
        """Return the confirmed batch class (equivalent to BidirectionalSmoother window dominant)."""
        return self._confirmed_batch_class

    def get_pending_summary(self) -> Dict[str, int]:
        """Return per-class counts of all pending (unconfirmed) items."""
        summary: Dict[str, int] = {}
        for record in self.get_pending_records():
            summary[record.class_name] = summary.get(record.class_name, 0) + 1
        return summary

    def cleanup(self):
        """Release resources; flush any remaining items."""
        if self._has_pending():
            self.flush_remaining()
        self.confirmed_records.clear()
        logger.info(
            f"[RunLengthStateMachine] Cleanup complete — "
            f"smoothed {self.smoothed_records}/{self.total_records} records"
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _has_pending(self) -> bool:
        return bool(
            self._accum_buffer or self._current_run or
            self._transition_buffer or self._output_queue
        )

    def _pending_count(self) -> int:
        return (
            len(self._accum_buffer) +
            len(self._current_run) +
            len(self._transition_buffer) +
            len(self._output_queue)
        )

    def _create_smoothed(
        self,
        original: ClassificationRecord,
        new_class: str,
        reason: str
    ) -> ClassificationRecord:
        """Return a new ClassificationRecord with an overridden class and smoothed=True."""
        self.smoothed_records += 1
        self._last_decision_reason = reason
        smoothed = ClassificationRecord(
            track_id=original.track_id,
            class_name=new_class,
            confidence=original.confidence,
            vote_ratio=original.vote_ratio,
            timestamp=original.timestamp,
            window_position=None,
            smoothed=True,
            original_class=original.class_name,
            non_rejected_rois=original.non_rejected_rois
        )
        logger.info(
            f"[RLSM] T{original.track_id} SMOOTHED | "
            f"{original.class_name}->{new_class} reason={reason}"
        )
        return smoothed

    def _emit(self, record: ClassificationRecord) -> ClassificationRecord:
        """Mark a record as confirmed, assign a sequence position, add to history."""
        record.window_position = self._confirm_count
        self._confirm_count += 1
        self.confirmed_records.append(record)
        if len(self.confirmed_records) > self._MAX_CONFIRMED_RECORDS:
            self.confirmed_records = self.confirmed_records[-self._MAX_CONFIRMED_RECORDS:]
        logger.info(
            f"[RLSM] T{record.track_id} CONFIRMED | "
            f"class={record.class_name} smoothed={record.smoothed} state={self._state}"
        )
        return record

    def _dequeue_one(self) -> Optional[ClassificationRecord]:
        """Pop one record from the output queue and emit it; return None if empty."""
        if self._output_queue:
            return self._emit(self._output_queue.popleft())
        return None

    def _record_transition(
        self,
        from_class: Optional[str],
        to_class: str,
        transition_type: str,
        run_length: int
    ):
        """Append an entry to transition_history (capped at 20 entries)."""
        self.transition_history.append({
            'from_class':  from_class,
            'to_class':    to_class,
            'timestamp':   time.time(),
            'type':        transition_type,
            'run_length':  run_length,
        })
        if len(self.transition_history) > self._MAX_TRANSITION_HISTORY:
            self.transition_history = self.transition_history[-self._MAX_TRANSITION_HISTORY:]

    # ── State-machine dispatch ────────────────────────────────────────────────

    def _process_item(self, record: ClassificationRecord):
        """Dispatch the incoming record to the active state handler."""
        if self._state == self.ACCUMULATING:
            self._process_accumulating(record)
        elif self._state == self.CONFIRMED_BATCH:
            self._process_confirmed_batch(record)
        elif self._state == self.TRANSITION:
            self._process_transition(record)

    # ── ACCUMULATING ──────────────────────────────────────────────────────────

    def _process_accumulating(self, record: ClassificationRecord):
        """
        Buffer items until a qualifying run of length >= min_run_length is seen.

        Rejected items are buffered separately; items of different classes that
        don't form a qualifying run are discarded and placed into the pre-run
        buffer for later absorption once the batch class is known.
        """
        class_name = record.class_name

        # Rejected during accumulation: hold in pre-run buffer
        if class_name == 'Rejected':
            self._accum_buffer.append(record)
            logger.debug(
                f"[RLSM] ACCUM | T{record.track_id} Rejected → buffered "
                f"(buf={len(self._accum_buffer)})"
            )
            return

        if not self._current_run_class or self._current_run_class == class_name:
            # Continue (or start) the current run
            self._current_run.append(record)
            self._current_run_class = class_name
            logger.debug(
                f"[RLSM] ACCUM | T{record.track_id} "
                f"run={class_name}×{len(self._current_run)}"
            )
        else:
            # Different class encountered
            if len(self._current_run) >= self.min_run_length:
                # Current run qualifies → lock batch, then handle new item as transition
                self._finalise_accumulation()
                self._process_confirmed_batch(record)
            else:
                # Short run — treat as noise, start fresh with the new class
                logger.debug(
                    f"[RLSM] ACCUM | short run discarded "
                    f"class={self._current_run_class}×{len(self._current_run)} "
                    f"switching_to={class_name}"
                )
                self._accum_buffer.extend(self._current_run)
                self._current_run = [record]
                self._current_run_class = class_name
            return

        # Check whether the run has reached the qualification threshold
        if len(self._current_run) >= self.min_run_length:
            self._finalise_accumulation()

    def _finalise_accumulation(self):
        """Lock the batch class and emit all buffered + run items."""
        batch_class = self._current_run_class
        self._confirmed_batch_class = batch_class
        self._state = self.CONFIRMED_BATCH
        logger.info(
            f"[RLSM] BATCH_ESTABLISHED | class={batch_class} "
            f"pre_buf={len(self._accum_buffer)} run={len(self._current_run)}"
        )

        # Override pre-run items to the confirmed batch class
        for item in self._accum_buffer:
            if item.class_name != batch_class:
                item = self._create_smoothed(item, batch_class, 'pre_run_absorbed')
            self._output_queue.append(item)
        self._accum_buffer = []

        # Emit the qualifying run as-is
        for item in self._current_run:
            self._output_queue.append(item)
        self._current_run = []
        self._current_run_class = batch_class

        self._record_transition(None, batch_class, 'initial_batch', self.min_run_length)

    # ── CONFIRMED_BATCH ───────────────────────────────────────────────────────

    def _process_confirmed_batch(self, record: ClassificationRecord):
        """
        Confirm items immediately within a locked batch.

        - Rejected → override to confirmed batch class.
        - Matches batch class → confirm as-is.
        - Different class → enter TRANSITION to evaluate.
        """
        batch_class = self._confirmed_batch_class
        class_name  = record.class_name

        if class_name == 'Rejected':
            record = self._create_smoothed(record, batch_class, 'rejected_to_batch')
            self._output_queue.append(record)
            return

        if class_name == batch_class:
            self._last_decision_reason = 'matches_batch'
            self._output_queue.append(record)
            return

        # Different class — begin transition evaluation
        logger.info(
            f"[RLSM] T{record.track_id} TRANSITION_START | "
            f"batch={batch_class} new={class_name}"
        )
        self._last_decision_reason = 'transition_start'
        self._state = self.TRANSITION
        self._transition_candidate_class = class_name
        self._transition_buffer = [record]

    # ── TRANSITION ────────────────────────────────────────────────────────────

    def _process_transition(self, record: ClassificationRecord):
        """
        Evaluate whether the new class run is a real batch change or a blip.

        - Continues candidate class → extend transition buffer.
          → Reaches transition_confirm_count: confirm real transition.
        - Returns to old batch class → absorb buffer as blip.
        - Mixed / third class → extend buffer; absorb if too large.
        - Rejected → treat as old batch class (blip continuation).
        """
        batch_class     = self._confirmed_batch_class
        candidate_class = self._transition_candidate_class
        class_name      = record.class_name

        # Rejected during transition → treat as old batch class
        if class_name == 'Rejected':
            record     = self._create_smoothed(record, batch_class, 'rejected_during_transition')
            class_name = batch_class

        if class_name == candidate_class:
            # Extends the candidate run
            self._transition_buffer.append(record)
            logger.debug(
                f"[RLSM] TRANSITION | "
                f"candidate={candidate_class}×{len(self._transition_buffer)}"
            )
            if len(self._transition_buffer) >= self.transition_confirm_count:
                self._confirm_transition()

        elif class_name == batch_class:
            # Reverted to old class → absorb blip
            self._absorb_blip(record, reason='class_reverted')

        else:
            # Third / mixed class — extend buffer; force-absorb if it grows too large
            self._transition_buffer.append(record)
            max_wait = self.max_blip + self.transition_confirm_count
            if len(self._transition_buffer) > max_wait:
                logger.info(
                    f"[RLSM] TRANSITION_OVERFLOW | "
                    f"buf={len(self._transition_buffer)} absorbing to {batch_class}"
                )
                self._absorb_blip(None, reason='transition_overflow')

    def _confirm_transition(self):
        """Commit a confirmed batch change and emit all transition-buffer items."""
        old_class = self._confirmed_batch_class
        new_class = self._transition_candidate_class
        logger.info(
            f"[RLSM] TRANSITION_CONFIRMED | "
            f"{old_class}->{new_class} run={len(self._transition_buffer)}"
        )
        self._confirmed_batch_class       = new_class
        self._state                       = self.CONFIRMED_BATCH
        self._last_decision_reason        = f'batch_transition:{old_class}->{new_class}'

        for item in self._transition_buffer:
            self._output_queue.append(item)
        self._transition_buffer           = []
        self._transition_candidate_class  = None

        self._record_transition(
            old_class, new_class, 'batch_transition', self.transition_confirm_count
        )

    def _absorb_blip(
        self,
        extra_record: Optional[ClassificationRecord],
        reason: str
    ):
        """
        Override all transition-buffer items to the confirmed batch class (blip absorption).

        Args:
            extra_record: The triggering item that should also be confirmed (may be None).
            reason:       Descriptive reason for absorption.
        """
        batch_class = self._confirmed_batch_class
        logger.info(
            f"[RLSM] BLIP_ABSORBED | "
            f"candidate={self._transition_candidate_class} "
            f"len={len(self._transition_buffer)} reason={reason} "
            f"absorb_to={batch_class}"
        )
        self._last_decision_reason = f'blip_absorbed:{reason}'

        for item in self._transition_buffer:
            if item.class_name != batch_class:
                item = self._create_smoothed(item, batch_class, 'blip_absorbed')
            self._output_queue.append(item)
        self._transition_buffer          = []
        self._transition_candidate_class = None
        self._state                      = self.CONFIRMED_BATCH

        if extra_record is not None:
            self._output_queue.append(extra_record)
