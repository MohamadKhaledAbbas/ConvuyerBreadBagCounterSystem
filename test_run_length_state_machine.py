"""
Tests for RunLengthStateMachine.

Validates the core state-machine logic:
1. ACCUMULATING → CONFIRMED_BATCH after min_run_length consecutive items
2. Pre-run items (including Rejected) absorbed to confirmed batch class
3. Steady-state confirmation: matching items confirmed immediately
4. Rejected override: always overridden to confirmed batch class
5. Blip absorption: short non-matching runs absorbed back to batch class
6. Real transition: long non-matching run confirms a new batch
7. Transition overflow: mixed/endless transition absorbed after max_blip + confirm
8. Flush / timeout behaviour
9. Statistics and interface compatibility
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.tracking.RunLengthStateMachine import RunLengthStateMachine
from src.tracking.BidirectionalSmoother import ClassificationRecord


# ── helpers ──────────────────────────────────────────────────────────────────

def make_sm(min_run=3, max_blip=2, transition_confirm=3, timeout=30.0):
    return RunLengthStateMachine(
        min_run_length=min_run,
        max_blip=max_blip,
        transition_confirm_count=transition_confirm,
        window_timeout_seconds=timeout,
    )


def add(sm, class_name, track_id=1, confidence=0.95):
    return sm.add_classification(
        track_id=track_id,
        class_name=class_name,
        confidence=confidence,
        vote_ratio=1.0,
    )


def add_many(sm, class_name, count, start_id=1):
    results = []
    for i in range(count):
        r = add(sm, class_name, track_id=start_id + i)
        results.append(r)
    return results


def confirmed_classes(records):
    """Return list of class names from a list of ClassificationRecord|None."""
    return [r.class_name for r in records if r is not None]


# ── test 1: ACCUMULATING → CONFIRMED_BATCH ───────────────────────────────────

def test_accumulation_establishes_batch():
    sm = make_sm(min_run=3)
    # First 2 items return None (accumulating)
    assert add(sm, 'Brown', 1) is None
    assert add(sm, 'Brown', 2) is None
    assert sm.state == RunLengthStateMachine.ACCUMULATING
    # 3rd item reaches threshold — batch confirmed; items start flowing out
    r = add(sm, 'Brown', 3)
    assert r is not None
    assert r.class_name == 'Brown'
    assert sm.confirmed_batch_class == 'Brown'
    assert sm.state == RunLengthStateMachine.CONFIRMED_BATCH
    print("PASS test_accumulation_establishes_batch")


# ── test 2: pre-run items absorbed to batch class ────────────────────────────

def test_pre_run_items_absorbed():
    sm = make_sm(min_run=3)
    # First item is Rejected — goes to pre-run buffer
    r0 = add(sm, 'Rejected', 1)
    assert r0 is None
    # Then 3 of 'Brown' — confirm accumulation
    results = add_many(sm, 'Brown', 3, start_id=2)
    emitted = [r for r in results if r is not None]
    # We should get at least 1 result (the Rejected + run items start draining)
    # The Rejected item should be smoothed to Brown
    all_confirmed = []
    for r in results:
        if r: all_confirmed.append(r)
    # Drain remaining
    for i in range(10):
        r = add(sm, 'Brown', 100 + i)
        if r: all_confirmed.append(r)
    classes = [r.class_name for r in all_confirmed]
    assert 'Rejected' not in classes, f"Rejected should have been absorbed; got: {classes}"
    assert all(c == 'Brown' for c in classes), f"All should be Brown; got: {classes}"
    print("PASS test_pre_run_items_absorbed")


# ── test 3: steady-state matching items confirmed immediately ─────────────────

def test_steady_state_confirmation():
    sm = make_sm(min_run=3)
    # Establish batch
    add_many(sm, 'Brown', 4, start_id=1)
    sm_state_before = sm.state
    # After batch confirmed, each new matching item should be confirmed promptly
    results = add_many(sm, 'Brown', 5, start_id=100)
    # Some may be drained from queue — all should be Brown
    for r in results:
        if r is not None:
            assert r.class_name == 'Brown', f"Expected Brown, got {r.class_name}"
    assert sm.confirmed_batch_class == 'Brown'
    print("PASS test_steady_state_confirmation")


# ── test 4: Rejected always overridden to batch class ────────────────────────

def test_rejected_overridden_to_batch():
    sm = make_sm(min_run=3)
    # Establish batch
    add_many(sm, 'Brown', 4, start_id=1)
    # Feed Rejected items
    results = [add(sm, 'Rejected', 200), add(sm, 'Rejected', 201)]
    # Drain remaining output — 5 more calls is enough to flush the 4-item batch
    # establishment queue (3 run items + 1 extra that was already queued)
    for i in range(5):
        r = add(sm, 'Brown', 300 + i)
        if r:
            results.append(r)
    # All confirmed records should be Brown (Rejected → Brown)
    for r in results:
        if r is not None:
            assert r.class_name == 'Brown', f"Expected Brown, got {r.class_name}"
    # The two Rejected-origin records should have been smoothed (check confirmed_records history)
    smoothed_from_rejected = [
        r for r in sm.confirmed_records if r.original_class == 'Rejected'
    ]
    assert len(smoothed_from_rejected) >= 2, (
        f"Expected at least 2 smoothed-from-Rejected records; got {smoothed_from_rejected}"
    )
    for r in smoothed_from_rejected:
        assert r.class_name == 'Brown', f"Rejected should become Brown, got {r.class_name}"
        assert r.smoothed is True
    print("PASS test_rejected_overridden_to_batch")


# ── test 5: blip absorption ───────────────────────────────────────────────────

def test_blip_absorption():
    """A short run (≤ max_blip) of a different class is absorbed as noise."""
    sm = make_sm(min_run=3, max_blip=2, transition_confirm=3)
    # Establish Brown batch
    for _ in range(4):
        add(sm, 'Brown', 1)
    # Introduce 2 blip items of a different class
    b1 = add(sm, 'Red', 50)
    b2 = add(sm, 'Red', 51)
    # Return to Brown — should trigger blip absorption
    r = add(sm, 'Brown', 52)
    # Collect all outputs
    all_out = [b1, b2, r]
    confirmed = [x for x in all_out if x is not None]
    for rec in confirmed:
        assert rec.class_name == 'Brown', (
            f"Blip items should be absorbed to Brown, got {rec.class_name}"
        )
    assert sm.state == RunLengthStateMachine.CONFIRMED_BATCH
    assert sm.confirmed_batch_class == 'Brown'
    print("PASS test_blip_absorption")


# ── test 6: real transition ────────────────────────────────────────────────────

def test_real_transition():
    """A run of transition_confirm_count new-class items confirms the transition."""
    sm = make_sm(min_run=3, max_blip=2, transition_confirm=3)
    # Establish Brown batch
    for i in range(4):
        add(sm, 'Brown', i)
    # Start a real new batch: 3 consecutive 'Red' items
    add(sm, 'Red', 10)
    add(sm, 'Red', 11)
    add(sm, 'Red', 12)   # threshold reached here
    # After 3 Red items, batch should transition to Red
    assert sm.confirmed_batch_class == 'Red', (
        f"Expected Red after transition, got {sm.confirmed_batch_class}"
    )
    assert sm.state == RunLengthStateMachine.CONFIRMED_BATCH
    # Items from the transition run should appear as Red in confirmed_records
    red_records = [r for r in sm.confirmed_records if r.class_name == 'Red']
    assert len(red_records) >= 1, (
        f"Expected at least 1 Red confirmed record; got {[r.class_name for r in sm.confirmed_records]}"
    )
    # Record a transition
    assert any(t['to_class'] == 'Red' for t in sm.transition_history), (
        "transition_history should contain Red transition"
    )
    print("PASS test_real_transition")


# ── test 7: transition overflow ───────────────────────────────────────────────

def test_transition_overflow():
    """
    If the transition buffer grows beyond max_blip + transition_confirm without
    resolution, items are force-absorbed to the old batch class.
    """
    sm = make_sm(min_run=3, max_blip=2, transition_confirm=3)
    # Establish Brown batch
    for i in range(4):
        add(sm, 'Brown', i)
    # Feed a mix of classes (Red starts, then Greens keep the buffer growing)
    # After max_blip + transition_confirm + 1 items, overflow triggers
    max_wait = sm.max_blip + sm.transition_confirm_count
    add(sm, 'Red', 10)                          # starts candidate=Red, buf len=1
    for i in range(max_wait):                   # 5 Greens push buf beyond max_wait
        add(sm, 'Green', 20 + i)
    # Overflow should have fired; return one Brown to settle any new mini-transition
    add(sm, 'Brown', 99)
    # Now state should be CONFIRMED_BATCH with batch still Brown
    assert sm.state == RunLengthStateMachine.CONFIRMED_BATCH, (
        f"Expected CONFIRMED_BATCH after settling, got {sm.state}"
    )
    assert sm.confirmed_batch_class == 'Brown', (
        f"Expected batch_class=Brown after absorption, got {sm.confirmed_batch_class}"
    )
    print("PASS test_transition_overflow")


# ── test 8: flush_remaining ───────────────────────────────────────────────────

def test_flush_remaining():
    """flush_remaining emits all pending items."""
    sm = make_sm(min_run=5)
    # Add fewer than min_run_length items so batch is never confirmed
    add(sm, 'Brown', 1)
    add(sm, 'Brown', 2)
    add(sm, 'Brown', 3)
    assert sm.state == RunLengthStateMachine.ACCUMULATING
    flushed = sm.flush_remaining()
    # All 3 items should be flushed (partial run, no confirmed batch class)
    assert len(flushed) == 3, f"Expected 3 flushed records, got {len(flushed)}"
    print("PASS test_flush_remaining")


# ── test 9: statistics and interface compatibility ────────────────────────────

def test_statistics_interface():
    sm = make_sm(min_run=3)
    add_many(sm, 'Brown', 5, start_id=1)
    stats = sm.get_statistics()
    assert 'total_records' in stats
    assert 'smoothed_records' in stats
    assert 'smoothing_rate' in stats
    assert 'confirmed_count' in stats
    assert 'pending_in_window' in stats
    assert stats['total_records'] == 5
    # get_dominant_class should return confirmed batch class
    assert sm.get_dominant_class() == sm.confirmed_batch_class
    # get_pending_summary should be a dict
    summary = sm.get_pending_summary()
    assert isinstance(summary, dict)
    # window_size attribute (compat shim)
    assert sm.window_size == sm.min_run_length
    print("PASS test_statistics_interface")


# ── test 10: short accumulation run resets ────────────────────────────────────

def test_short_accum_run_resets():
    """
    If we see a short run of class A (< min_run_length) then switch to class B,
    class A run is discarded and class B becomes the new candidate.
    """
    sm = make_sm(min_run=4)
    add(sm, 'Red',   1)
    add(sm, 'Red',   2)   # only 2 Red — below threshold
    add(sm, 'Brown', 3)   # switches class; Red run should be discarded
    add(sm, 'Brown', 4)
    add(sm, 'Brown', 5)
    r = add(sm, 'Brown', 6)  # 4th Brown — should confirm batch as Brown
    assert sm.confirmed_batch_class == 'Brown', (
        f"Expected batch=Brown, got {sm.confirmed_batch_class}"
    )
    print("PASS test_short_accum_run_resets")


# ── run all tests ─────────────────────────────────────────────────────────────

# ── test 11: rejected during transition → transition confirms ─────────────────

def test_rejected_neutral_during_confirmed_transition():
    """
    Rejected items during a genuine batch transition should NOT suppress
    the transition.  They are held neutrally and assigned to the NEW batch
    class once the transition is confirmed.

    Sequence: Brown batch → Red, Rejected, Red, Red → Red batch confirmed
    """
    sm = make_sm(min_run=3, max_blip=2, transition_confirm=3)
    # Establish Brown batch
    add_many(sm, 'Brown', 4, start_id=1)
    assert sm.confirmed_batch_class == 'Brown'

    # Start transition: Red, Rejected, Red, Red
    add(sm, 'Red', 10)        # evidence=1
    assert sm.state == RunLengthStateMachine.TRANSITION
    add(sm, 'Rejected', 11)   # neutral — should NOT count as Brown evidence
    assert sm.state == RunLengthStateMachine.TRANSITION
    add(sm, 'Red', 12)        # evidence=2
    assert sm.state == RunLengthStateMachine.TRANSITION
    add(sm, 'Red', 13)        # evidence=3 → threshold reached

    assert sm.confirmed_batch_class == 'Red', (
        f"Expected Red after transition, got {sm.confirmed_batch_class}"
    )
    assert sm.state == RunLengthStateMachine.CONFIRMED_BATCH

    # The Rejected item (track 11) should have been smoothed to Red (new batch)
    smoothed_rejected = [
        r for r in sm.confirmed_records
        if r.original_class == 'Rejected' and r.track_id == 11
    ]
    # It may not be emitted yet (output queue drains one per call), so drain
    for i in range(10):
        add(sm, 'Red', 100 + i)
    smoothed_rejected = [
        r for r in sm.confirmed_records
        if r.original_class == 'Rejected' and r.track_id == 11
    ]
    assert len(smoothed_rejected) == 1, (
        f"Expected 1 smoothed Rejected→Red record for T11; "
        f"found {len(smoothed_rejected)}"
    )
    assert smoothed_rejected[0].class_name == 'Red', (
        f"Neutral Rejected should be assigned to NEW batch (Red), "
        f"got {smoothed_rejected[0].class_name}"
    )
    assert smoothed_rejected[0].smoothed is True
    print("PASS test_rejected_neutral_during_confirmed_transition")


# ── test 12: rejected during transition → blip absorbed ──────────────────────

def test_rejected_neutral_during_blip_absorption():
    """
    If a transition turns out to be a blip (reverts to old batch),
    neutral Rejected items are assigned to the OLD batch class.

    Sequence: Brown batch → Red, Rejected, Brown → blip absorbed to Brown
    """
    sm = make_sm(min_run=3, max_blip=2, transition_confirm=3)
    # Establish Brown batch
    add_many(sm, 'Brown', 4, start_id=1)
    assert sm.confirmed_batch_class == 'Brown'

    # Start transition, then revert
    add(sm, 'Red', 10)        # evidence=1
    add(sm, 'Rejected', 11)   # neutral
    add(sm, 'Brown', 12)      # revert → blip absorbed

    assert sm.confirmed_batch_class == 'Brown'
    assert sm.state == RunLengthStateMachine.CONFIRMED_BATCH

    # Drain the output queue
    for i in range(10):
        add(sm, 'Brown', 100 + i)

    # The Rejected item (track 11) should have been smoothed to Brown (old batch)
    smoothed_rejected = [
        r for r in sm.confirmed_records
        if r.original_class == 'Rejected' and r.track_id == 11
    ]
    assert len(smoothed_rejected) == 1, (
        f"Expected 1 smoothed Rejected→Brown record for T11; "
        f"found {len(smoothed_rejected)}"
    )
    assert smoothed_rejected[0].class_name == 'Brown', (
        f"Neutral Rejected should be assigned to OLD batch (Brown) on blip, "
        f"got {smoothed_rejected[0].class_name}"
    )
    print("PASS test_rejected_neutral_during_blip_absorption")


# ── test 13: interleaved rejected + candidate confirms transition ─────────────

def test_interleaved_rejected_and_candidate():
    """
    Rejected items interleaved with candidate items should not prevent
    the candidate evidence counter from reaching the threshold.

    Sequence: Brown → Red, Rejected, Red, Rejected, Red → Red confirmed
    """
    sm = make_sm(min_run=3, max_blip=2, transition_confirm=3)
    add_many(sm, 'Brown', 4, start_id=1)

    # Interleaved: Red, Rejected, Red, Rejected, Red
    add(sm, 'Red', 10)        # evidence=1
    add(sm, 'Rejected', 11)   # neutral
    add(sm, 'Red', 12)        # evidence=2
    add(sm, 'Rejected', 13)   # neutral
    add(sm, 'Red', 14)        # evidence=3 → confirm

    assert sm.confirmed_batch_class == 'Red', (
        f"Expected Red, got {sm.confirmed_batch_class}"
    )
    assert sm.state == RunLengthStateMachine.CONFIRMED_BATCH

    # Both Rejected items should be smoothed to Red
    for i in range(10):
        add(sm, 'Red', 100 + i)
    rejected_records = [
        r for r in sm.confirmed_records if r.original_class == 'Rejected'
    ]
    for r in rejected_records:
        assert r.class_name == 'Red', (
            f"Neutral Rejected should become Red, got {r.class_name}"
        )
    print("PASS test_interleaved_rejected_and_candidate")


# ── test 14: many rejected don't trigger overflow ─────────────────────────────

def test_many_rejected_dont_cause_overflow():
    """
    A stream of Rejected items during transition should NOT trigger overflow
    (they are neutral). Only third-class items count toward overflow.

    Sequence: Brown → Red, Rejected x4, Red, Red → Red confirmed
    (With max_blip=2, transition_confirm=3, even 4 Rejected items don't overflow)
    """
    sm = make_sm(min_run=3, max_blip=2, transition_confirm=3)
    add_many(sm, 'Brown', 4, start_id=1)

    add(sm, 'Red', 10)         # evidence=1
    add(sm, 'Rejected', 20)    # neutral
    add(sm, 'Rejected', 21)    # neutral
    add(sm, 'Rejected', 22)    # neutral
    add(sm, 'Rejected', 23)    # neutral — still in TRANSITION, not absorbed
    assert sm.state == RunLengthStateMachine.TRANSITION, (
        f"Expected TRANSITION with 4 neutrals, got {sm.state}"
    )
    add(sm, 'Red', 30)         # evidence=2
    add(sm, 'Red', 31)         # evidence=3 → confirm

    assert sm.confirmed_batch_class == 'Red', (
        f"Expected Red after transition with neutrals, got {sm.confirmed_batch_class}"
    )
    print("PASS test_many_rejected_dont_cause_overflow")


# ── test 15: third-class items still trigger overflow ─────────────────────────

def test_third_class_triggers_overflow():
    """
    Third-class items (not candidate, not old batch, not Rejected) should
    still trigger overflow via the noise counter.
    """
    sm = make_sm(min_run=3, max_blip=2, transition_confirm=3)
    add_many(sm, 'Brown', 4, start_id=1)

    add(sm, 'Red', 10)     # candidate=Red, evidence=1
    add(sm, 'Green', 20)   # noise=1
    add(sm, 'Green', 21)   # noise=2
    add(sm, 'Green', 22)   # noise=3 > max_blip(2) → overflow → absorb to Brown

    # After overflow, should settle back to Brown
    # May need a Brown to settle if overflow triggers mini-transition
    for i in range(3):
        add(sm, 'Brown', 100 + i)

    assert sm.confirmed_batch_class == 'Brown', (
        f"Expected Brown after overflow, got {sm.confirmed_batch_class}"
    )
    assert sm.state == RunLengthStateMachine.CONFIRMED_BATCH
    print("PASS test_third_class_triggers_overflow")


# ── test 16: rejected before transition start (in confirmed batch) ────────────

def test_rejected_before_transition_start():
    """
    Rejected items arriving in CONFIRMED_BATCH (before any transition starts)
    should still be immediately smoothed to the current batch class.
    """
    sm = make_sm(min_run=3, max_blip=2, transition_confirm=3)
    add_many(sm, 'Brown', 4, start_id=1)

    # Rejected in steady state
    r1 = add(sm, 'Rejected', 50)
    r2 = add(sm, 'Rejected', 51)
    # Then a real transition
    add(sm, 'Red', 60)
    add(sm, 'Red', 61)
    add(sm, 'Red', 62)  # confirm transition

    assert sm.confirmed_batch_class == 'Red'

    # Drain and check
    for i in range(10):
        add(sm, 'Red', 200 + i)

    # The Rejected items before transition should be Brown (old batch)
    rejected_before = [
        r for r in sm.confirmed_records
        if r.original_class == 'Rejected' and r.track_id in (50, 51)
    ]
    for r in rejected_before:
        assert r.class_name == 'Brown', (
            f"Pre-transition Rejected should be Brown, got {r.class_name}"
        )
    print("PASS test_rejected_before_transition_start")


# ── test 17: rejected-only stream hits hard cap ──────────────────────────────

def test_rejected_only_stream_hits_hard_cap():
    """
    If only Rejected items arrive during transition (no candidate evidence,
    no noise), the hard cap (_MAX_TRANSITION_BUFFER) eventually forces
    resolution by absorbing to old batch (since evidence < threshold).
    """
    sm = make_sm(min_run=3, max_blip=2, transition_confirm=3)
    add_many(sm, 'Brown', 4, start_id=1)

    # Start transition with one Red
    add(sm, 'Red', 10)  # evidence=1
    assert sm.state == RunLengthStateMachine.TRANSITION

    # Feed only Rejected items up to the hard cap
    cap = RunLengthStateMachine._MAX_TRANSITION_BUFFER
    for i in range(cap + 5):  # well past the cap
        add(sm, 'Rejected', 100 + i)

    # After hard cap, should have been force-resolved
    # evidence=1 < transition_confirm=3, so absorbed to Brown
    assert sm.state == RunLengthStateMachine.CONFIRMED_BATCH, (
        f"Expected CONFIRMED_BATCH after hard cap, got {sm.state}"
    )
    assert sm.confirmed_batch_class == 'Brown', (
        f"Expected Brown (absorbed) after hard cap, got {sm.confirmed_batch_class}"
    )
    print("PASS test_rejected_only_stream_hits_hard_cap")


# ── test 18: hard cap with enough evidence confirms transition ────────────────

def test_hard_cap_with_enough_evidence_confirms():
    """
    If the hard cap is hit but there IS enough candidate evidence,
    the transition should be confirmed rather than absorbed.
    """
    sm = make_sm(min_run=3, max_blip=2, transition_confirm=3)
    add_many(sm, 'Brown', 4, start_id=1)

    # Start transition: 2 Red items (not enough yet)
    add(sm, 'Red', 10)   # evidence=1
    add(sm, 'Red', 11)   # evidence=2

    # Lots of Rejected items, then a 3rd Red to reach threshold before cap
    for i in range(10):
        add(sm, 'Rejected', 100 + i)
    add(sm, 'Red', 12)   # evidence=3 → should confirm on this item, before cap

    assert sm.confirmed_batch_class == 'Red', (
        f"Expected Red after evidence threshold, got {sm.confirmed_batch_class}"
    )
    print("PASS test_hard_cap_with_enough_evidence_confirms")


# ── test 19: transition with rejected doesn't break run_length property ───────

def test_transition_run_length_reflects_evidence():
    """
    current_run_length during TRANSITION should reflect candidate evidence
    count, not total buffer size (which includes neutral Rejected items).
    """
    sm = make_sm(min_run=3, max_blip=2, transition_confirm=5)
    add_many(sm, 'Brown', 4, start_id=1)

    add(sm, 'Red', 10)        # evidence=1
    add(sm, 'Rejected', 11)   # neutral (buf=2, evidence=1)
    add(sm, 'Red', 12)        # evidence=2 (buf=3, evidence=2)
    add(sm, 'Rejected', 13)   # neutral (buf=4, evidence=2)

    assert sm.state == RunLengthStateMachine.TRANSITION
    assert sm.current_run_length == 2, (
        f"current_run_length should be evidence count (2), "
        f"got {sm.current_run_length}"
    )
    assert sm.current_run_class == 'Red'
    print("PASS test_transition_run_length_reflects_evidence")


# ── test 20: realistic conveyor scenario ──────────────────────────────────────

def test_realistic_conveyor_batch_change():
    """
    Simulate a realistic conveyor scenario:
    - 10 Brown bags, then a batch change to Red
    - At the boundary: 2 flipped (Rejected) bags
    - Then Red bags continue

    The Rejected bags at the boundary should NOT prevent the transition.
    """
    sm = make_sm(min_run=5, max_blip=3, transition_confirm=5)

    # Brown batch
    for i in range(10):
        add(sm, 'Brown', i)
    assert sm.confirmed_batch_class == 'Brown'

    # Boundary: operator switching batches, a couple bags are flipped
    add(sm, 'Rejected', 100)  # still in CONFIRMED_BATCH → smoothed to Brown
    add(sm, 'Rejected', 101)  # still in CONFIRMED_BATCH → smoothed to Brown

    # New batch starts
    add(sm, 'Red', 200)       # TRANSITION starts
    add(sm, 'Rejected', 201)  # neutral during transition
    add(sm, 'Red', 202)
    add(sm, 'Red', 203)
    add(sm, 'Rejected', 204)  # neutral during transition
    add(sm, 'Red', 205)
    add(sm, 'Red', 206)       # evidence=5 → transition confirmed

    assert sm.confirmed_batch_class == 'Red', (
        f"Expected Red after realistic batch change, got {sm.confirmed_batch_class}"
    )

    # Drain and verify all items are counted
    for i in range(20):
        add(sm, 'Red', 300 + i)

    # Check that T201 and T204 (Rejected during transition) became Red
    boundary_rejected = [
        r for r in sm.confirmed_records
        if r.original_class == 'Rejected' and r.track_id in (201, 204)
    ]
    for r in boundary_rejected:
        assert r.class_name == 'Red', (
            f"Boundary Rejected T{r.track_id} should be Red, got {r.class_name}"
        )

    # Check that T100/T101 (Rejected before transition) became Brown
    pre_boundary_rejected = [
        r for r in sm.confirmed_records
        if r.original_class == 'Rejected' and r.track_id in (100, 101)
    ]
    for r in pre_boundary_rejected:
        assert r.class_name == 'Brown', (
            f"Pre-boundary Rejected T{r.track_id} should be Brown, got {r.class_name}"
        )
    print("PASS test_realistic_conveyor_batch_change")


if __name__ == '__main__':
    test_accumulation_establishes_batch()
    test_pre_run_items_absorbed()
    test_steady_state_confirmation()
    test_rejected_overridden_to_batch()
    test_blip_absorption()
    test_real_transition()
    test_transition_overflow()
    test_flush_remaining()
    test_statistics_interface()
    test_short_accum_run_resets()
    # New tests for neutral-rejected behavior
    test_rejected_neutral_during_confirmed_transition()
    test_rejected_neutral_during_blip_absorption()
    test_interleaved_rejected_and_candidate()
    test_many_rejected_dont_cause_overflow()
    test_third_class_triggers_overflow()
    test_rejected_before_transition_start()
    test_rejected_only_stream_hits_hard_cap()
    test_hard_cap_with_enough_evidence_confirms()
    test_transition_run_length_reflects_evidence()
    test_realistic_conveyor_batch_change()
    print("\nAll RunLengthStateMachine tests passed!")
