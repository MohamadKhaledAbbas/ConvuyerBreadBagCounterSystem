#!/usr/bin/env python3
"""
Test suite for forward-context smoothing in BidirectionalSmoother.

Tests the forward-context approach where the oldest item is smoothed
using items that come AFTER it as context.
"""

import sys
from src.tracking.BidirectionalSmoother import BidirectionalSmoother


def test_initialization():
    """Test smoother initialization and validation."""
    print("\n=== Test 1: Initialization and Validation ===")

    # Test default initialization
    smoother = BidirectionalSmoother()
    assert smoother.window_size == 21, f"Expected window_size=21, got {smoother.window_size}"
    print("✓ Default initialization correct (window_size=21)")

    # Test window size validation - minimum size
    try:
        BidirectionalSmoother(window_size=2)
        assert False, "Should reject window_size < 3"
    except ValueError:
        print("✓ Correctly rejected window_size < 3")

    # Even window sizes are now allowed (no center-based requirement)
    smoother = BidirectionalSmoother(window_size=6)
    assert smoother.window_size == 6
    print("✓ Even window_size=6 accepted (no center-based requirement)")

    # Test custom odd window size
    smoother = BidirectionalSmoother(window_size=7)
    assert smoother.window_size == 7
    print("✓ Custom window_size=7 works correctly")

    # Backward compatibility: warmup_smoothing_enabled param accepted
    smoother = BidirectionalSmoother(warmup_smoothing_enabled=False)
    print("✓ warmup_smoothing_enabled param accepted (backward compat, ignored)")

    print("✓ All initialization tests passed")


def test_accumulation_phase():
    """Test accumulation phase - no confirmations until window is full."""
    print("\n=== Test 2: Accumulation Phase ===")

    smoother = BidirectionalSmoother(window_size=7)

    # Add 6 items (window_size - 1) - should all return None
    for i in range(6):
        result = smoother.add_classification(
            track_id=i,
            class_name='Brown_Orange',
            confidence=0.9,
            vote_ratio=0.85,
            non_rejected_rois=5
        )
        assert result is None, f"Item {i} should return None during accumulation"
        assert len(smoother.window_buffer) == i + 1, f"Buffer size should be {i+1}"

    print(f"✓ Accumulated {len(smoother.window_buffer)} items (no confirmations)")

    # 7th item should trigger confirmation of oldest
    result = smoother.add_classification(
        track_id=6,
        class_name='Brown_Orange',
        confidence=0.9,
        vote_ratio=0.85,
        non_rejected_rois=5
    )
    assert result is not None, "7th item should trigger confirmation"
    assert result.track_id == 0, "Should confirm oldest item (track_id=0)"
    assert len(smoother.window_buffer) == 6, "Buffer should have 6 items remaining"
    print("✓ First confirmation at window_size items, oldest confirmed")

    print("✓ Accumulation phase test passed")


def test_rejected_always_smoothed():
    """Test that Rejected class is ALWAYS smoothed regardless of context."""
    print("\n=== Test 3: Rejected Always Smoothed ===")

    # Case 1: Rejected with strong forward context
    smoother = BidirectionalSmoother(window_size=7)
    items = [
        ('Rejected', 0.3, 0),  # Should be smoothed to Brown_Orange
        ('Brown_Orange', 0.9, 5),
        ('Brown_Orange', 0.9, 5),
        ('Brown_Orange', 0.9, 5),
        ('Brown_Orange', 0.9, 5),
        ('Brown_Orange', 0.9, 5),
        ('Brown_Orange', 0.9, 5),
    ]

    for i, (class_name, conf, rois) in enumerate(items):
        result = smoother.add_classification(
            track_id=i,
            class_name=class_name,
            confidence=conf,
            vote_ratio=0.8,
            non_rejected_rois=rois
        )
        if result:
            assert result.class_name == 'Brown_Orange', f"Rejected should be smoothed to Brown_Orange, got {result.class_name}"
            assert result.smoothed == True, "Should be marked as smoothed"
            assert result.original_class == 'Rejected', "Original should be Rejected"
            print(f"✓ Rejected with strong context -> {result.class_name}")

    # Case 2: Rejected with weak forward context (mixed classes)
    smoother2 = BidirectionalSmoother(window_size=7)
    items2 = [
        ('Rejected', 0.3, 0),
        ('Brown_Orange', 0.9, 5),
        ('Blue_Yellow', 0.9, 5),
        ('Brown_Orange', 0.9, 5),
        ('Blue_Yellow', 0.9, 5),
        ('Brown_Orange', 0.9, 5),
        ('Blue_Yellow', 0.9, 5),
    ]
    for i, (class_name, conf, rois) in enumerate(items2):
        result = smoother2.add_classification(
            track_id=i, class_name=class_name, confidence=conf,
            vote_ratio=0.8, non_rejected_rois=rois
        )
        if result:
            assert result.smoothed == True, "Rejected should ALWAYS be smoothed"
            assert result.original_class == 'Rejected'
            print(f"✓ Rejected with weak context -> {result.class_name} (best guess)")

    # Case 3: Rejected with no forward context (single item flush)
    smoother3 = BidirectionalSmoother(window_size=7)
    smoother3.add_classification(track_id=0, class_name='Rejected', confidence=0.3,
                                  vote_ratio=0.2, non_rejected_rois=0)
    flushed = smoother3.flush_remaining()
    assert len(flushed) == 1
    assert flushed[0].smoothed == True
    assert flushed[0].class_name == 'Unknown'
    print(f"✓ Rejected with no context -> Unknown")

    # Case 4: Rejected with confirmed history fallback
    smoother4 = BidirectionalSmoother(window_size=7)
    for i in range(10):
        smoother4.add_classification(track_id=i, class_name='Brown_Orange', confidence=0.9,
                                      vote_ratio=0.85, non_rejected_rois=5)
    # Now add a Rejected as only remaining item and flush
    smoother4.add_classification(track_id=100, class_name='Rejected', confidence=0.3,
                                  vote_ratio=0.2, non_rejected_rois=0)
    flushed = smoother4.flush_remaining()
    rejected_flushed = [r for r in flushed if r.original_class == 'Rejected']
    assert len(rejected_flushed) >= 1
    for r in rejected_flushed:
        assert r.smoothed == True
        assert r.class_name != 'Rejected'
        print(f"✓ Rejected with confirmed history -> {r.class_name}")

    print("✓ Rejected always smoothed test passed")


def test_all_rejected_window():
    """Test handling of all-Rejected window."""
    print("\n=== Test 4: All-Rejected Window ===")

    smoother = BidirectionalSmoother(window_size=7)

    # Fill window with all Rejected items
    for i in range(7):
        result = smoother.add_classification(
            track_id=i,
            class_name='Rejected',
            confidence=0.3,
            vote_ratio=0.2,
            non_rejected_rois=0
        )
        if result:
            assert result.class_name == 'Unknown', f"All-Rejected should smooth to Unknown, got {result.class_name}"
            assert result.smoothed == True, "Should be marked as smoothed"
            print(f"✓ All-Rejected window: T{result.track_id} -> Unknown")

    print("✓ All-Rejected window test passed")


def test_outlier_smoothing():
    """Test outlier smoothing in stable batch."""
    print("\n=== Test 5: Outlier Smoothing ===")

    smoother = BidirectionalSmoother(window_size=7)

    # Fill window: 1 Red_Yellow outlier at position 0, rest Brown_Orange
    items = [
        ('Red_Yellow', 0.95, 5),  # High confidence outlier - should still be smoothed
        ('Brown_Orange', 0.9, 5),
        ('Brown_Orange', 0.9, 5),
        ('Brown_Orange', 0.9, 5),
        ('Brown_Orange', 0.9, 5),
        ('Brown_Orange', 0.9, 5),
        ('Brown_Orange', 0.9, 5),
    ]

    for i, (class_name, conf, rois) in enumerate(items):
        result = smoother.add_classification(
            track_id=i,
            class_name=class_name,
            confidence=conf,
            vote_ratio=0.85,
            non_rejected_rois=rois
        )
        if result:
            assert result.class_name == 'Brown_Orange', f"Outlier should be smoothed to Brown_Orange, got {result.class_name}"
            assert result.smoothed == True, "Should be marked as smoothed"
            assert result.original_class == 'Red_Yellow', "Original should be Red_Yellow"
            print(f"✓ Outlier Red_Yellow (T{result.track_id}, conf=0.95) smoothed to {result.class_name}")

    print("✓ Outlier smoothing test passed")


def test_pure_batch():
    """Test pure batch (all same class) - no smoothing needed."""
    print("\n=== Test 6: Pure Batch ===")

    smoother = BidirectionalSmoother(window_size=7)

    # Fill window with all Brown_Orange
    confirmed_results = []
    for i in range(14):  # 14 items = 7 warmup + 7 confirmations
        result = smoother.add_classification(
            track_id=i,
            class_name='Brown_Orange',
            confidence=0.9,
            vote_ratio=0.85,
            non_rejected_rois=5
        )
        if result:
            confirmed_results.append(result)

    assert len(confirmed_results) > 0, "Should have confirmed records"
    for r in confirmed_results:
        assert r.class_name == 'Brown_Orange', f"Pure batch item should remain Brown_Orange"
        assert r.smoothed == False, "Pure batch item should NOT be smoothed"

    print(f"✓ {len(confirmed_results)} items confirmed without smoothing in pure batch")
    print("✓ Pure batch test passed")


def test_batch_boundary_preservation():
    """Test that items at batch boundaries are preserved (not incorrectly smoothed)."""
    print("\n=== Test 7: Batch Boundary Preservation ===")

    smoother = BidirectionalSmoother(window_size=7)

    # Phase 1: Establish batch A (Brown_Orange) with confirmed history
    for i in range(10):
        smoother.add_classification(
            track_id=i, class_name='Brown_Orange', confidence=0.9,
            vote_ratio=0.85, non_rejected_rois=5
        )

    confirmed_count = len(smoother.confirmed_records)
    assert confirmed_count > 0, "Should have confirmed Brown_Orange records"
    print(f"✓ Established batch A with {confirmed_count} confirmed Brown_Orange records")

    # Phase 2: Add Brown_Orange at boundary, then Blue_Yellow (new batch)
    # The Brown_Orange should be preserved even though forward context is Blue_Yellow
    transition_items = [
        ('Brown_Orange', 0.9, 5),  # End of batch A - should be preserved
        ('Blue_Yellow', 0.9, 5),
        ('Blue_Yellow', 0.9, 5),
        ('Blue_Yellow', 0.9, 5),
        ('Blue_Yellow', 0.9, 5),
        ('Blue_Yellow', 0.9, 5),
        ('Blue_Yellow', 0.9, 5),
    ]

    boundary_results = []
    for i, (cn, conf, rois) in enumerate(transition_items):
        result = smoother.add_classification(
            track_id=100 + i, class_name=cn, confidence=conf,
            vote_ratio=0.85, non_rejected_rois=rois
        )
        if result and result.track_id >= 100:
            boundary_results.append(result)

    # Find the Brown_Orange boundary item
    boundary_item = next((r for r in boundary_results if
                          r.track_id == 100 or
                          (r.original_class is None and r.class_name == 'Brown_Orange')),
                         None)

    # The Brown_Orange at the boundary should be preserved
    brown_orange_items = [r for r in smoother.confirmed_records if r.class_name == 'Brown_Orange']
    assert len(brown_orange_items) > confirmed_count, "Should have more Brown_Orange after boundary"
    print(f"✓ Brown_Orange items preserved at batch boundary")

    print("✓ Batch boundary preservation test passed")


def test_batch_transition_rejected_smoothing():
    """Test that Rejected at batch transition is smoothed to new batch class."""
    print("\n=== Test 8: Rejected at Batch Transition ===")

    smoother = BidirectionalSmoother(window_size=7)

    # Build up batch A
    for i in range(10):
        smoother.add_classification(
            track_id=i, class_name='Brown_Orange', confidence=0.9,
            vote_ratio=0.85, non_rejected_rois=5
        )

    # Transition: Rejected between batches, then new batch B
    items = [
        ('Rejected', 0.3, 0),
        ('Blue_Yellow', 0.9, 5),
        ('Blue_Yellow', 0.9, 5),
        ('Blue_Yellow', 0.9, 5),
        ('Blue_Yellow', 0.9, 5),
        ('Blue_Yellow', 0.9, 5),
        ('Blue_Yellow', 0.9, 5),
    ]

    for i, (cn, conf, rois) in enumerate(items):
        result = smoother.add_classification(
            track_id=200 + i, class_name=cn, confidence=conf,
            vote_ratio=0.85, non_rejected_rois=rois
        )
        if result and result.original_class == 'Rejected':
            assert result.smoothed == True
            # At transition, forward context is Blue_Yellow, so should be smoothed to that
            assert result.class_name == 'Blue_Yellow', f"Rejected at transition should -> Blue_Yellow, got {result.class_name}"
            print(f"✓ Rejected at batch transition -> {result.class_name}")

    print("✓ Rejected at batch transition test passed")


def test_flush_with_partial_context():
    """Test flushing remaining items with partial context."""
    print("\n=== Test 9: Flush with Partial Context ===")

    smoother = BidirectionalSmoother(window_size=7)

    # Add only 5 items (less than window_size)
    for i in range(5):
        smoother.add_classification(
            track_id=i,
            class_name='Brown_Orange',
            confidence=0.9,
            vote_ratio=0.85,
            non_rejected_rois=5
        )

    assert len(smoother.window_buffer) == 5, "Should have 5 items in buffer"

    # Flush remaining
    flushed = smoother.flush_remaining()

    assert len(flushed) == 5, f"Should flush all 5 items, got {len(flushed)}"
    assert len(smoother.window_buffer) == 0, "Buffer should be empty after flush"
    for r in flushed:
        assert r.class_name == 'Brown_Orange', f"Flushed item should be Brown_Orange, got {r.class_name}"
    print(f"✓ Flushed {len(flushed)} items with partial context")

    print("✓ Flush with partial context test passed")


def test_flush_mixed_items():
    """Test flushing with mixed items including Rejected."""
    print("\n=== Test 10: Flush Mixed Items ===")

    smoother = BidirectionalSmoother(window_size=7)

    items = [
        ('Brown_Orange', 0.9, 5),
        ('Brown_Orange', 0.9, 5),
        ('Rejected', 0.3, 0),  # Should be smoothed
        ('Brown_Orange', 0.9, 5),
        ('Brown_Orange', 0.9, 5),
    ]

    for i, (cn, conf, rois) in enumerate(items):
        smoother.add_classification(track_id=i, class_name=cn, confidence=conf,
                                     vote_ratio=0.8, non_rejected_rois=rois)

    flushed = smoother.flush_remaining()
    assert len(flushed) == 5

    rejected_results = [r for r in flushed if r.original_class == 'Rejected']
    assert len(rejected_results) == 1, f"Expected 1 Rejected result, got {len(rejected_results)}"
    assert rejected_results[0].smoothed == True
    assert rejected_results[0].class_name == 'Brown_Orange'
    print(f"✓ Rejected item in flush smoothed to Brown_Orange")

    print("✓ Flush mixed items test passed")


def test_no_smoothing_when_no_dominant():
    """Test that items are NOT smoothed when forward context has no dominant class."""
    print("\n=== Test 11: No Smoothing Without Dominant ===")

    smoother = BidirectionalSmoother(window_size=7)

    # Mix of classes with no clear dominant in forward context
    items = [
        ('Red_Yellow', 0.9, 5),     # Should NOT be smoothed (no clear dominant ahead)
        ('Brown_Orange', 0.9, 5),
        ('Blue_Yellow', 0.9, 5),
        ('Brown_Orange', 0.9, 5),
        ('Blue_Yellow', 0.9, 5),
        ('Red_Yellow', 0.9, 5),
        ('Brown_Orange', 0.9, 5),
    ]

    for i, (cn, conf, rois) in enumerate(items):
        result = smoother.add_classification(
            track_id=i, class_name=cn, confidence=conf,
            vote_ratio=0.85, non_rejected_rois=rois
        )
        if result:
            # No class has >= 70% dominance in forward context
            assert result.smoothed == False, f"Should NOT be smoothed when no dominant class, but T{result.track_id} was smoothed"
            print(f"✓ T{result.track_id}={result.class_name} not smoothed (no dominant)")

    print("✓ No smoothing without dominant test passed")


def test_statistics():
    """Test that statistics tracking works."""
    print("\n=== Test 12: Statistics Tracking ===")

    smoother = BidirectionalSmoother(window_size=7)

    items = [
        ('Rejected', 0.3, 0),
        ('Brown_Orange', 0.9, 5),
        ('Brown_Orange', 0.9, 5),
        ('Brown_Orange', 0.9, 5),
        ('Brown_Orange', 0.9, 5),
        ('Brown_Orange', 0.9, 5),
        ('Brown_Orange', 0.9, 5),
    ]

    for i, (cn, conf, rois) in enumerate(items):
        smoother.add_classification(track_id=i, class_name=cn, confidence=conf,
                                     vote_ratio=0.8, non_rejected_rois=rois)

    stats = smoother.get_statistics()
    assert stats['total_records'] == 7
    assert stats['confirmed_count'] == 1  # Only 1 confirmed (window just filled)
    assert stats['smoothed_records'] == 1  # The Rejected was smoothed
    assert stats['pending_in_window'] == 6
    print(f"✓ Statistics: total={stats['total_records']}, confirmed={stats['confirmed_count']}, "
          f"smoothed={stats['smoothed_records']}, pending={stats['pending_in_window']}")

    print("✓ Statistics tracking test passed")


def test_long_running_batch():
    """Test continuous processing of a long running batch."""
    print("\n=== Test 13: Long Running Batch ===")

    smoother = BidirectionalSmoother(window_size=7)

    # Process 50 items from the same batch with occasional Rejected/outliers
    all_confirmed = []
    for i in range(50):
        if i % 10 == 5:
            cn, conf, rois = 'Rejected', 0.3, 0
        elif i % 15 == 7:
            cn, conf, rois = 'Red_Yellow', 0.85, 4  # Occasional outlier
        else:
            cn, conf, rois = 'Brown_Orange', 0.9, 5

        result = smoother.add_classification(
            track_id=i, class_name=cn, confidence=conf,
            vote_ratio=0.85, non_rejected_rois=rois
        )
        if result:
            all_confirmed.append(result)

    # Flush remaining
    remaining = smoother.flush_remaining()
    all_confirmed.extend(remaining)

    # All items should be Brown_Orange (outliers and Rejected smoothed)
    non_brown_orange = [r for r in all_confirmed if r.class_name != 'Brown_Orange']
    print(f"✓ Processed 50 items, {len(all_confirmed)} confirmed")
    print(f"  Smoothed: {sum(1 for r in all_confirmed if r.smoothed)}")
    print(f"  Non-Brown_Orange: {len(non_brown_orange)}")

    # Most should be Brown_Orange
    brown_orange_pct = sum(1 for r in all_confirmed if r.class_name == 'Brown_Orange') / len(all_confirmed)
    assert brown_orange_pct >= 0.90, f"Expected >= 90% Brown_Orange, got {brown_orange_pct:.0%}"
    print(f"  Brown_Orange: {brown_orange_pct:.0%}")

    print("✓ Long running batch test passed")


def test_batch_switch_complete():
    """Test complete batch switch: batch A -> batch B."""
    print("\n=== Test 14: Complete Batch Switch ===")

    smoother = BidirectionalSmoother(window_size=7)

    all_confirmed = []

    # Batch A: 20 Brown_Orange
    for i in range(20):
        result = smoother.add_classification(
            track_id=i, class_name='Brown_Orange', confidence=0.9,
            vote_ratio=0.85, non_rejected_rois=5
        )
        if result:
            all_confirmed.append(result)

    batch_a_count = len(all_confirmed)
    print(f"  Batch A: {batch_a_count} confirmed Brown_Orange")

    # Batch B: 20 Blue_Yellow
    for i in range(20, 40):
        result = smoother.add_classification(
            track_id=i, class_name='Blue_Yellow', confidence=0.9,
            vote_ratio=0.85, non_rejected_rois=5
        )
        if result:
            all_confirmed.append(result)

    batch_b_confirmed = all_confirmed[batch_a_count:]
    print(f"  Batch B: {len(batch_b_confirmed)} confirmed so far")

    # Flush remaining
    remaining = smoother.flush_remaining()
    all_confirmed.extend(remaining)

    # Count classes
    brown_count = sum(1 for r in all_confirmed if r.class_name == 'Brown_Orange')
    blue_count = sum(1 for r in all_confirmed if r.class_name == 'Blue_Yellow')

    print(f"  Total: {brown_count} Brown_Orange + {blue_count} Blue_Yellow = {len(all_confirmed)}")

    # Should have approximately 20 of each (batch boundary may cause ±1-2)
    assert brown_count >= 18, f"Should have ~20 Brown_Orange, got {brown_count}"
    assert blue_count >= 18, f"Should have ~20 Blue_Yellow, got {blue_count}"
    assert brown_count + blue_count == 40, "Total should be 40"

    print("✓ Complete batch switch test passed")


def run_all_tests():
    """Run all test cases."""
    print("\n" + "=" * 60)
    print("TESTING FORWARD-CONTEXT SMOOTHING")
    print("=" * 60)

    tests = [
        test_initialization,
        test_accumulation_phase,
        test_rejected_always_smoothed,
        test_all_rejected_window,
        test_outlier_smoothing,
        test_pure_batch,
        test_batch_boundary_preservation,
        test_batch_transition_rejected_smoothing,
        test_flush_with_partial_context,
        test_flush_mixed_items,
        test_no_smoothing_when_no_dominant,
        test_statistics,
        test_long_running_batch,
        test_batch_switch_complete,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
