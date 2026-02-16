#!/usr/bin/env python3
"""
Test suite for BidirectionalSmoother (originally tested center-based smoothing,
now updated to test the forward-context replacement).

This file preserves the original test structure and scenarios while
adapting to the forward-context smoothing implementation.
"""

import sys
from src.tracking.BidirectionalSmoother import BidirectionalSmoother


def test_initialization():
    """Test smoother initialization and validation."""
    print("\n=== Test 1: Initialization and Validation ===")

    # Test default initialization
    smoother = BidirectionalSmoother()
    assert smoother.window_size == 21, f"Expected window_size=21, got {smoother.window_size}"
    print("✓ Default initialization correct")

    # Test window size validation
    try:
        BidirectionalSmoother(window_size=2)
        assert False, "Should reject window_size < 3"
    except ValueError:
        print("✓ Correctly rejected window_size < 3")

    # Even window sizes are allowed (no center-based requirement)
    smoother = BidirectionalSmoother(window_size=6)
    assert smoother.window_size == 6
    print("✓ Even window_size=6 accepted")

    # Test custom window size
    smoother = BidirectionalSmoother(window_size=7)
    assert smoother.window_size == 7
    print("✓ Custom window_size=7 works correctly")

    print("✓ All initialization tests passed")


def test_warmup_phase():
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

    print(f"✓ Accumulated {len(smoother.window_buffer)} items during accumulation (no confirmations)")

    # 7th item should trigger confirmation
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


def test_warmup_smoothing():
    """Test that Rejected items in the buffer are properly smoothed on confirmation."""
    print("\n=== Test 3: Rejected Smoothing ===")

    smoother = BidirectionalSmoother(window_size=7)

    # Add items with some Rejected - they will be smoothed when confirmed
    items = [
        ('Rejected', 0.3, 0),     # Position 0 - will be smoothed when confirmed
        ('Brown_Orange', 0.9, 5),
        ('Brown_Orange', 0.9, 5),
        ('Brown_Orange', 0.9, 5),
        ('Brown_Orange', 0.9, 5),
        ('Brown_Orange', 0.9, 5),
        ('Brown_Orange', 0.9, 5),  # Triggers confirmation of position 0
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
            assert result.class_name == 'Brown_Orange', f"Expected Brown_Orange, got {result.class_name}"
            assert result.smoothed == True
            assert result.original_class == 'Rejected'
            print(f"✓ Rejected item smoothed to Brown_Orange on confirmation")

    print("✓ Rejected smoothing test passed")


def test_center_based_analysis():
    """Test forward-context analysis (replaces old center-based analysis)."""
    print("\n=== Test 4: Forward-Context Analysis ===")

    smoother = BidirectionalSmoother(window_size=7)

    # Fill window with items for batch transition
    items = [
        ('Blue_Yellow', 0.9, 5),
        ('Blue_Yellow', 0.9, 5),
        ('Blue_Yellow', 0.9, 5),
        ('Blue_Yellow', 0.9, 5),
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
            # Forward context of oldest (Blue_Yellow): 3 Blue_Yellow + 3 Brown_Orange
            # No clear dominant (50/50 split) - should NOT be smoothed
            assert result.class_name == 'Blue_Yellow', f"Should preserve Blue_Yellow at transition, got {result.class_name}"
            assert result.smoothed == False, "Should not be smoothed during transition"
            print(f"✓ Blue_Yellow preserved at batch transition (no dominant in forward context)")

    print("✓ Forward-context analysis test passed")


def test_rejected_smoothing():
    """Test that Rejected class is always smoothed."""
    print("\n=== Test 5: Rejected Smoothing ===")

    smoother = BidirectionalSmoother(window_size=7)

    # Fill window: 1 Rejected at position 0, rest Brown_Orange
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
            print(f"✓ Rejected (track_id={result.track_id}) smoothed to {result.class_name}")

    print("✓ Rejected smoothing test passed")


def test_outlier_smoothing():
    """Test outlier smoothing in stable batch."""
    print("\n=== Test 6: Outlier Smoothing ===")

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
            print(f"✓ Outlier Red_Yellow (track_id={result.track_id}) smoothed to {result.class_name}")

    print("✓ Outlier smoothing test passed")


def test_batch_transition_preservation():
    """Test that valid batch transitions are preserved."""
    print("\n=== Test 7: Batch Transition Preservation ===")

    smoother = BidirectionalSmoother(window_size=7)

    # Create transition: first 4 Blue_Yellow, then 3 Brown_Orange
    items = [
        ('Blue_Yellow', 0.9, 5),
        ('Blue_Yellow', 0.9, 5),
        ('Blue_Yellow', 0.9, 5),
        ('Blue_Yellow', 0.9, 5),
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
            # First confirmation (track_id=0) - Blue_Yellow
            # Forward context is mixed (no clear dominant) - should NOT be smoothed
            assert result.class_name == 'Blue_Yellow', f"Transition item should be preserved, got {result.class_name}"
            assert result.smoothed == False, "Transition item should NOT be smoothed"
            print(f"✓ Transition item {result.class_name} (track_id={result.track_id}) preserved (not smoothed)")

    print("✓ Batch transition preservation test passed")


def test_flush_with_partial_context():
    """Test flushing remaining items with partial context."""
    print("\n=== Test 8: Flush with Partial Context ===")

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
    print(f"✓ Flushed {len(flushed)} items with partial context")

    print("✓ Flush with partial context test passed")


def test_all_rejected_window():
    """Test handling of all-Rejected window."""
    print("\n=== Test 9: All-Rejected Window ===")

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
            print(f"✓ All-Rejected window: smoothed to Unknown (track_id={result.track_id})")

    print("✓ All-Rejected window test passed")


def test_pure_batch():
    """Test pure batch (all same class)."""
    print("\n=== Test 10: Pure Batch ===")

    smoother = BidirectionalSmoother(window_size=7)

    # Fill window with all Brown_Orange
    for i in range(7):
        result = smoother.add_classification(
            track_id=i,
            class_name='Brown_Orange',
            confidence=0.9,
            vote_ratio=0.85,
            non_rejected_rois=5
        )
        if result:
            assert result.class_name == 'Brown_Orange', f"Pure batch item should remain Brown_Orange"
            assert result.smoothed == False, "Pure batch item should NOT be smoothed"
            print(f"✓ Pure batch item preserved (track_id={result.track_id}, smoothed={result.smoothed})")

    print("✓ Pure batch test passed")


def run_all_tests():
    """Run all test cases."""
    print("\n" + "=" * 60)
    print("TESTING FORWARD-CONTEXT SMOOTHING")
    print("=" * 60)

    tests = [
        test_initialization,
        test_warmup_phase,
        test_warmup_smoothing,
        test_center_based_analysis,
        test_rejected_smoothing,
        test_outlier_smoothing,
        test_batch_transition_preservation,
        test_flush_with_partial_context,
        test_all_rejected_window,
        test_pure_batch,
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
