#!/usr/bin/env python3
"""
Test suite for center-based context smoothing with analyze-center-confirm-oldest strategy.

This tests the new implementation in BidirectionalSmoother.py
"""

import sys
from src.tracking.BidirectionalSmoother import BidirectionalSmoother


def test_initialization():
    """Test smoother initialization and validation."""
    print("\n=== Test 1: Initialization and Validation ===")
    
    # Test default initialization
    smoother = BidirectionalSmoother()
    assert smoother.window_size == 21, f"Expected window_size=21, got {smoother.window_size}"
    assert smoother.center_index == 10, f"Expected center_index=10, got {smoother.center_index}"
    assert smoother.warmup_smoothing_enabled, "Expected warmup_smoothing_enabled=True"
    print("✓ Default initialization correct")
    
    # Test window size validation
    try:
        BidirectionalSmoother(window_size=2)
        assert False, "Should reject window_size < 3"
    except ValueError:
        print("✓ Correctly rejected window_size < 3")
    
    try:
        BidirectionalSmoother(window_size=4)
        assert False, "Should reject even window_size"
    except ValueError:
        print("✓ Correctly rejected even window_size")
    
    # Test custom window size
    smoother = BidirectionalSmoother(window_size=7)
    assert smoother.center_index == 3, f"Expected center_index=3 for window_size=7, got {smoother.center_index}"
    print("✓ Custom window_size=7 works correctly")
    
    print("✓ All initialization tests passed")


def test_warmup_phase():
    """Test warmup phase accumulation."""
    print("\n=== Test 2: Warmup Phase ===")
    
    smoother = BidirectionalSmoother(window_size=7, warmup_smoothing_enabled=False)
    
    # Add 6 items (window_size - 1) - should all return None
    for i in range(6):
        result = smoother.add_classification(
            track_id=i,
            class_name='Brown_Orange',
            confidence=0.9,
            vote_ratio=0.85,
            non_rejected_rois=5
        )
        assert result is None, f"Item {i} should return None during warmup"
        assert len(smoother.window_buffer) == i + 1, f"Buffer size should be {i+1}"
    
    print(f"✓ Accumulated {len(smoother.window_buffer)} items during warmup (no confirmations)")
    
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
    
    print("✓ Warmup phase test passed")


def test_warmup_smoothing():
    """Test warmup smoothing of Rejected and outliers."""
    print("\n=== Test 3: Warmup Smoothing ===")
    
    smoother = BidirectionalSmoother(window_size=7, warmup_smoothing_enabled=True)
    
    # Add items with some Rejected - should be smoothed in-place during warmup
    items = [
        ('Brown_Orange', 0.9, 5),
        ('Brown_Orange', 0.9, 5),
        ('Rejected', 0.3, 0),  # Should be smoothed to Brown_Orange
        ('Brown_Orange', 0.9, 5),
        ('Brown_Orange', 0.9, 5),
        ('Rejected', 0.2, 0),  # Should be smoothed to Brown_Orange
    ]
    
    for i, (class_name, conf, rois) in enumerate(items):
        result = smoother.add_classification(
            track_id=i,
            class_name=class_name,
            confidence=conf,
            vote_ratio=0.8,
            non_rejected_rois=rois
        )
        assert result is None, f"Should return None during warmup (item {i})"
    
    # Check that Rejected items were smoothed in buffer
    rejected_count = sum(1 for r in smoother.window_buffer if r.class_name == 'Rejected')
    assert rejected_count == 0, f"Expected 0 Rejected in buffer, got {rejected_count}"
    print(f"✓ Warmup smoothing cleaned {2} Rejected items from buffer")
    
    print("✓ Warmup smoothing test passed")


def test_center_based_analysis():
    """Test center-based context analysis."""
    print("\n=== Test 4: Center-Based Analysis ===")
    
    smoother = BidirectionalSmoother(window_size=7, warmup_smoothing_enabled=False)
    
    # Fill window with items for batch transition
    # Items 0-2: Blue_Yellow (past)
    # Item 3: center
    # Items 4-6: Brown_Orange (future)
    items = [
        ('Blue_Yellow', 0.9, 5),
        ('Blue_Yellow', 0.9, 5),
        ('Blue_Yellow', 0.9, 5),
        ('Blue_Yellow', 0.9, 5),  # center
        ('Brown_Orange', 0.9, 5),
        ('Brown_Orange', 0.9, 5),
        ('Brown_Orange', 0.9, 5),
    ]
    
    for i, (class_name, conf, rois) in enumerate(items):
        smoother.add_classification(
            track_id=i,
            class_name=class_name,
            confidence=conf,
            vote_ratio=0.85,
            non_rejected_rois=rois
        )
    
    # Analyze the context
    context = smoother._analyze_batch_context()
    print(f"  Dominant class: {context['dominant_class']}")
    print(f"  Dominance ratio: {context['dominance_ratio']:.2f}")
    print(f"  In transition: {context['in_transition']}")
    print(f"  Past dominant: {context['past_dominant']}")
    print(f"  Future dominant: {context['future_dominant']}")
    
    # Should detect transition
    assert context['in_transition'], "Should detect transition"
    assert context['past_dominant'] == 'Blue_Yellow', "Past should be Blue_Yellow"
    assert context['future_dominant'] == 'Brown_Orange', "Future should be Brown_Orange"
    
    print("✓ Center-based analysis correctly detected batch transition")
    print("✓ Center-based analysis test passed")


def test_rejected_smoothing():
    """Test that Rejected class is always smoothed."""
    print("\n=== Test 5: Rejected Smoothing ===")
    
    smoother = BidirectionalSmoother(window_size=7, warmup_smoothing_enabled=False)
    
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
            # Should be first confirmation (track_id=0, was Rejected)
            assert result.class_name == 'Brown_Orange', f"Rejected should be smoothed to Brown_Orange, got {result.class_name}"
            assert result.smoothed == True, "Should be marked as smoothed"
            assert result.original_class == 'Rejected', "Original should be Rejected"
            print(f"✓ Rejected (track_id={result.track_id}) smoothed to {result.class_name}")
    
    print("✓ Rejected smoothing test passed")


def test_outlier_smoothing():
    """Test outlier smoothing in stable batch."""
    print("\n=== Test 6: Outlier Smoothing ===")
    
    smoother = BidirectionalSmoother(window_size=7, warmup_smoothing_enabled=False)
    
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
            # Should smooth outlier to dominant
            assert result.class_name == 'Brown_Orange', f"Outlier should be smoothed to Brown_Orange, got {result.class_name}"
            assert result.smoothed == True, "Should be marked as smoothed"
            assert result.original_class == 'Red_Yellow', "Original should be Red_Yellow"
            print(f"✓ Outlier Red_Yellow (track_id={result.track_id}) smoothed to {result.class_name}")
    
    print("✓ Outlier smoothing test passed")


def test_batch_transition_preservation():
    """Test that valid batch transitions are preserved."""
    print("\n=== Test 7: Batch Transition Preservation ===")
    
    smoother = BidirectionalSmoother(window_size=7, warmup_smoothing_enabled=False)
    
    # Create transition: first 4 Blue_Yellow, then 3 Brown_Orange
    items = [
        ('Blue_Yellow', 0.9, 5),  # Past batch - should NOT be smoothed
        ('Blue_Yellow', 0.9, 5),
        ('Blue_Yellow', 0.9, 5),
        ('Blue_Yellow', 0.9, 5),  # center
        ('Brown_Orange', 0.9, 5),  # Future batch starts
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
            # First confirmation (track_id=0) - Blue_Yellow in past context
            # Should NOT be smoothed during transition
            assert result.class_name == 'Blue_Yellow', f"Transition item should be preserved, got {result.class_name}"
            assert result.smoothed == False, "Transition item should NOT be smoothed"
            print(f"✓ Transition item {result.class_name} (track_id={result.track_id}) preserved (not smoothed)")
    
    print("✓ Batch transition preservation test passed")


def test_flush_with_partial_context():
    """Test flushing remaining items with partial context."""
    print("\n=== Test 8: Flush with Partial Context ===")
    
    smoother = BidirectionalSmoother(window_size=7, warmup_smoothing_enabled=False)
    
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
    
    smoother = BidirectionalSmoother(window_size=7, warmup_smoothing_enabled=False)
    
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
            # Should be smoothed to Unknown (fallback)
            assert result.class_name == 'Unknown', f"All-Rejected should smooth to Unknown, got {result.class_name}"
            assert result.smoothed == True, "Should be marked as smoothed"
            print(f"✓ All-Rejected window: smoothed to Unknown (track_id={result.track_id})")
    
    print("✓ All-Rejected window test passed")


def test_pure_batch():
    """Test pure batch (all same class)."""
    print("\n=== Test 10: Pure Batch ===")
    
    smoother = BidirectionalSmoother(window_size=7, warmup_smoothing_enabled=False)
    
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
            # Should NOT be smoothed - pure batch
            assert result.class_name == 'Brown_Orange', f"Pure batch item should remain Brown_Orange"
            assert result.smoothed == False, "Pure batch item should NOT be smoothed"
            print(f"✓ Pure batch item preserved (track_id={result.track_id}, smoothed={result.smoothed})")
    
    print("✓ Pure batch test passed")


def run_all_tests():
    """Run all test cases."""
    print("\n" + "="*60)
    print("TESTING CENTER-BASED CONTEXT SMOOTHING")
    print("="*60)
    
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
    
    print("\n" + "="*60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
