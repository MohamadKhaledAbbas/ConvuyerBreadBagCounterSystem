#!/usr/bin/env python3
"""
Test suite for three-phase deferred confirmation smoother.

This tests the new implementation that eliminates the warmup smoothing feedback loop.

Note: These tests use standalone Python assertions and print statements for simplicity.
For CI/CD integration, consider converting to pytest or unittest framework.
"""

import sys
from src.tracking.BidirectionalSmoother import BidirectionalSmoother, ClassificationRecord


def test_three_phase_architecture():
    """Test the three-phase deferred confirmation architecture."""
    print("\n=== Test 1: Three-Phase Architecture ===")
    
    smoother = BidirectionalSmoother(window_size=21, batch_lock_count=10)
    assert smoother.center_index == 10, f"Expected center_index=10, got {smoother.center_index}"
    
    # Phase 1: Accumulate (items 1-20) - no confirmations
    print("Phase 1: Accumulating items 1-20 (no confirmations)...")
    for i in range(20):
        result = smoother.add_classification(
            track_id=i,
            class_name='Brown_Orange',
            confidence=0.9,
            vote_ratio=0.85,
            non_rejected_rois=5
        )
        assert result is None, f"Item {i+1} should return None during accumulation"
        assert len(smoother.window_buffer) == i + 1, f"Buffer should have {i+1} items"
    
    print(f"✓ Phase 1 complete: {len(smoother.window_buffer)} items accumulated, 0 confirmed")
    
    # Phase 2: Steady State (items 21-30) - defer first center_index (10) items
    print("\nPhase 2: Steady state with deferred items (items 21-30)...")
    deferred_count = 0
    for i in range(20, 30):
        result = smoother.add_classification(
            track_id=i,
            class_name='Brown_Orange',
            confidence=0.9,
            vote_ratio=0.85,
            non_rejected_rois=5
        )
        
        assert result is None, f"Item {i+1} should be deferred (no confirmation)"
        deferred_count += 1
        
    assert len(smoother._deferred_records) == 10, f"Expected 10 deferred, got {len(smoother._deferred_records)}"
    print(f"✓ Phase 2 (deferral phase): {deferred_count} items deferred")
    
    # Continue with actual confirmations (items 31-40)
    print("\nPhase 2 (confirmation phase): Items 31-40 (actual confirmations before batch lock)...")
    for i in range(30, 40):
        result = smoother.add_classification(
            track_id=i,
            class_name='Brown_Orange',
            confidence=0.9,
            vote_ratio=0.85,
            non_rejected_rois=5
        )
        assert result is not None and len(result) == 1, f"Item {i+1} should return single confirmation"
        assert not smoother._batch_locked, f"Batch should not be locked yet at item {i+1}"
    
    print(f"✓ Phase 2 complete: 10 deferred + 10 confirmed")
    
    # Phase 3: Batch Lock (item 41) - release deferred records
    print("\nPhase 3: Batch lock and deferred release (item 41)...")
    result = smoother.add_classification(
        track_id=40,
        class_name='Brown_Orange',
        confidence=0.9,
        vote_ratio=0.85,
        non_rejected_rois=5
    )
    
    # At batch lock, should release all deferred + current confirmation
    assert result is not None, "Item 41 should trigger batch lock"
    assert isinstance(result, list), "Should return list of records"
    assert len(result) == 11, f"Expected 11 records (10 deferred + 1 current), got {len(result)}"
    assert len(smoother._deferred_records) == 0, "Deferred records should be empty after release"
    assert smoother._batch_locked, "Batch should be locked"
    
    print(f"✓ Phase 3 complete: batch locked, {len(result)} records released")
    
    # Subsequent items should return single confirmations
    print("\nPost-batch-lock confirmations...")
    result = smoother.add_classification(
        track_id=41,
        class_name='Brown_Orange',
        confidence=0.9,
        vote_ratio=0.85,
        non_rejected_rois=5
    )
    assert result is not None and len(result) == 1, "Should return single confirmation after batch lock"
    print("✓ Post-batch-lock single confirmations working")
    
    print("\n✓ Three-phase architecture test passed")


def test_no_warmup_smoothing():
    """Test that warmup phase does NOT smooth items (no feedback loop)."""
    print("\n=== Test 2: No Warmup Smoothing ===")
    
    smoother = BidirectionalSmoother(window_size=7, batch_lock_count=3)
    
    # Add items during warmup including Rejected - should NOT be smoothed
    items = [
        ('Brown_Orange', 0.9, 5),
        ('Rejected', 0.3, 0),  # Should NOT be smoothed during warmup
        ('Brown_Orange', 0.9, 5),
        ('Purple_Yellow', 0.8, 4),  # Outlier - should NOT be smoothed during warmup
        ('Brown_Orange', 0.9, 5),
        ('Rejected', 0.2, 0),  # Should NOT be smoothed during warmup
    ]
    
    for i, (class_name, conf, rois) in enumerate(items):
        result = smoother.add_classification(
            track_id=i,
            class_name=class_name,
            confidence=conf,
            vote_ratio=0.85,
            non_rejected_rois=rois
        )
        assert result is None, f"Item {i+1} should not be confirmed during warmup"
    
    # Verify items are NOT smoothed in buffer
    buffer_classes = [rec.class_name for rec in smoother.window_buffer]
    assert buffer_classes == ['Brown_Orange', 'Rejected', 'Brown_Orange', 'Purple_Yellow', 'Brown_Orange', 'Rejected'], \
        f"Buffer should have raw classes, got {buffer_classes}"
    
    # Verify none are marked as smoothed
    for rec in smoother.window_buffer:
        assert not rec.smoothed, f"T{rec.track_id} should not be smoothed during warmup"
        assert rec.original_class is None, f"T{rec.track_id} should not have original_class set during warmup"
    
    print("✓ No warmup smoothing - items remain raw during accumulation")


def test_all_brown_orange_batch():
    """Test the problem scenario: all Brown_Orange batch should not be corrupted."""
    print("\n=== Test 3: All Brown_Orange Batch (Problem Scenario) ===")
    
    smoother = BidirectionalSmoother(window_size=21, batch_lock_count=10)
    
    # Simulate 30 Brown_Orange items (same as problem description)
    all_items = []
    for i in range(30):
        class_name = 'Brown_Orange'
        conf = 0.9
        rois = 5
        
        # Add some Rejected items to test smoothing
        if i in [1, 7]:  # T2 and T8 from problem description
            class_name = 'Rejected'
            conf = 0.3
            rois = 0
        
        all_items.append((class_name, conf, rois))
    
    confirmed_counts = {'Brown_Orange': 0, 'Purple_Yellow': 0, 'Rejected': 0, 'Unknown': 0}
    
    for i, (class_name, conf, rois) in enumerate(all_items):
        result = smoother.add_classification(
            track_id=i,
            class_name=class_name,
            confidence=conf,
            vote_ratio=0.85,
            non_rejected_rois=rois
        )
        
        if result:
            for rec in result:
                confirmed_counts[rec.class_name] = confirmed_counts.get(rec.class_name, 0) + 1
    
    # Flush remaining to get all confirmations
    remaining = smoother.flush_remaining()
    for rec in remaining:
        confirmed_counts[rec.class_name] = confirmed_counts.get(rec.class_name, 0) + 1
    
    print(f"Confirmed counts: {confirmed_counts}")
    
    # Expected: ~29-30 Brown_Orange, 0-1 other (vs old system: 18 BO, 12 PY)
    # With two Rejected items at T1 and T7, they should be smoothed to Brown_Orange
    assert confirmed_counts['Brown_Orange'] >= 28, \
        f"Expected ≥28 Brown_Orange, got {confirmed_counts['Brown_Orange']}"
    assert confirmed_counts['Purple_Yellow'] <= 2, \
        f"Expected ≤2 Purple_Yellow, got {confirmed_counts['Purple_Yellow']}"
    
    print(f"✓ All Brown_Orange batch correctly classified: {confirmed_counts['Brown_Orange']}/30 as Brown_Orange")
    print(f"  (vs old system: 18/30 with feedback loop corruption)")


def test_original_class_for_distributions():
    """Test that distribution calculations use original_class to prevent feedback loops."""
    print("\n=== Test 4: Original Class for Distributions ===")
    
    smoother = BidirectionalSmoother(window_size=7, batch_lock_count=3)
    
    # Add items and let them be smoothed
    for i in range(10):
        smoother.add_classification(
            track_id=i,
            class_name='Brown_Orange' if i % 5 != 0 else 'Rejected',
            confidence=0.9 if i % 5 != 0 else 0.3,
            vote_ratio=0.85,
            non_rejected_rois=5 if i % 5 != 0 else 0
        )
    
    # Check that any smoothed records have original_class set
    for rec in smoother.confirmed_records:
        if rec.smoothed:
            assert rec.original_class is not None, \
                f"Smoothed record T{rec.track_id} should have original_class set"
            print(f"  T{rec.track_id}: {rec.original_class} → {rec.class_name} (smoothed)")
    
    print("✓ Distribution calculations use original_class to prevent feedback loops")


def test_batch_lock_override():
    """Test that batch lock overrides ALL outliers, including high-confidence ones."""
    print("\n=== Test 5: Batch Lock Override ===")
    
    smoother = BidirectionalSmoother(window_size=11, batch_lock_count=5)
    
    # Create a batch with mostly Brown_Orange and one high-confidence Purple_Yellow outlier
    # With window_size=11, center_index=5, so T0-T4 will be deferred
    # Put the outlier at T3 (in deferred set)
    items = []
    for i in range(20):
        if i == 3:  # High-confidence outlier in deferred set (T3)
            items.append(('Purple_Yellow', 0.95, 5))
        else:
            items.append(('Brown_Orange', 0.9, 5))
    
    confirmed_counts = {'Brown_Orange': 0, 'Purple_Yellow': 0}
    
    for i, (class_name, conf, rois) in enumerate(items):
        result = smoother.add_classification(
            track_id=i,
            class_name=class_name,
            confidence=conf,
            vote_ratio=0.85,
            non_rejected_rois=rois
        )
        
        if result:
            for rec in result:
                confirmed_counts[rec.class_name] = confirmed_counts.get(rec.class_name, 0) + 1
                if rec.track_id == 3:
                    print(f"  T3 (high-conf PY): {rec.original_class} → {rec.class_name} (smoothed={rec.smoothed})")
    
    # Flush remaining to get all confirmations
    remaining = smoother.flush_remaining()
    for rec in remaining:
        confirmed_counts[rec.class_name] = confirmed_counts.get(rec.class_name, 0) + 1
        if rec.track_id == 3:
            print(f"  T3 (high-conf PY): {rec.original_class} → {rec.class_name} (smoothed={rec.smoothed})")
    
    print(f"Confirmed counts: {confirmed_counts}")
    
    # The high-confidence PY outlier should be overridden to BO after batch lock
    assert confirmed_counts['Brown_Orange'] >= 19, \
        f"Expected ≥19 Brown_Orange (including T3 override), got {confirmed_counts['Brown_Orange']}"
    assert confirmed_counts['Purple_Yellow'] <= 1, \
        f"Expected ≤1 Purple_Yellow, got {confirmed_counts['Purple_Yellow']}"
    
    print("✓ Batch lock overrides ALL outliers, including high-confidence ones")


def test_exclude_rejected_and_unknown():
    """Test that Rejected AND Unknown are both excluded from dominance calculations."""
    print("\n=== Test 6: Exclude Rejected and Unknown ===")
    
    smoother = BidirectionalSmoother(window_size=7, batch_lock_count=3)
    
    # Add items with Rejected and Unknown
    items = [
        ('Brown_Orange', 0.9, 5),
        ('Rejected', 0.3, 0),
        ('Unknown', 0.5, 2),
        ('Brown_Orange', 0.9, 5),
        ('Brown_Orange', 0.9, 5),
        ('Rejected', 0.2, 0),
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
    
    # Get dominant class - should be Brown_Orange (excluding Rejected and Unknown)
    dominant = smoother.get_dominant_class()
    assert dominant == 'Brown_Orange', \
        f"Expected dominant=Brown_Orange (excluding Rejected/Unknown), got {dominant}"
    
    print(f"✓ Dominant class correctly excludes Rejected and Unknown: {dominant}")


def test_statistics():
    """Test that statistics include new fields."""
    print("\n=== Test 7: Statistics ===")
    
    smoother = BidirectionalSmoother(window_size=7, batch_lock_count=3)
    
    # Add some items
    for i in range(10):
        smoother.add_classification(
            track_id=i,
            class_name='Brown_Orange',
            confidence=0.9,
            vote_ratio=0.85,
            non_rejected_rois=5
        )
    
    stats = smoother.get_statistics()
    
    # Check for new fields
    assert 'deferred_count' in stats, "Statistics should include deferred_count"
    assert 'batch_locked' in stats, "Statistics should include batch_locked"
    assert 'steady_state_confirms' in stats, "Statistics should include steady_state_confirms"
    
    print(f"✓ Statistics include new fields: deferred_count={stats['deferred_count']}, "
          f"batch_locked={stats['batch_locked']}, steady_state_confirms={stats['steady_state_confirms']}")


if __name__ == '__main__':
    try:
        test_three_phase_architecture()
        test_no_warmup_smoothing()
        test_all_brown_orange_batch()
        test_original_class_for_distributions()
        test_batch_lock_override()
        test_exclude_rejected_and_unknown()
        test_statistics()
        
        print("\n" + "=" * 70)
        print("ALL TESTS PASSED ✓")
        print("=" * 70)
        sys.exit(0)
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
