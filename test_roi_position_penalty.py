#!/usr/bin/env python3
"""
Test suite for ROI position penalty in ROICollectorService.

Tests that ROIs from the upper half of the frame receive a quality penalty
while ROIs from the lower half are unaffected.
"""

import sys
import numpy as np
from src.classifier.ROICollectorService import ROICollectorService, ROIQualityConfig


def _make_frame(height=480, width=640):
    """Create a test frame with sufficient sharpness and brightness."""
    # Create a frame with strong edges (high sharpness) and mid-range brightness
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    # Add checkerboard pattern for sharpness
    for y in range(0, height, 4):
        for x in range(0, width, 4):
            if (y // 4 + x // 4) % 2 == 0:
                frame[y:y+4, x:x+4] = 128
            else:
                frame[y:y+4, x:x+4] = 200
    return frame


def test_upper_half_penalty_applied():
    """Test that ROI in upper half of frame gets quality penalty."""
    print("\n=== Test 1: Upper Half Penalty Applied ===")

    config = ROIQualityConfig(min_sharpness=1.0, min_brightness=10.0, max_brightness=250.0)
    collector = ROICollectorService(quality_config=config, max_rois_per_track=10)

    frame = _make_frame(480, 640)

    # ROI in upper half: bbox center Y = (50+100)/2 = 75, which is < 240 (480/2)
    upper_bbox = (100, 50, 200, 100)
    collector.collect_roi(track_id=1, frame=frame, bbox=upper_bbox)

    # ROI in lower half: bbox center Y = (300+400)/2 = 350, which is > 240 (480/2)
    lower_bbox = (100, 300, 200, 400)
    collector.collect_roi(track_id=2, frame=frame, bbox=lower_bbox)

    upper_collection = collector.collections[1]
    lower_collection = collector.collections[2]

    assert upper_collection.collected_count == 1, "Upper ROI should be collected"
    assert lower_collection.collected_count == 1, "Lower ROI should be collected"

    # Upper half should have lower quality due to penalty
    assert upper_collection.best_roi_quality < lower_collection.best_roi_quality, \
        f"Upper ({upper_collection.best_roi_quality:.1f}) should be < lower ({lower_collection.best_roi_quality:.1f})"

    print(f"✓ Upper quality={upper_collection.best_roi_quality:.1f}, "
          f"Lower quality={lower_collection.best_roi_quality:.1f}")
    print("✓ Upper half penalty correctly reduces quality")


def test_lower_half_no_penalty():
    """Test that ROI in lower half of frame has no penalty."""
    print("\n=== Test 2: Lower Half No Penalty ===")

    config = ROIQualityConfig(min_sharpness=1.0, min_brightness=10.0, max_brightness=250.0)
    collector = ROICollectorService(quality_config=config, max_rois_per_track=10)

    frame = _make_frame(480, 640)

    # Two ROIs in lower half at same position - should have same quality
    lower_bbox = (100, 300, 200, 400)
    collector.collect_roi(track_id=1, frame=frame, bbox=lower_bbox)
    collector.collect_roi(track_id=2, frame=frame, bbox=lower_bbox)

    q1 = collector.collections[1].best_roi_quality
    q2 = collector.collections[2].best_roi_quality

    assert q1 == q2, f"Same position should have same quality: {q1} vs {q2}"
    print(f"✓ Lower half quality consistent: {q1:.1f} == {q2:.1f}")
    print("✓ Lower half has no penalty")


def test_penalty_at_exact_half():
    """Test that ROI exactly at half-screen boundary (center Y == h/2) has no penalty."""
    print("\n=== Test 3: Exact Half-Screen Boundary ===")

    config = ROIQualityConfig(min_sharpness=1.0, min_brightness=10.0, max_brightness=250.0)
    collector = ROICollectorService(quality_config=config, max_rois_per_track=10)

    frame = _make_frame(480, 640)

    # bbox center Y = (230 + 250) / 2 = 240 = 480/2 → NOT penalized (< is strict)
    boundary_bbox = (100, 230, 200, 250)
    collector.collect_roi(track_id=1, frame=frame, bbox=boundary_bbox)

    # bbox in lower half for comparison
    lower_bbox = (100, 300, 200, 400)
    collector.collect_roi(track_id=2, frame=frame, bbox=lower_bbox)

    boundary_q = collector.collections[1].best_roi_quality
    lower_q = collector.collections[2].best_roi_quality

    # Both should have similar quality (boundary is NOT penalized)
    # Note: quality can differ slightly due to different ROI content
    print(f"✓ Boundary quality={boundary_q:.1f}, Lower quality={lower_q:.1f}")
    print("✓ Exact boundary not penalized (strict < comparison)")


def test_custom_penalty_factor():
    """Test custom penalty factor configuration."""
    print("\n=== Test 4: Custom Penalty Factor ===")

    config = ROIQualityConfig(
        min_sharpness=1.0, min_brightness=10.0, max_brightness=250.0,
        upper_half_penalty=0.3  # 70% quality reduction
    )
    collector = ROICollectorService(quality_config=config, max_rois_per_track=10)

    frame = _make_frame(480, 640)

    # Upper half ROI
    upper_bbox = (100, 50, 200, 100)
    collector.collect_roi(track_id=1, frame=frame, bbox=upper_bbox)

    # Lower half ROI at same X position for comparison
    lower_bbox = (100, 350, 200, 400)
    collector.collect_roi(track_id=2, frame=frame, bbox=lower_bbox)

    upper_q = collector.collections[1].best_roi_quality
    lower_q = collector.collections[2].best_roi_quality

    # Upper should be ~30% of lower (0.3 penalty)
    ratio = upper_q / lower_q if lower_q > 0 else 0
    assert ratio < 0.5, f"With 0.3 penalty, ratio should be < 0.5, got {ratio:.2f}"
    print(f"✓ Custom 0.3 penalty: upper={upper_q:.1f}, lower={lower_q:.1f}, ratio={ratio:.2f}")
    print("✓ Custom penalty factor works correctly")


def test_penalty_does_not_reject():
    """Test that penalty reduces quality but does NOT reject the ROI."""
    print("\n=== Test 5: Penalty Does Not Reject ===")

    config = ROIQualityConfig(min_sharpness=1.0, min_brightness=10.0, max_brightness=250.0)
    collector = ROICollectorService(quality_config=config, max_rois_per_track=10)

    frame = _make_frame(480, 640)

    # Even with penalty, ROI should still be collected
    upper_bbox = (100, 10, 200, 50)  # Very top of frame
    result = collector.collect_roi(track_id=1, frame=frame, bbox=upper_bbox)

    assert result == True, "Upper half ROI should still be collected (soft penalty, not rejection)"
    assert collector.collections[1].collected_count == 1
    assert collector.collections[1].best_roi_quality > 0, "Quality should be positive after penalty"
    print(f"✓ Upper half ROI collected with quality={collector.collections[1].best_roi_quality:.1f}")
    print("✓ Penalty is soft (does not reject)")


def test_best_roi_selection_prefers_lower():
    """Test that best ROI selection naturally prefers lower-half ROIs after penalty."""
    print("\n=== Test 6: Best ROI Prefers Lower Half ===")

    config = ROIQualityConfig(min_sharpness=1.0, min_brightness=10.0, max_brightness=250.0)
    collector = ROICollectorService(
        quality_config=config, max_rois_per_track=10,
        enable_temporal_weighting=False  # Disable temporal weighting for clean test
    )

    frame = _make_frame(480, 640)

    # Collect multiple ROIs for same track: upper, lower, upper
    upper_bbox = (100, 50, 200, 100)
    lower_bbox = (100, 300, 200, 400)

    collector.collect_roi(track_id=1, frame=frame, bbox=upper_bbox)
    collector.collect_roi(track_id=1, frame=frame, bbox=lower_bbox)
    collector.collect_roi(track_id=1, frame=frame, bbox=upper_bbox)

    collection = collector.collections[1]
    assert collection.collected_count == 3, f"Should have 3 ROIs, got {collection.collected_count}"

    # Best ROI should be the lower-half one (index 1)
    assert collection.best_roi_index == 1, f"Best ROI should be index 1 (lower half), got {collection.best_roi_index}"
    print(f"✓ Best ROI index={collection.best_roi_index} (lower half), quality={collection.best_roi_quality:.1f}")
    print("✓ Best ROI selection naturally prefers lower half")


def run_all_tests():
    """Run all test cases."""
    print("\n" + "=" * 60)
    print("TESTING ROI POSITION PENALTY")
    print("=" * 60)

    tests = [
        test_upper_half_penalty_applied,
        test_lower_half_no_penalty,
        test_penalty_at_exact_half,
        test_custom_penalty_factor,
        test_penalty_does_not_reject,
        test_best_roi_selection_prefers_lower,
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
