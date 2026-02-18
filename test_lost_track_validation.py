"""
Simple direct test for lost track recovery logic.
Tests the _validate_lost_track_as_completed method directly.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.tracking.ConveyorTracker import ConveyorTracker, TrackedObject
from src.config.tracking_config import TrackingConfig
from src.detection.BaseDetection import Detection
from collections import deque


def test_validate_lost_track_method():
    """Direct test of _validate_lost_track_as_completed method."""
    print("=" * 70)
    print("Testing _validate_lost_track_as_completed() Method Directly")
    print("=" * 70)

    config = TrackingConfig()
    tracker = ConveyorTracker(config)
    tracker.frame_height = 1000
    tracker.frame_width = 1000

    print(f"\nFrame size: {tracker.frame_width}x{tracker.frame_height}")
    print(f"Entry zone: bottom 60% (y >= {1000 * 0.4:.0f})")
    print(f"Exit zone for rescue: top 40% (y <= {1000 * 0.4:.0f})")
    print(f"Min travel: 30% of frame height ({1000 * 0.3:.0f}px)\n")

    all_passed = True

    # Test 1: Valid journey
    print("[Test 1] Valid journey: start=800, end=300, hits=10")
    track1 = TrackedObject(track_id=1, bbox=(100, 250, 200, 350), confidence=0.9)
    track1.entry_center_y = 800
    track1.position_history = deque([(100, 800), (100, 700), (100, 600), (100, 500), (100, 400), (100, 300)])
    track1.hits = 10
    track1.age = 2

    result = tracker._validate_lost_track_as_completed(track1)
    if result:
        print("  ✓ PASS: Validated as completed")
    else:
        print("  ✗ FAIL: Not validated")
        all_passed = False

    # Test 2: Started too high
    print("\n[Test 2] Invalid: started too high (start=300, end=200)")
    track2 = TrackedObject(track_id=2, bbox=(100, 150, 200, 250), confidence=0.9)
    track2.entry_center_y = 300
    track2.position_history = deque([(100, 300), (100, 250), (100, 200)])
    track2.hits = 5
    track2.age = 1

    result = tracker._validate_lost_track_as_completed(track2)
    if not result:
        print("  ✓ PASS: Correctly rejected")
    else:
        print("  ✗ FAIL: Should not validate")
        all_passed = False

    # Test 3: Insufficient travel
    print("\n[Test 3] Invalid: insufficient travel (start=800, end=600, only 200px)")
    track3 = TrackedObject(track_id=3, bbox=(100, 550, 200, 650), confidence=0.9)
    track3.entry_center_y = 800
    track3.position_history = deque([(100, 800), (100, 700), (100, 600)])
    track3.hits = 5
    track3.age = 1

    result = tracker._validate_lost_track_as_completed(track3)
    if not result:
        print("  ✓ PASS: Correctly rejected (travel < 300px)")
    else:
        print("  ✗ FAIL: Should not validate")
        all_passed = False

    # Test 4: Low hit rate
    print("\n[Test 4] Invalid: low hit rate (hits=3, age=10, hit_rate=0.23)")
    track4 = TrackedObject(track_id=4, bbox=(100, 250, 200, 350), confidence=0.9)
    track4.entry_center_y = 800
    track4.position_history = deque([(100, 800), (100, 300)])
    track4.hits = 3
    track4.age = 10

    result = tracker._validate_lost_track_as_completed(track4)
    if not result:
        print("  ✓ PASS: Correctly rejected (hit_rate < 0.5)")
    else:
        print("  ✗ FAIL: Should not validate")
        all_passed = False

    # Test 5: Borderline case - exactly at thresholds
    print("\n[Test 5] Borderline: start=600, end=400, travel=200px (< 300px required)")
    track5 = TrackedObject(track_id=5, bbox=(100, 350, 200, 450), confidence=0.9)
    track5.entry_center_y = 600
    track5.position_history = deque([(100, 600), (100, 500), (100, 400)])
    track5.hits = 5
    track5.age = 1

    result = tracker._validate_lost_track_as_completed(track5)
    if not result:
        print("  ✓ PASS: Correctly rejected (insufficient travel)")
    else:
        print("  ✗ FAIL: Should not validate")
        all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    success = test_validate_lost_track_method()
    sys.exit(0 if success else 1)
