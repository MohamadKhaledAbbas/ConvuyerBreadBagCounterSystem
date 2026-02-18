"""
Test script to verify the Valid Journey Recovery logic.

This tests that lost tracks which traveled from bottom to near-top
are correctly rescued as 'track_completed' instead of 'track_lost'.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.tracking.ConveyorTracker import ConveyorTracker
from src.config.tracking_config import TrackingConfig
from src.detection.BaseDetection import Detection


def test_valid_journey_recovery():
    """Test that lost tracks with valid journeys are rescued."""
    print("=" * 70)
    print("Testing Valid Journey Recovery Logic")
    print("=" * 70)

    # Setup tracker with short timeout for testing
    config = TrackingConfig()
    config.max_frames_without_detection = 5  # Short timeout for test
    config.min_track_duration_frames = 3

    tracker = ConveyorTracker(config)
    tracker.frame_height = 1000
    tracker.frame_width = 1000

    print(f"\nFrame size: {tracker.frame_width}x{tracker.frame_height}")
    print(f"Entry zone: bottom 60% (y >= 400)")
    print(f"Exit zone for rescue: top 40% (y <= 400)")
    print(f"Min travel: 30% of frame height (300px)")

    all_passed = True

    # ---------------------------------------------------------
    # Test Case 1: Valid Journey (Bottom -> Near-Top -> Lost)
    # Should be RESCUED as track_completed
    # ---------------------------------------------------------
    print("\n" + "-" * 70)
    print("[Test 1] Valid Journey: Bottom (y=800) -> Near-Top (y=250) -> Lost")
    print("-" * 70)

    # Start at bottom but NOT in bottom exit zone (bottom 15% = y >= 850)
    # So start at y=800 which is valid
    det = Detection(bbox=(450, 750, 550, 850), confidence=0.9, class_id=0)
    tracker.update([det], (1000, 1000))

    if not tracker.tracks:
        print("  ✗ FAIL: Track not created - check bottom exit zone filtering")
        return False

    track_id_1 = list(tracker.tracks.keys())[0]
    print(f"  Created T{track_id_1} at y=800")

    # Move up: 800 -> 600 -> 400 -> 250
    # Keep ABOVE strict exit zone (y > 150) but within rescue zone (y <= 400)
    for y in [600, 400, 250]:
        det = Detection(bbox=(450, y-50, 550, y+50), confidence=0.9, class_id=0)
        tracker.update([det])
    print(f"  Moved T{track_id_1} to y=250 (4 hits, in rescue zone but not exit zone)")

    # Simulate loss (no detections)
    print(f"  Simulating loss ({config.max_frames_without_detection + 1} empty frames)...")
    for _ in range(config.max_frames_without_detection + 1):
        tracker.update([])

    # Check result
    events = tracker.get_completed_events()
    event = next((e for e in events if e.track_id == track_id_1), None)

    if event and event.event_type == 'track_completed':
        print(f"  ✓ PASS: T{track_id_1} rescued as '{event.event_type}'")
    else:
        print(f"  ✗ FAIL: T{track_id_1} got '{event.event_type if event else 'no event'}'")
        all_passed = False

    # ---------------------------------------------------------
    # Test Case 2: Started Too High (Mid -> Top -> Lost)
    # Should remain track_lost
    # ---------------------------------------------------------
    print("\n" + "-" * 70)
    print("[Test 2] Invalid Start: Mid (y=300) -> Near-Top (y=200) -> Lost")
    print("-" * 70)

    # Start at middle (y=300, which is < 400, so NOT in entry zone)
    det = Detection(bbox=(650, 250, 750, 350), confidence=0.9, class_id=0)
    tracker.update([det])

    if not tracker.tracks:
        print("  ✗ FAIL: Track T2 not created")
        return False

    track_id_2 = list(tracker.tracks.keys())[0]
    print(f"  Created T{track_id_2} at y=300 (above entry zone threshold)")

    # Move up a bit to y=200 (above strict exit zone at 150)
    for y in [250, 200]:
        det = Detection(bbox=(650, y-50, 750, y+50), confidence=0.9, class_id=0)
        tracker.update([det])
    print(f"  Moved T{track_id_2} to y=200")

    # Simulate loss
    print(f"  Simulating loss...")
    for _ in range(config.max_frames_without_detection + 1):
        tracker.update([])

    # Check result
    events = tracker.get_completed_events()
    event = next((e for e in events if e.track_id == track_id_2), None)

    if event and event.event_type == 'track_lost':
        print(f"  ✓ PASS: T{track_id_2} correctly marked as '{event.event_type}'")
    else:
        print(f"  ✗ FAIL: T{track_id_2} got '{event.event_type if event else 'no event'}' (expected 'track_lost')")
        all_passed = False

    # ---------------------------------------------------------
    # Test Case 3: Insufficient Travel (Bottom -> Mid -> Lost)
    # Should remain track_lost
    # ---------------------------------------------------------
    print("\n" + "-" * 70)
    print("[Test 3] Insufficient Travel: Bottom (y=800) -> Mid (y=600) -> Lost")
    print("-" * 70)

    # Start at bottom (but not in bottom exit zone)
    det = Detection(bbox=(250, 750, 350, 850), confidence=0.9, class_id=0)
    tracker.update([det])

    if not tracker.tracks:
        print("  ✗ FAIL: Track T3 not created")
        return False

    track_id_3 = list(tracker.tracks.keys())[0]
    print(f"  Created T{track_id_3} at y=800")

    # Move up only a little (to y=600, which is > 400, so NOT in rescue zone)
    for y in [700, 600]:
        det = Detection(bbox=(250, y-50, 350, y+50), confidence=0.9, class_id=0)
        tracker.update([det])
    print(f"  Moved T{track_id_3} to y=600 (didn't reach near-top)")

    # Simulate loss
    print(f"  Simulating loss...")
    for _ in range(config.max_frames_without_detection + 1):
        tracker.update([])

    # Check result
    events = tracker.get_completed_events()
    event = next((e for e in events if e.track_id == track_id_3), None)

    if event and event.event_type == 'track_lost':
        print(f"  ✓ PASS: T{track_id_3} correctly marked as '{event.event_type}'")
    else:
        print(f"  ✗ FAIL: T{track_id_3} got '{event.event_type if event else 'no event'}' (expected 'track_lost')")
        all_passed = False

    # ---------------------------------------------------------
    # Test Case 4: Normal Exit (Bottom -> Top Exit Zone)
    # Should be track_completed (normal path)
    # ---------------------------------------------------------
    print("\n" + "-" * 70)
    print("[Test 4] Normal Exit: Bottom (y=800) -> Top Exit Zone (y=50)")
    print("-" * 70)

    # Start at bottom (but not in bottom exit zone)
    det = Detection(bbox=(100, 750, 200, 850), confidence=0.9, class_id=0)
    tracker.update([det])

    if not tracker.tracks:
        print("  ✗ FAIL: Track T4 not created")
        return False

    track_id_4 = list(tracker.tracks.keys())[0]
    print(f"  Created T{track_id_4} at y=800")

    # Move all the way to exit zone (y <= 150 for 15% exit zone)
    # Need enough frames to meet min_track_duration_frames
    for y in [700, 600, 500, 400, 300, 200, 100, 50]:
        det = Detection(bbox=(100, y-50, 200, y+50), confidence=0.9, class_id=0)
        tracker.update([det])
    print(f"  Moved T{track_id_4} to y=50 (exit zone), hits={tracker.tracks.get(track_id_4).hits if track_id_4 in tracker.tracks else 'completed'}")

    # This should trigger immediate completion (exiting frame)
    events = tracker.get_completed_events()
    event = next((e for e in events if e.track_id == track_id_4), None)

    if event and event.event_type == 'track_completed':
        print(f"  ✓ PASS: T{track_id_4} completed normally as '{event.event_type}'")
    else:
        print(f"  ✗ FAIL: T{track_id_4} got '{event.event_type if event else 'no event'}' (expected 'track_completed')")
        all_passed = False

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    success = test_valid_journey_recovery()
    sys.exit(0 if success else 1)
