"""
Test script to verify the track lifecycle behavior.

Updated for the new system where:
- Lost tracks are NEVER counted (moved to ghost buffer, then expire)
- Ghost tracks can be re-associated when bags reappear
- Tracks must exit from top to be counted

This replaces the old "valid journey rescue" tests.
"""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.tracking.ConveyorTracker import ConveyorTracker
from src.config.tracking_config import TrackingConfig
from src.detection.BaseDetection import Detection


def test_track_lifecycle():
    """Test that the new track lifecycle works correctly."""
    print("=" * 70)
    print("Testing Track Lifecycle (Ghost Recovery, No Lost Rescue)")
    print("=" * 70)

    config = TrackingConfig()
    config.max_frames_without_detection = 5
    config.min_track_duration_frames = 3
    config.min_travel_duration_seconds = 0.0
    config.ghost_track_max_age_seconds = 0.01

    tracker = ConveyorTracker(config)
    tracker.frame_height = 1000
    tracker.frame_width = 1000

    all_passed = True

    # Test 1: Lost track NOT rescued
    print("\n" + "-" * 70)
    print("[Test 1] Lost track is NOT rescued (ghost expires)")
    print("-" * 70)

    det = Detection(bbox=(450, 780, 550, 820), confidence=0.9, class_id=0)
    tracker.update([det], (1000, 1000))
    track_id_1 = list(tracker.tracks.keys())[0]

    for y in [770, 760, 750]:
        det = Detection(bbox=(450, y-20, 550, y+20), confidence=0.9, class_id=0)
        tracker.update([det])

    for _ in range(config.max_frames_without_detection + 1):
        tracker.update([])

    time.sleep(0.05)
    tracker.update([])

    events = tracker.get_completed_events()
    event = next((e for e in events if e.track_id == track_id_1), None)

    if event and event.event_type == 'track_lost':
        print(f"  ✓ PASS: T{track_id_1} correctly NOT rescued ('{event.event_type}')")
    else:
        event_type = event.event_type if event else 'no event'
        print(f"  ✗ FAIL: T{track_id_1} got '{event_type}' (expected 'track_lost')")
        all_passed = False

    # Test 2: Normal exit from top
    print("\n" + "-" * 70)
    print("[Test 2] Normal Exit: Bottom -> Top Exit Zone")
    print("-" * 70)

    det = Detection(bbox=(100, 750, 200, 850), confidence=0.9, class_id=0)
    tracker.update([det])
    track_id_2 = list(tracker.tracks.keys())[0]

    for y in [730, 660, 590, 520, 450, 380, 310, 240, 170, 100]:
        det = Detection(bbox=(100, y-50, 200, y+50), confidence=0.9, class_id=0)
        tracker.update([det])

    det = Detection(bbox=(100, 0, 200, 20), confidence=0.9, class_id=0)
    tracker.update([det])

    events = tracker.get_completed_events()
    event = next((e for e in events if e.track_id == track_id_2), None)

    if event and event.event_type == 'track_completed':
        print(f"  ✓ PASS: T{track_id_2} completed normally")
    else:
        event_type = event.event_type if event else 'no event'
        print(f"  ✗ FAIL: T{track_id_2} got '{event_type}' (expected 'track_completed')")
        all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 70)
    return all_passed


if __name__ == "__main__":
    success = test_track_lifecycle()
    sys.exit(0 if success else 1)
