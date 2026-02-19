"""
Test script for entry type classification and ghost track validation.

Updated for the new system where:
- _validate_lost_track_as_completed() has been removed
- Lost tracks go to ghost buffer for potential re-association
- Entry type classification provides diagnostics (bottom_entry, midway_entry, thrown_entry)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.tracking.ConveyorTracker import ConveyorTracker, TrackedObject
from src.config.tracking_config import TrackingConfig
from src.detection.BaseDetection import Detection


def test_entry_type_classification():
    """Test entry type classification for tracks."""
    print("=" * 70)
    print("Testing Entry Type Classification")
    print("=" * 70)

    config = TrackingConfig()
    tracker = ConveyorTracker(config)
    tracker.frame_height = 1000
    tracker.frame_width = 1000

    all_passed = True

    # Test 1: Bottom entry
    print("\n[Test 1] Bottom entry (y=800)")
    det = Detection(bbox=(100, 750, 200, 850), confidence=0.9, class_id=0)
    tracker.update([det], (1000, 1000))
    track = list(tracker.tracks.values())[0]
    if track.entry_type == "bottom_entry":
        print("  ✓ PASS: Correctly classified as bottom_entry")
    else:
        print(f"  ✗ FAIL: Got {track.entry_type}, expected bottom_entry")
        all_passed = False

    tracker.cleanup()

    # Test 2: Midway entry (y=400, above bottom 40% line)
    print("\n[Test 2] Midway entry (y=400)")
    det = Detection(bbox=(100, 350, 200, 450), confidence=0.9, class_id=0)
    tracker.update([det], (1000, 1000))
    track = list(tracker.tracks.values())[0]
    if track.entry_type == "midway_entry":
        print("  ✓ PASS: Correctly classified as midway_entry")
    else:
        print(f"  ✗ FAIL: Got {track.entry_type}, expected midway_entry")
        all_passed = False

    tracker.cleanup()

    # Test 3: Ghost recovery count
    print("\n[Test 3] Ghost recovery count starts at 0")
    det = Detection(bbox=(100, 750, 200, 850), confidence=0.9, class_id=0)
    tracker.update([det], (1000, 1000))
    track = list(tracker.tracks.values())[0]
    if track.ghost_recovery_count == 0:
        print("  ✓ PASS: ghost_recovery_count=0")
    else:
        print(f"  ✗ FAIL: ghost_recovery_count={track.ghost_recovery_count}")
        all_passed = False

    # Test 4: Shadow fields initialized correctly
    print("\n[Test 4] Shadow fields initialization")
    if track.shadow_of is None and len(track.shadow_tracks) == 0:
        print("  ✓ PASS: shadow_of=None, shadow_tracks={}")
    else:
        print(f"  ✗ FAIL: shadow_of={track.shadow_of}, shadow_tracks={track.shadow_tracks}")
        all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 70)
    return all_passed


if __name__ == "__main__":
    success = test_entry_type_classification()
    sys.exit(0 if success else 1)
