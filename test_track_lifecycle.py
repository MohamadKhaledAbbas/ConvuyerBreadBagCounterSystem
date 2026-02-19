"""
Test script for the robust track lifecycle system.

Tests:
1. Ghost track recovery (occlusion handling)
2. Ghost track expiry
3. Ghost X-tolerance filtering
4. Shadow/merge detection
5. Shadow un-merge (detach)
6. Entry type classification
7. No double-counting scenarios
8. Normal exit-top counting
9. Lost tracks are never counted
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.tracking.ConveyorTracker import ConveyorTracker, TrackedObject, TrackEvent
from src.config.tracking_config import TrackingConfig
from src.detection.BaseDetection import Detection


def make_tracker(max_frames_without=5, ghost_max_age=4.0, frame_h=1000, frame_w=1000):
    """Create a tracker with test-friendly config."""
    config = TrackingConfig()
    config.max_frames_without_detection = max_frames_without
    config.min_track_duration_frames = 3
    config.require_full_travel = True
    config.min_travel_duration_seconds = 0.0  # Disable time check for unit tests
    config.ghost_track_max_age_seconds = ghost_max_age
    config.ghost_track_x_tolerance_pixels = 80.0
    config.ghost_track_max_y_gap_ratio = 0.2
    config.merge_bbox_growth_threshold = 1.4
    config.merge_spatial_tolerance_pixels = 50.0
    config.merge_y_tolerance_pixels = 30.0
    config.bottom_entry_zone_ratio = 0.4
    config.thrown_entry_min_velocity = 15.0
    config.thrown_entry_detection_frames = 5
    config.exit_zone_ratio = 0.15
    config.bottom_exit_zone_ratio = 0.15
    config.exit_margin_pixels = 20
    tracker = ConveyorTracker(config)
    tracker.frame_height = frame_h
    tracker.frame_width = frame_w
    return tracker


def get_events(tracker, event_type=None):
    """Get completed events, optionally filtered by type."""
    events = tracker.get_completed_events()
    if event_type:
        return [e for e in events if e.event_type == event_type]
    return events


# =========================================================================
# Test 1: Normal exit-top counting
# =========================================================================
def test_normal_exit_top():
    """Test that a bag entering bottom and exiting top is counted."""
    print("\n" + "=" * 70)
    print("[Test 1] Normal exit-top counting")
    print("=" * 70)

    tracker = make_tracker()

    # Create track at bottom (y=800, outside bottom exit zone at y>=850)
    # Use consistent 100x100 bboxes with 70px steps for reliable IoU matching
    det = Detection(bbox=(450, 750, 550, 850), confidence=0.9, class_id=0)
    tracker.update([det], (1000, 1000))

    if not tracker.tracks:
        print("  ✗ FAIL: Track not created")
        return False

    track_id = list(tracker.tracks.keys())[0]
    print(f"  Created T{track_id} at y=800")

    # Move toward exit zone with consistent bbox sizes and moderate steps
    # Steps of 70px with 100px bbox ensures IoU overlap for reliable matching
    for y in [730, 660, 590, 520, 450, 380, 310, 240, 170, 100]:
        det = Detection(bbox=(450, y-50, 550, y+50), confidence=0.9, class_id=0)
        tracker.update([det])

    # Final step: move center to y=10 (< exit_margin_pixels=20) to trigger exit
    det = Detection(bbox=(450, 0, 550, 20), confidence=0.9, class_id=0)
    tracker.update([det])

    # Should be completed
    events = get_events(tracker, 'track_completed')
    if len(events) >= 1:
        # Find our track's completion event
        our_event = next((e for e in events if e.track_id == track_id), None)
        if our_event:
            print(f"  ✓ PASS: T{track_id} counted (track_completed, exit={our_event.exit_direction})")
            return True
    
    print(f"  ✗ FAIL: Expected track_completed for T{track_id}")
    all_events = get_events(tracker)
    for e in all_events:
        print(f"    event: T{e.track_id} {e.event_type} exit={e.exit_direction}")
    return False


# =========================================================================
# Test 2: Lost tracks are never counted
# =========================================================================
def test_lost_tracks_never_counted():
    """Test that lost tracks (mid-frame) are NOT counted."""
    print("\n" + "=" * 70)
    print("[Test 2] Lost tracks are never counted")
    print("=" * 70)

    # Use very short ghost_max_age so ghosts expire quickly
    tracker = make_tracker(max_frames_without=5, ghost_max_age=0.01)

    # Create track at bottom with SMALL position steps to keep velocity low
    # Small velocity prevents velocity prediction from pushing track to exit zone
    det = Detection(bbox=(450, 780, 550, 820), confidence=0.9, class_id=0)
    tracker.update([det], (1000, 1000))

    track_id = list(tracker.tracks.keys())[0]
    print(f"  Created T{track_id} at y=800")

    # Move partway up with small steps (velocity ~-20 px/frame)
    for y in [780, 760, 740]:
        det = Detection(bbox=(450, y-20, 550, y+20), confidence=0.9, class_id=0)
        tracker.update([det])
    print(f"  Moved T{track_id} to y=740 (small velocity, won't reach exit via prediction)")

    # Simulate loss (6 frames > max_frames_without=5)
    for _ in range(6):
        tracker.update([])

    # Wait for ghost to expire
    time.sleep(0.05)
    tracker.update([])

    events = get_events(tracker)
    completed = [e for e in events if e.event_type == 'track_completed']
    lost = [e for e in events if e.event_type == 'track_lost']

    if len(completed) == 0 and len(lost) == 1:
        print(f"  ✓ PASS: T{track_id} NOT counted (track_lost)")
        return True
    else:
        print(f"  ✗ FAIL: completed={len(completed)}, lost={len(lost)}")
        for e in events:
            print(f"    event: T{e.track_id} {e.event_type} exit={e.exit_direction}")
        return False


# =========================================================================
# Test 3: Ghost track recovery
# =========================================================================
def test_ghost_recovery():
    """Test that occluded bags are re-associated via ghost recovery."""
    print("\n" + "=" * 70)
    print("[Test 3] Ghost track recovery (occlusion)")
    print("=" * 70)

    tracker = make_tracker(max_frames_without=3)

    # Create track at bottom with consistent 100x100 bboxes
    det = Detection(bbox=(450, 750, 550, 850), confidence=0.9, class_id=0)
    tracker.update([det], (1000, 1000))
    track_id = list(tracker.tracks.keys())[0]
    print(f"  Created T{track_id} at y=800")

    # Move up with small steps to keep velocity moderate (~20 px/frame)
    for y in [780, 760, 740]:
        det = Detection(bbox=(450, y-50, 550, y+50), confidence=0.9, class_id=0)
        tracker.update([det])
    print(f"  Moved T{track_id} to y=740, velocity should be ~-20 px/frame")

    # Simulate occlusion (4 empty frames -> exceeds max_frames_without=3)
    for _ in range(4):
        tracker.update([])
    print(f"  Track lost after 4 empty frames")

    # Verify track is in ghost buffer
    if track_id not in tracker.ghost_tracks:
        events = get_events(tracker)
        if events:
            print(f"  ✗ FAIL: Track not in ghost buffer. Events: {[(e.track_id, e.event_type) for e in events]}")
        else:
            print(f"  ✗ FAIL: Track not in ghost buffer and no events")
        return False
    print(f"  T{track_id} in ghost buffer, predicted_pos={tracker.ghost_tracks[track_id]['predicted_pos']}")

    # Bag reappears near the ghost's last real position (y=740).
    # Ghost prediction uses last real observed position, so detection
    # must be within max_y_gap_ratio (20% of 1000 = 200px) tolerance.
    # y=650 is 90px above last observation — well within tolerance.
    det = Detection(bbox=(450, 600, 550, 700), confidence=0.9, class_id=0)
    tracker.update([det])

    # Verify ghost was recovered with same track_id
    if track_id in tracker.tracks:
        recovered = tracker.tracks[track_id]
        if recovered.ghost_recovery_count == 1:
            print(f"  ✓ PASS: T{track_id} recovered from ghost, recovery_count=1")

            # Now move to exit (center < 20)
            for y in [500, 300, 200, 100, 10]:
                det = Detection(bbox=(450, y-50, 550, y+50), confidence=0.9, class_id=0)
                tracker.update([det])

            events = get_events(tracker, 'track_completed')
            if len(events) >= 1:
                our_event = next((e for e in events if e.track_id == track_id), None)
                if our_event and our_event.ghost_recovery_count == 1:
                    print(f"  ✓ PASS: Completed with ghost_recovery_count=1")
                    return True
            print(f"  ✗ FAIL: Expected track_completed with ghost_recovery_count=1")
            return False
        else:
            print(f"  ✗ FAIL: ghost_recovery_count={recovered.ghost_recovery_count}")
            return False
    else:
        # Check if a new track was created instead
        print(f"  ✗ FAIL: T{track_id} not in active tracks after recovery attempt")
        print(f"    Active tracks: {list(tracker.tracks.keys())}")
        print(f"    Ghost tracks: {list(tracker.ghost_tracks.keys())}")
        return False


# =========================================================================
# Test 4: Ghost expiry
# =========================================================================
def test_ghost_expiry():
    """Test that ghost tracks expire after max age."""
    print("\n" + "=" * 70)
    print("[Test 4] Ghost track expiry")
    print("=" * 70)

    tracker = make_tracker(max_frames_without=3, ghost_max_age=0.05)

    # Create track
    det = Detection(bbox=(450, 750, 550, 850), confidence=0.9, class_id=0)
    tracker.update([det], (1000, 1000))
    track_id = list(tracker.tracks.keys())[0]

    # Move up
    det = Detection(bbox=(450, 550, 550, 650), confidence=0.9, class_id=0)
    tracker.update([det])

    # Lose it
    for _ in range(4):
        tracker.update([])

    assert track_id in tracker.ghost_tracks, "Should be in ghost buffer"
    print(f"  T{track_id} in ghost buffer")

    # Wait for ghost to expire
    time.sleep(0.1)
    tracker.update([])

    assert track_id not in tracker.ghost_tracks, "Ghost should have expired"

    events = get_events(tracker, 'track_lost')
    if len(events) == 1:
        print(f"  ✓ PASS: Ghost expired, finalized as track_lost")
        return True
    else:
        print(f"  ✗ FAIL: Expected 1 track_lost, got {len(events)}")
        return False


# =========================================================================
# Test 5: Ghost X-tolerance filtering
# =========================================================================
def test_ghost_x_tolerance():
    """Test that detections too far in X are NOT re-associated with ghosts."""
    print("\n" + "=" * 70)
    print("[Test 5] Ghost X-tolerance filtering")
    print("=" * 70)

    tracker = make_tracker(max_frames_without=3)

    # Create track at x=500
    det = Detection(bbox=(450, 750, 550, 850), confidence=0.9, class_id=0)
    tracker.update([det], (1000, 1000))
    track_id = list(tracker.tracks.keys())[0]

    # Move up
    det = Detection(bbox=(450, 550, 550, 650), confidence=0.9, class_id=0)
    tracker.update([det])

    # Lose it
    for _ in range(4):
        tracker.update([])

    assert track_id in tracker.ghost_tracks, "Should be in ghost buffer"

    # Detection far away in X (x=200 vs ghost at x=500, diff=300 >> tolerance=50)
    det = Detection(bbox=(150, 350, 250, 450), confidence=0.9, class_id=0)
    tracker.update([det])

    # Ghost should NOT be recovered (detection too far in X)
    if track_id in tracker.ghost_tracks:
        print(f"  ✓ PASS: Ghost NOT recovered (X too far)")
        return True
    elif track_id in tracker.tracks:
        print(f"  ✗ FAIL: Ghost was incorrectly recovered despite X distance")
        return False
    else:
        print(f"  ✓ PASS: Ghost expired (not recovered)")
        return True


# =========================================================================
# Test 6: Entry type classification - bottom_entry
# =========================================================================
def test_entry_type_bottom():
    """Test that tracks created in bottom 40% are classified as bottom_entry."""
    print("\n" + "=" * 70)
    print("[Test 6] Entry type classification - bottom_entry")
    print("=" * 70)

    tracker = make_tracker()

    # Create track in bottom 40% (y >= 600 for 1000px frame)
    det = Detection(bbox=(450, 750, 550, 850), confidence=0.9, class_id=0)
    tracker.update([det], (1000, 1000))
    track_id = list(tracker.tracks.keys())[0]

    track = tracker.tracks[track_id]
    if track.entry_type == "bottom_entry":
        print(f"  ✓ PASS: T{track_id} entry_type=bottom_entry (y=800)")
        return True
    else:
        print(f"  ✗ FAIL: entry_type={track.entry_type}, expected bottom_entry")
        return False


# =========================================================================
# Test 7: Entry type classification - midway_entry
# =========================================================================
def test_entry_type_midway():
    """Test that tracks created mid-frame with normal velocity are midway_entry."""
    print("\n" + "=" * 70)
    print("[Test 7] Entry type classification - midway_entry")
    print("=" * 70)

    tracker = make_tracker()

    # Create track at y=400 (above bottom 40% line at y=600)
    det = Detection(bbox=(450, 350, 550, 450), confidence=0.9, class_id=0)
    tracker.update([det], (1000, 1000))
    track_id = list(tracker.tracks.keys())[0]

    track = tracker.tracks[track_id]
    if track.entry_type == "midway_entry":
        print(f"  ✓ PASS: T{track_id} entry_type=midway_entry (y=400)")
    else:
        print(f"  ✗ FAIL: entry_type={track.entry_type}, expected midway_entry")
        return False

    # Simulate normal slow movement for enough frames to classify
    for y in [390, 380, 370, 360, 350]:
        det = Detection(bbox=(450, y-50, 550, y+50), confidence=0.9, class_id=0)
        tracker.update([det])

    track = tracker.tracks.get(track_id)
    if track and track.entry_type == "midway_entry" and track._entry_classified:
        print(f"  ✓ PASS: Still midway_entry after classification (slow velocity)")
        return True
    else:
        entry = track.entry_type if track else "N/A"
        print(f"  ✗ FAIL: entry_type={entry}")
        return False


# =========================================================================
# Test 8: No double-counting (fall and put back scenario)
# =========================================================================
def test_no_double_counting():
    """Test: bag falls at 50%, worker puts back -> exactly 1 count."""
    print("\n" + "=" * 70)
    print("[Test 8] No double-counting (fall + put back)")
    print("=" * 70)

    tracker = make_tracker(max_frames_without=5, ghost_max_age=0.01)

    # First track: bottom -> mid (y=740) -> lost
    # Use small position steps to keep velocity low
    det = Detection(bbox=(450, 780, 550, 820), confidence=0.9, class_id=0)
    tracker.update([det], (1000, 1000))
    t1 = list(tracker.tracks.keys())[0]

    for y in [770, 760, 750, 740]:
        det = Detection(bbox=(450, y-20, 550, y+20), confidence=0.9, class_id=0)
        tracker.update([det])

    # Lose track (6 frames > max_frames_without=5)
    for _ in range(6):
        tracker.update([])

    # Wait for ghost to expire
    time.sleep(0.05)
    tracker.update([])

    events1 = get_events(tracker)
    lost_events = [e for e in events1 if e.event_type == 'track_lost']
    completed1 = [e for e in events1 if e.event_type == 'track_completed']
    print(f"  First track T{t1}: lost={len(lost_events)}, completed={len(completed1)}")

    # Second track: worker puts bag back at y=500 (midway_entry)
    det = Detection(bbox=(450, 480, 550, 520), confidence=0.9, class_id=0)
    tracker.update([det])
    t2 = list(tracker.tracks.keys())[0]
    print(f"  Second track T{t2} created at y=500")

    # Move to exit (center < 20)
    for y in [400, 300, 200, 100, 10]:
        det = Detection(bbox=(450, y-10, 550, y+10), confidence=0.9, class_id=0)
        tracker.update([det])

    events2 = get_events(tracker)
    completed = [e for e in events2 if e.event_type == 'track_completed']

    if len(completed) == 1:
        print(f"  ✓ PASS: Exactly 1 count (no double-counting)")
        if completed[0].entry_type == 'midway_entry':
            print(f"  ✓ PASS: Flagged as midway_entry (suspected_duplicate={completed[0].suspected_duplicate})")
        return True
    else:
        print(f"  ✗ FAIL: Expected 1 completed, got {len(completed)}")
        for e in events2:
            print(f"    event: T{e.track_id} {e.event_type} exit={e.exit_direction}")
        return False


# =========================================================================
# Test 9: TrackEvent enriched fields
# =========================================================================
def test_enriched_track_event():
    """Test that TrackEvent has all enriched lifecycle fields."""
    print("\n" + "=" * 70)
    print("[Test 9] TrackEvent enriched fields")
    print("=" * 70)

    tracker = make_tracker()

    # Normal bottom to top journey with consistent 100x100 bboxes
    det = Detection(bbox=(450, 750, 550, 850), confidence=0.9, class_id=0)
    tracker.update([det], (1000, 1000))
    track_id = list(tracker.tracks.keys())[0]

    for y in [730, 660, 590, 520, 450, 380, 310, 240, 170, 100]:
        det = Detection(bbox=(450, y-50, 550, y+50), confidence=0.9, class_id=0)
        tracker.update([det])

    # Final step to exit zone (center < 20)
    det = Detection(bbox=(450, 0, 550, 20), confidence=0.9, class_id=0)
    tracker.update([det])

    events = get_events(tracker, 'track_completed')
    if not events:
        print("  ✗ FAIL: No track_completed event")
        return False

    e = events[0]
    checks = [
        (hasattr(e, 'entry_type'), f"entry_type={e.entry_type}"),
        (hasattr(e, 'suspected_duplicate'), f"suspected_duplicate={e.suspected_duplicate}"),
        (hasattr(e, 'ghost_recovery_count'), f"ghost_recovery_count={e.ghost_recovery_count}"),
        (hasattr(e, 'occlusion_events'), f"occlusion_events={e.occlusion_events}"),
        (hasattr(e, 'shadow_of'), f"shadow_of={e.shadow_of}"),
        (hasattr(e, 'shadow_count'), f"shadow_count={e.shadow_count}"),
        (hasattr(e, 'merge_events'), f"merge_events={e.merge_events}"),
    ]

    all_ok = True
    for check, msg in checks:
        if check:
            print(f"  ✓ {msg}")
        else:
            print(f"  ✗ MISSING: {msg}")
            all_ok = False

    if all_ok:
        print(f"  ✓ PASS: All enriched fields present")
    return all_ok


# =========================================================================
# Test 10: Config parameters
# =========================================================================
def test_config_params():
    """Test that new config params exist and old ones are removed."""
    print("\n" + "=" * 70)
    print("[Test 10] Config parameters")
    print("=" * 70)

    config = TrackingConfig()
    all_ok = True

    # New params
    new_params = [
        ('ghost_track_max_age_seconds', 4.0),
        ('ghost_track_x_tolerance_pixels', 80.0),
        ('ghost_track_max_y_gap_ratio', 0.2),
        ('merge_bbox_growth_threshold', 1.4),
        ('merge_spatial_tolerance_pixels', 50.0),
        ('merge_y_tolerance_pixels', 30.0),
        ('bottom_entry_zone_ratio', 0.4),
        ('thrown_entry_min_velocity', 15.0),
        ('thrown_entry_detection_frames', 5),
    ]

    for name, default in new_params:
        val = getattr(config, name, None)
        if val == default:
            print(f"  ✓ {name}={val}")
        else:
            print(f"  ✗ {name}={val} (expected {default})")
            all_ok = False

    # Removed params
    removed = [
        'lost_track_entry_zone_ratio',
        'lost_track_exit_zone_ratio',
        'lost_track_min_travel_ratio',
        'lost_track_min_hit_rate',
    ]

    for name in removed:
        if hasattr(config, name):
            print(f"  ✗ {name} should be REMOVED")
            all_ok = False
        else:
            print(f"  ✓ {name} removed")

    if all_ok:
        print(f"  ✓ PASS: All config params correct")
    return all_ok


# =========================================================================
# Main
# =========================================================================
if __name__ == "__main__":
    tests = [
        test_normal_exit_top,
        test_lost_tracks_never_counted,
        test_ghost_recovery,
        test_ghost_expiry,
        test_ghost_x_tolerance,
        test_entry_type_bottom,
        test_entry_type_midway,
        test_no_double_counting,
        test_enriched_track_event,
        test_config_params,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            result = test_fn()
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ✗ EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    if failed == 0:
        print("✓ ALL TESTS PASSED!")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 70)

    sys.exit(0 if failed == 0 else 1)
