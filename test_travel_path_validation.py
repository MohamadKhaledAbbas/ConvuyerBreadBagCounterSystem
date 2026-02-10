#!/usr/bin/env python3
"""
Tests for travel path validation in ConveyorTracker.

Validates the requirement that bread bags must travel from bottom to top
on the conveyor belt to be counted and classified.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config.tracking_config import TrackingConfig
from src.detection.BaseDetection import Detection
from src.tracking.ConveyorTracker import ConveyorTracker, TrackedObject


# Frame dimensions: 1280x720 (720p)
FRAME_W = 1280
FRAME_H = 720
FRAME_SHAPE = (FRAME_H, FRAME_W)


def make_detection(cx, cy, w=60, h=80, confidence=0.9):
    """Helper to create a Detection from center coordinates."""
    x1 = cx - w // 2
    y1 = cy - h // 2
    x2 = cx + w // 2
    y2 = cy + h // 2
    return Detection(bbox=(x1, y1, x2, y2), confidence=confidence)


def make_config(**overrides):
    """Create a TrackingConfig with test-friendly defaults."""
    defaults = {
        'require_full_travel': True,
        'entry_zone_ratio': 0.25,
        'exit_zone_ratio': 0.15,
        'min_track_duration_frames': 3,
        'max_frames_without_detection': 5,
        'exit_margin_pixels': 20,
        'iou_threshold': 0.3,
        'min_confidence_new_track': 0.5,
    }
    defaults.update(overrides)
    config = TrackingConfig()
    for key, val in defaults.items():
        setattr(config, key, val)
    return config


def test_entry_zone_detection():
    """Test that _is_in_entry_zone correctly identifies bottom zone."""
    config = make_config()
    tracker = ConveyorTracker(config=config)
    tracker.frame_height = FRAME_H

    # Bottom 25% of 720 = Y >= 540
    assert tracker._is_in_entry_zone(700), "Y=700 should be in entry zone (bottom)"
    assert tracker._is_in_entry_zone(540), "Y=540 should be in entry zone (boundary)"
    assert not tracker._is_in_entry_zone(400), "Y=400 should NOT be in entry zone (mid-frame)"
    assert not tracker._is_in_entry_zone(100), "Y=100 should NOT be in entry zone (top)"

    print("✓ test_entry_zone_detection passed")


def test_exit_zone_detection():
    """Test that _is_in_exit_zone correctly identifies top zone."""
    config = make_config()
    tracker = ConveyorTracker(config=config)
    tracker.frame_height = FRAME_H

    # Top 15% of 720 = Y <= 108
    assert tracker._is_in_exit_zone(10), "Y=10 should be in exit zone (top)"
    assert tracker._is_in_exit_zone(108), "Y=108 should be in exit zone (boundary)"
    assert not tracker._is_in_exit_zone(400), "Y=400 should NOT be in exit zone (mid-frame)"
    assert not tracker._is_in_exit_zone(700), "Y=700 should NOT be in exit zone (bottom)"

    print("✓ test_exit_zone_detection passed")


def test_valid_travel_bottom_to_top():
    """Test that a track traveling from bottom to top is marked 'track_completed'."""
    config = make_config()
    tracker = ConveyorTracker(config=config)

    # Simulate a bag traveling from bottom (Y=680) to top (Y=10)
    # Step 1: Bag appears at bottom
    y_positions = [680, 600, 500, 400, 300, 200, 100, 10]
    cx = 640

    for i, cy in enumerate(y_positions):
        det = make_detection(cx, cy)
        tracker.update([det], frame_shape=FRAME_SHAPE)

    # The track should exit at the top and be marked 'track_completed'
    events = tracker.get_completed_events()

    completed = [e for e in events if e.event_type == 'track_completed']
    invalid = [e for e in events if e.event_type == 'track_invalid']

    assert len(completed) == 1, f"Expected 1 track_completed, got {len(completed)}"
    assert len(invalid) == 0, f"Expected 0 track_invalid, got {len(invalid)}"

    print("✓ test_valid_travel_bottom_to_top passed")


def test_invalid_travel_mid_frame_entry():
    """Test that a track appearing mid-frame is marked 'track_invalid'."""
    config = make_config()
    tracker = ConveyorTracker(config=config)

    # Simulate a bag appearing at mid-frame (Y=400) and exiting at top (Y=10)
    y_positions = [400, 300, 200, 100, 10]
    cx = 640

    for cy in y_positions:
        det = make_detection(cx, cy)
        tracker.update([det], frame_shape=FRAME_SHAPE)

    events = tracker.get_completed_events()

    completed = [e for e in events if e.event_type == 'track_completed']
    invalid = [e for e in events if e.event_type == 'track_invalid']

    assert len(completed) == 0, f"Expected 0 track_completed (mid-frame entry), got {len(completed)}"
    assert len(invalid) == 1, f"Expected 1 track_invalid, got {len(invalid)}"

    print("✓ test_invalid_travel_mid_frame_entry passed")


def test_invalid_travel_exits_side():
    """Test that a track entering from bottom but exiting from the side is marked 'track_invalid'."""
    config = make_config()
    tracker = ConveyorTracker(config=config)

    # Simulate a bag starting at bottom and drifting to exit from the right side
    positions = [
        (640, 680), (700, 600), (800, 500), (900, 400),
        (1000, 350), (1100, 300), (1270, 280)  # Exits right side
    ]

    for cx, cy in positions:
        det = make_detection(cx, cy)
        tracker.update([det], frame_shape=FRAME_SHAPE)

    events = tracker.get_completed_events()

    completed = [e for e in events if e.event_type == 'track_completed']
    invalid = [e for e in events if e.event_type == 'track_invalid']

    assert len(completed) == 0, f"Expected 0 track_completed (side exit), got {len(completed)}"
    assert len(invalid) == 1, f"Expected 1 track_invalid, got {len(invalid)}"

    print("✓ test_invalid_travel_exits_side passed")


def test_lost_track_not_classified():
    """Test that a lost track (disappeared mid-frame) generates 'track_lost' event."""
    config = make_config(max_frames_without_detection=3)
    tracker = ConveyorTracker(config=config)

    # Simulate a bag that appears at bottom and then disappears mid-frame
    # Step 1: Bag appears at bottom
    det = make_detection(640, 680)
    tracker.update([det], frame_shape=FRAME_SHAPE)

    det = make_detection(640, 600)
    tracker.update([det], frame_shape=FRAME_SHAPE)

    det = make_detection(640, 500)
    tracker.update([det], frame_shape=FRAME_SHAPE)

    # Step 2: Bag disappears (no detections) for max_frames_without_detection + 1 frames
    for _ in range(4):
        tracker.update([], frame_shape=FRAME_SHAPE)

    events = tracker.get_completed_events()

    lost = [e for e in events if e.event_type == 'track_lost']
    completed = [e for e in events if e.event_type == 'track_completed']

    assert len(lost) == 1, f"Expected 1 track_lost, got {len(lost)}"
    assert len(completed) == 0, f"Expected 0 track_completed, got {len(completed)}"

    print("✓ test_lost_track_not_classified passed")


def test_require_full_travel_disabled():
    """Test that disabling require_full_travel allows all tracks to be 'track_completed'."""
    config = make_config(require_full_travel=False)
    tracker = ConveyorTracker(config=config)

    # Simulate a bag appearing mid-frame and exiting at top
    y_positions = [400, 300, 200, 100, 10]
    cx = 640

    for cy in y_positions:
        det = make_detection(cx, cy)
        tracker.update([det], frame_shape=FRAME_SHAPE)

    events = tracker.get_completed_events()

    completed = [e for e in events if e.event_type == 'track_completed']
    invalid = [e for e in events if e.event_type == 'track_invalid']

    assert len(completed) == 1, f"Expected 1 track_completed (validation disabled), got {len(completed)}"
    assert len(invalid) == 0, f"Expected 0 track_invalid, got {len(invalid)}"

    print("✓ test_require_full_travel_disabled passed")


def test_entry_center_y_recorded():
    """Test that entry_center_y is recorded when a track is created."""
    config = make_config()
    tracker = ConveyorTracker(config=config)

    det = make_detection(640, 680)
    tracker.update([det], frame_shape=FRAME_SHAPE)

    tracks = list(tracker.tracks.values())
    assert len(tracks) == 1, f"Expected 1 track, got {len(tracks)}"

    track = tracks[0]
    assert track.entry_center_y is not None, "entry_center_y should be set"
    assert track.entry_center_y == 680, f"entry_center_y should be 680, got {track.entry_center_y}"

    print("✓ test_entry_center_y_recorded passed")


def test_invalid_travel_exits_bottom():
    """Test that a track exiting from the bottom is marked 'track_invalid'."""
    config = make_config()
    tracker = ConveyorTracker(config=config)

    # Simulate a bag that enters at bottom, moves up a bit, then goes back down
    positions = [
        (640, 680), (640, 620), (640, 560),
        (640, 620), (640, 680), (640, 710)  # Exits bottom
    ]

    for cx, cy in positions:
        det = make_detection(cx, cy)
        tracker.update([det], frame_shape=FRAME_SHAPE)

    events = tracker.get_completed_events()

    completed = [e for e in events if e.event_type == 'track_completed']
    invalid = [e for e in events if e.event_type == 'track_invalid']

    # Should be invalid since it exits from bottom, not top
    assert len(completed) == 0, f"Expected 0 track_completed (bottom exit), got {len(completed)}"

    print("✓ test_invalid_travel_exits_bottom passed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Travel Path Validation")
    print("=" * 60)
    print()

    tests = [
        test_entry_zone_detection,
        test_exit_zone_detection,
        test_valid_travel_bottom_to_top,
        test_invalid_travel_mid_frame_entry,
        test_invalid_travel_exits_side,
        test_lost_track_not_classified,
        test_require_full_travel_disabled,
        test_entry_center_y_recorded,
        test_invalid_travel_exits_bottom,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test_fn.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test_fn.__name__} ERROR: {e}")
            failed += 1

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
