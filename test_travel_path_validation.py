#!/usr/bin/env python3
"""
Tests for travel path validation in ConveyorTracker.

Validates the time-based approach for travel path validation:
1. Track must be visible for minimum duration (e.g., 2 seconds)
2. Track must exit from the top (not bottom)
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config.tracking_config import TrackingConfig
from src.detection.BaseDetection import Detection
from src.tracking.ConveyorTracker import ConveyorTracker


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
        'min_travel_duration_seconds': 2.0,
        'exit_zone_ratio': 0.15,
        'bottom_exit_zone_ratio': 0.15,
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


def test_bottom_exit_zone_detection():
    """Test that _is_in_bottom_exit_zone correctly identifies bottom zone."""
    config = make_config()
    tracker = ConveyorTracker(config=config)
    tracker.frame_height = FRAME_H

    # Bottom 15% of 720 = Y >= 612
    assert tracker._is_in_bottom_exit_zone(700), "Y=700 should be in bottom exit zone"
    assert tracker._is_in_bottom_exit_zone(612), "Y=612 should be in bottom exit zone (boundary)"
    assert not tracker._is_in_bottom_exit_zone(400), "Y=400 should NOT be in bottom exit zone (mid-frame)"
    assert not tracker._is_in_bottom_exit_zone(100), "Y=100 should NOT be in bottom exit zone (top)"

    print("✓ test_bottom_exit_zone_detection passed")


def test_valid_travel_with_sufficient_time():
    """Test that a track with sufficient travel time exiting at top is marked 'track_completed'."""
    config = make_config(min_travel_duration_seconds=0.1)  # Use short time for testing
    tracker = ConveyorTracker(config=config)

    # Simulate a bag traveling from bottom to top
    y_positions = [680, 600, 500, 400, 300, 200, 100, 10]
    cx = 640

    for i, cy in enumerate(y_positions):
        det = make_detection(cx, cy)
        tracker.update([det], frame_shape=FRAME_SHAPE)
        if i < len(y_positions) - 1:
            time.sleep(0.02)  # Small delay to accumulate time

    # The track should exit at the top and be marked 'track_completed'
    events = tracker.get_completed_events()

    completed = [e for e in events if e.event_type == 'track_completed']
    invalid = [e for e in events if e.event_type == 'track_invalid']

    assert len(completed) == 1, f"Expected 1 track_completed, got {len(completed)}"
    assert len(invalid) == 0, f"Expected 0 track_invalid, got {len(invalid)}"

    print("✓ test_valid_travel_with_sufficient_time passed")


def test_invalid_travel_insufficient_time():
    """Test that a track with insufficient travel time is marked 'track_invalid'."""
    config = make_config(min_travel_duration_seconds=10.0)  # Require 10 seconds (won't be met)
    tracker = ConveyorTracker(config=config)

    # Simulate a bag traveling from bottom to top quickly
    y_positions = [680, 500, 300, 100, 10]
    cx = 640

    for cy in y_positions:
        det = make_detection(cx, cy)
        tracker.update([det], frame_shape=FRAME_SHAPE)

    events = tracker.get_completed_events()

    completed = [e for e in events if e.event_type == 'track_completed']
    invalid = [e for e in events if e.event_type == 'track_invalid']

    assert len(completed) == 0, f"Expected 0 track_completed (insufficient time), got {len(completed)}"
    assert len(invalid) == 1, f"Expected 1 track_invalid (insufficient time), got {len(invalid)}"

    print("✓ test_invalid_travel_insufficient_time passed")


def test_invalid_travel_exits_bottom():
    """Test that a track exiting from the bottom is marked 'track_invalid'."""
    config = make_config(min_travel_duration_seconds=0.1, min_track_duration_frames=3)
    tracker = ConveyorTracker(config=config)

    # Simulate a bag that starts mid-frame, moves down and exits at bottom
    # Need enough frames for track to be confirmed (min_track_duration_frames=3)
    # y=710 is within exit_margin_pixels=20 from bottom (720-20=700, so y>700 triggers exit)
    positions = [
        (640, 300), (640, 400), (640, 500), (640, 600), (640, 710)  # Exits bottom
    ]

    for i, (cx, cy) in enumerate(positions):
        det = make_detection(cx, cy)
        tracker.update([det], frame_shape=FRAME_SHAPE)
        if i < len(positions) - 1:
            time.sleep(0.05)

    events = tracker.get_completed_events()

    completed = [e for e in events if e.event_type == 'track_completed']
    invalid = [e for e in events if e.event_type == 'track_invalid']

    # Should be invalid since it exits from bottom, not top
    assert len(completed) == 0, f"Expected 0 track_completed (bottom exit), got {len(completed)}"
    assert len(invalid) == 1, f"Expected 1 track_invalid (bottom exit), got {len(invalid)}"

    print("✓ test_invalid_travel_exits_bottom passed")


def test_invalid_travel_exits_side():
    """Test that a track entering from bottom but exiting from the side is marked 'track_invalid'."""
    config = make_config(min_travel_duration_seconds=0.1)
    tracker = ConveyorTracker(config=config)

    # Simulate a bag drifting to exit from the right side
    positions = [
        (640, 680), (700, 600), (800, 500), (900, 400),
        (1000, 350), (1100, 300), (1270, 280)  # Exits right side
    ]

    for i, (cx, cy) in enumerate(positions):
        det = make_detection(cx, cy)
        tracker.update([det], frame_shape=FRAME_SHAPE)
        if i < len(positions) - 1:
            time.sleep(0.02)

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

    # Simulate a bag that appears and then disappears mid-frame
    det = make_detection(640, 680)
    tracker.update([det], frame_shape=FRAME_SHAPE)

    det = make_detection(640, 600)
    tracker.update([det], frame_shape=FRAME_SHAPE)

    det = make_detection(640, 500)
    tracker.update([det], frame_shape=FRAME_SHAPE)

    # Bag disappears (no detections) for max_frames_without_detection + 1 frames
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

    # Simulate a bag appearing mid-frame and exiting at top quickly
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


def test_time_based_validation_realistic():
    """Test time-based validation with mocked time for realistic scenario."""
    config = make_config(min_travel_duration_seconds=2.0)
    tracker = ConveyorTracker(config=config)

    start_time = time.time()

    # First detection - track creation
    det = make_detection(640, 680)
    tracker.update([det], frame_shape=FRAME_SHAPE)

    # Get the track and manually adjust its created_at to simulate 2.5 seconds ago
    track = list(tracker.tracks.values())[0]
    track.created_at = start_time - 2.5  # Simulate track was created 2.5s ago

    # Continue moving the track to the top
    for cy in [500, 300, 100, 10]:
        det = make_detection(640, cy)
        tracker.update([det], frame_shape=FRAME_SHAPE)

    events = tracker.get_completed_events()

    completed = [e for e in events if e.event_type == 'track_completed']
    invalid = [e for e in events if e.event_type == 'track_invalid']

    assert len(completed) == 1, f"Expected 1 track_completed (sufficient time), got {len(completed)}"
    assert len(invalid) == 0, f"Expected 0 track_invalid, got {len(invalid)}"

    # Verify duration is reported correctly
    event = completed[0]
    assert event.duration_seconds >= 2.0, f"Duration should be >= 2.0s, got {event.duration_seconds:.2f}s"

    print("✓ test_time_based_validation_realistic passed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Time-Based Travel Path Validation")
    print("=" * 60)
    print()

    tests = [
        test_exit_zone_detection,
        test_bottom_exit_zone_detection,
        test_valid_travel_with_sufficient_time,
        test_invalid_travel_insufficient_time,
        test_invalid_travel_exits_bottom,
        test_invalid_travel_exits_side,
        test_lost_track_not_classified,
        test_require_full_travel_disabled,
        test_entry_center_y_recorded,
        test_time_based_validation_realistic,
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
            import traceback
            traceback.print_exc()
            failed += 1

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
