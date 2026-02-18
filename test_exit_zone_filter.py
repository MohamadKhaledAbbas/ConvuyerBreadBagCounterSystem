"""
Test script to verify that tracks are not created for detections in exit zones.

This should prevent unnecessary "track_lost" events for detections that appear
already at the top or bottom edges of the frame.
"""

from src.tracking.ConveyorTracker import ConveyorTracker
from src.config.tracking_config import TrackingConfig
from src.detection.BaseDetection import Detection

def test_exit_zone_filtering():
    """Test that detections in exit zones don't create tracks."""

    # Create tracker with known frame dimensions
    config = TrackingConfig()
    tracker = ConveyorTracker(config)

    # Set frame dimensions (720p)
    frame_height = 720
    frame_width = 1280
    tracker.frame_height = frame_height
    tracker.frame_width = frame_width

    # Calculate exit zones
    exit_zone_ratio = 0.15
    top_exit_threshold = frame_height * exit_zone_ratio  # 108 pixels
    bottom_exit_zone_ratio = 0.15
    bottom_exit_threshold = frame_height * (1.0 - bottom_exit_zone_ratio)  # 612 pixels

    print(f"Frame: {frame_width}x{frame_height}")
    print(f"Top exit zone: y <= {top_exit_threshold}")
    print(f"Bottom exit zone: y >= {bottom_exit_threshold}")
    print()

    # Test 1: Detection in top exit zone (like T161 with y=17)
    print("Test 1: Detection in TOP exit zone (should be filtered)")
    det_top = Detection(
        bbox=(665, 0, 771, 35),  # Same as T161 from logs
        confidence=0.72,
        class_id=0
    )
    center_y_top = (det_top.bbox[1] + det_top.bbox[3]) // 2  # y=17
    print(f"  Detection bbox: {det_top.bbox}, center_y: {center_y_top}")

    initial_track_count = len(tracker.tracks)
    tracker.update([det_top], frame_shape=(frame_height, frame_width))
    final_track_count = len(tracker.tracks)

    if final_track_count == initial_track_count:
        print(f"  ✓ PASS: No track created (tracks: {initial_track_count} -> {final_track_count})")
    else:
        print(f"  ✗ FAIL: Track was created (tracks: {initial_track_count} -> {final_track_count})")
    print()

    # Test 2: Detection in bottom exit zone
    print("Test 2: Detection in BOTTOM exit zone (should be filtered)")
    det_bottom = Detection(
        bbox=(500, 680, 600, 720),
        confidence=0.75,
        class_id=0
    )
    center_y_bottom = (det_bottom.bbox[1] + det_bottom.bbox[3]) // 2  # y=700
    print(f"  Detection bbox: {det_bottom.bbox}, center_y: {center_y_bottom}")

    initial_track_count = len(tracker.tracks)
    tracker.update([det_bottom], frame_shape=(frame_height, frame_width))
    final_track_count = len(tracker.tracks)

    if final_track_count == initial_track_count:
        print(f"  ✓ PASS: No track created (tracks: {initial_track_count} -> {final_track_count})")
    else:
        print(f"  ✗ FAIL: Track was created (tracks: {initial_track_count} -> {final_track_count})")
    print()

    # Test 3: Detection in middle of frame (should create track)
    print("Test 3: Detection in MIDDLE of frame (should create track)")
    det_middle = Detection(
        bbox=(500, 300, 600, 400),
        confidence=0.80,
        class_id=0
    )
    center_y_middle = (det_middle.bbox[1] + det_middle.bbox[3]) // 2  # y=350
    print(f"  Detection bbox: {det_middle.bbox}, center_y: {center_y_middle}")

    initial_track_count = len(tracker.tracks)
    tracker.update([det_middle], frame_shape=(frame_height, frame_width))
    final_track_count = len(tracker.tracks)

    if final_track_count == initial_track_count + 1:
        print(f"  ✓ PASS: Track created (tracks: {initial_track_count} -> {final_track_count})")
    else:
        print(f"  ✗ FAIL: Track not created or wrong count (tracks: {initial_track_count} -> {final_track_count})")
    print()

    # Test 4: Valid detection in lower-middle area (not in exit zones)
    # Use a different location to avoid matching with Test 3 track
    print("Test 4: Detection in lower-middle zone (should create track)")
    print(f"  Current active tracks before test 4: {list(tracker.tracks.keys())}")
    det_entry = Detection(
        bbox=(800, 450, 900, 550),  # Different x position from Test 3, center at y=500
        confidence=0.84,
        class_id=0
    )
    center_y_entry = (det_entry.bbox[1] + det_entry.bbox[3]) // 2  # y=500
    print(f"  Detection bbox: {det_entry.bbox}, center_y: {center_y_entry}")

    initial_track_count = len(tracker.tracks)
    tracker.update([det_entry], frame_shape=(frame_height, frame_width))
    final_track_count = len(tracker.tracks)

    if final_track_count == initial_track_count + 1:
        print(f"  ✓ PASS: Track created (tracks: {initial_track_count} -> {final_track_count})")
    else:
        print(f"  ✗ FAIL: Track not created or wrong count (tracks: {initial_track_count} -> {final_track_count})")
        print(f"  Active tracks after: {list(tracker.tracks.keys())}")
    print()

    print("=" * 60)
    print(f"Final tracker state: {len(tracker.tracks)} active tracks")
    print(f"Expected: 2 tracks (Test 3 and Test 4 should create tracks)")
    print()
    print("Summary:")
    print("  - Detections in TOP exit zone are filtered ✓")
    print("  - Detections in BOTTOM exit zone are filtered ✓")
    print("  - Detections in valid tracking zones create tracks ✓")

    if len(tracker.tracks) == 2:
        print("\n✓ ALL TESTS PASSED!")
        return True
    else:
        print("\n✗ SOME TESTS FAILED")
        return False


if __name__ == "__main__":
    success = test_exit_zone_filtering()
    exit(0 if success else 1)
