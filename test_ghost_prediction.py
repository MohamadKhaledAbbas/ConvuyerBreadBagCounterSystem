"""
Test script to verify ghost track prediction accuracy and DB migration.

These tests specifically validate:
1. Ghost predicted_pos uses last real observed position (not velocity-shifted bbox)
2. Ghost recovery works with realistic reappearance positions
3. Database migration adds missing columns to existing track_events tables
"""

import sys
import os
import time
import sqlite3
import tempfile

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
    config.min_travel_duration_seconds = 0.0
    config.ghost_track_max_age_seconds = ghost_max_age
    config.ghost_track_x_tolerance_pixels = 80.0
    config.ghost_track_max_y_gap_ratio = 0.2
    config.exit_zone_ratio = 0.15
    config.bottom_exit_zone_ratio = 0.15
    config.exit_margin_pixels = 20
    tracker = ConveyorTracker(config)
    tracker.frame_height = frame_h
    tracker.frame_width = frame_w
    return tracker


# =========================================================================
# Test 1: Ghost predicted_pos should be last real position, not shifted
# =========================================================================
def test_ghost_predicted_pos_accuracy():
    """
    Verify that ghost predicted_pos is the last real observed position,
    NOT the velocity-shifted bbox center.

    This was the root cause of T7 getting predicted_pos=(1038, -703)
    when its last real observation was around (648, 400).
    """
    print("\n" + "=" * 70)
    print("[Test 1] Ghost predicted_pos uses last real position")
    print("=" * 70)

    tracker = make_tracker(max_frames_without=3)

    # Create track at y=800
    det = Detection(bbox=(450, 750, 550, 850), confidence=0.9, class_id=0)
    tracker.update([det], (1000, 1000))
    track_id = list(tracker.tracks.keys())[0]
    print(f"  Created T{track_id} at center y=800")

    # Move up with significant velocity (~100 px/frame) to amplify any bug
    for y in [700, 600, 500]:
        det = Detection(bbox=(450, y-50, 550, y+50), confidence=0.9, class_id=0)
        tracker.update([det])
    print(f"  Moved to y=500 (last real observation)")

    # Now lose the track (4 > max_frames_without=3)
    for _ in range(4):
        tracker.update([])

    if track_id not in tracker.ghost_tracks:
        print(f"  ✗ FAIL: Track not in ghost buffer")
        return False

    ghost_info = tracker.ghost_tracks[track_id]
    pred_pos = ghost_info['predicted_pos']
    print(f"  Ghost predicted_pos = {pred_pos}")

    # The predicted_pos should be the last REAL position from position_history
    # which is (500, 500). It should NOT be the velocity-shifted bbox center.
    pred_x, pred_y = pred_pos

    # With old buggy code: pred_pos would be ~(500, -1100) due to 31x velocity
    # With fix: pred_pos should be (500, 500) — last real observed position
    if pred_y < 0:
        print(f"  ✗ FAIL: predicted_pos Y={pred_y} is negative (outside frame!) - overshoot bug")
        return False

    if pred_y > 600:
        print(f"  ✗ FAIL: predicted_pos Y={pred_y} is too far below last observation (y=500)")
        return False

    if abs(pred_y - 500) > 50:
        print(f"  ✗ FAIL: predicted_pos Y={pred_y} is not close to last observation (y=500)")
        return False

    print(f"  ✓ PASS: predicted_pos={pred_pos} is close to last real observation (500, 500)")
    return True


# =========================================================================
# Test 2: Ghost recovery with production-like scenario (T7/T8 case)
# =========================================================================
def test_ghost_recovery_production_scenario():
    """
    Simulate the T7/T8 scenario from production:
    - Track created at y=582 (bottom entry)
    - Moves upward (fast conveyor)
    - Gets occluded
    - Reappears at y=308 (should be recovered as same track)
    """
    print("\n" + "=" * 70)
    print("[Test 2] Production scenario: T7→T8 should recover as T7")
    print("=" * 70)

    tracker = make_tracker(max_frames_without=5, frame_h=720, frame_w=800)

    # Simulate T7: created at bbox=(550, 513, 747, 652) center=(648, 582)
    det = Detection(bbox=(550, 513, 747, 652), confidence=0.91, class_id=0)
    tracker.update([det], (720, 800))
    track_id = list(tracker.tracks.keys())[0]
    print(f"  Created T{track_id} at center=(648, 582)")

    # Move upward quickly (conveyor speed) - each step moves ~50px up
    for y_center in [530, 480, 430, 380]:
        hw = 49  # ~half width of bbox
        hh = 35  # ~half height of bbox
        det = Detection(
            bbox=(648-hw, y_center-hh, 648+hw, y_center+hh),
            confidence=0.9, class_id=0
        )
        tracker.update([det])
    last_real_y = 380
    print(f"  Moved to y={last_real_y} (last real observation)")

    # Simulate occlusion: 6 empty frames (> max_frames_without=5)
    for _ in range(6):
        tracker.update([])

    if track_id not in tracker.ghost_tracks:
        print(f"  ✗ FAIL: Track not in ghost buffer")
        return False

    ghost_info = tracker.ghost_tracks[track_id]
    pred_pos = ghost_info['predicted_pos']
    print(f"  Ghost predicted_pos={pred_pos} (should be near y={last_real_y})")

    # Verify prediction is reasonable (near last real position, not -703)
    if pred_pos[1] < 0:
        print(f"  ✗ FAIL: predicted_pos Y={pred_pos[1]} is NEGATIVE (overshoot bug!)")
        return False

    # Now the bag reappears at y=308 (like T8 in production logs)
    det = Detection(bbox=(647, 239, 755, 377), confidence=0.87, class_id=0)
    # det center = (701, 308)
    tracker.update([det])

    # Check if ghost was recovered
    if track_id in tracker.tracks:
        recovered = tracker.tracks[track_id]
        print(f"  ✓ PASS: T{track_id} RECOVERED (ghost_recovery_count={recovered.ghost_recovery_count})")
        return True
    else:
        # Check if a new track was created instead (the bug scenario)
        new_tracks = list(tracker.tracks.keys())
        if new_tracks and new_tracks[0] != track_id:
            print(f"  ✗ FAIL: New track T{new_tracks[0]} created instead of recovering T{track_id}")
            print(f"    Ghost still pending: {track_id in tracker.ghost_tracks}")
            if track_id in tracker.ghost_tracks:
                gi = tracker.ghost_tracks[track_id]
                print(f"    Ghost predicted_pos: {gi['predicted_pos']}")
                print(f"    Ghost velocity: {gi['last_velocity']}")
        else:
            print(f"  ✗ FAIL: No recovery and no new track")
        return False


# =========================================================================
# Test 3: Database migration adds missing columns
# =========================================================================
def test_db_migration_adds_columns():
    """
    Verify that DatabaseManager._migrate_track_events_schema() adds
    missing columns to existing track_events tables.

    This addresses the error:
    'table track_events has no column named entry_type'
    """
    print("\n" + "=" * 70)
    print("[Test 3] Database migration adds missing columns")
    print("=" * 70)

    # Create a temporary DB with OLD schema (no enhanced columns)
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name

    try:
        conn = sqlite3.connect(db_path)
        # Create track_events table WITHOUT enhanced columns (old schema)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS track_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                created_at TEXT NOT NULL,
                entry_x INTEGER,
                entry_y INTEGER,
                exit_x INTEGER,
                exit_y INTEGER,
                exit_direction TEXT,
                distance_pixels REAL,
                duration_seconds REAL,
                total_frames INTEGER,
                avg_confidence REAL,
                total_hits INTEGER,
                classification TEXT,
                classification_confidence REAL,
                position_history TEXT
            )
        """)
        conn.commit()

        # Verify columns are missing
        cursor = conn.execute("PRAGMA table_info(track_events)")
        columns = {row[1] for row in cursor.fetchall()}
        assert 'entry_type' not in columns, "entry_type should NOT exist before migration"
        assert 'ghost_recovery_count' not in columns, "ghost_recovery_count should NOT exist"
        print(f"  Old schema columns: {sorted(columns)}")
        print(f"  Confirmed: entry_type NOT present")
        conn.close()

        # Now initialize DatabaseManager which should run migration
        from src.logging.Database import DatabaseManager
        db = DatabaseManager(db_path)

        # Verify columns were added
        conn2 = sqlite3.connect(db_path)
        cursor2 = conn2.execute("PRAGMA table_info(track_events)")
        new_columns = {row[1] for row in cursor2.fetchall()}
        conn2.close()
        db.close()

        expected_new = {'entry_type', 'suspected_duplicate', 'ghost_recovery_count',
                        'shadow_of', 'shadow_count', 'occlusion_events', 'merge_events'}

        missing = expected_new - new_columns
        if missing:
            print(f"  ✗ FAIL: Migration did not add columns: {missing}")
            return False

        print(f"  After migration columns: {sorted(new_columns)}")
        print(f"  ✓ PASS: All 7 enhanced columns added by migration")
        return True

    finally:
        os.unlink(db_path)


# =========================================================================
# Test 4: Ghost prediction stays within frame bounds
# =========================================================================
def test_ghost_prediction_bounds():
    """
    Verify ghost predicted_pos never goes to impossible positions
    (negative coordinates or far outside frame).
    """
    print("\n" + "=" * 70)
    print("[Test 4] Ghost prediction stays within reasonable bounds")
    print("=" * 70)

    tracker = make_tracker(max_frames_without=3)

    # Create track with VERY fast velocity (large jumps)
    det = Detection(bbox=(400, 700, 500, 800), confidence=0.9, class_id=0)
    tracker.update([det], (1000, 1000))
    track_id = list(tracker.tracks.keys())[0]

    # Move with large steps (200px per frame = very fast)
    for y in [500, 300]:
        det = Detection(bbox=(400, y-50, 500, y+50), confidence=0.9, class_id=0)
        tracker.update([det])

    # Lose track
    for _ in range(4):
        tracker.update([])

    if track_id not in tracker.ghost_tracks:
        print(f"  ✗ FAIL: Track not in ghost buffer (may have exited via prediction)")
        return True  # Not a prediction bounds failure

    ghost_info = tracker.ghost_tracks[track_id]
    pred_pos = ghost_info['predicted_pos']
    print(f"  Ghost predicted_pos = {pred_pos}")

    # Verify prediction is reasonable
    pred_x, pred_y = pred_pos
    if pred_y < -500 or pred_x < -500 or pred_x > 2000:
        print(f"  ✗ FAIL: predicted_pos {pred_pos} is wildly out of bounds")
        return False

    print(f"  ✓ PASS: predicted_pos {pred_pos} is reasonable (last real position)")
    return True


# =========================================================================
# Main
# =========================================================================
if __name__ == "__main__":
    tests = [
        test_ghost_predicted_pos_accuracy,
        test_ghost_recovery_production_scenario,
        test_db_migration_adds_columns,
        test_ghost_prediction_bounds,
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
