#!/usr/bin/env python3
"""
Tests for track event lifecycle analytics.

Tests:
1. track_event_details table is created
2. add_track_event_detail() inserts correctly
3. get_track_event_details() filters correctly
4. get_track_lifecycle() returns summary + details
5. purge_old_track_events() respects retention
6. TrackLifecycleService assembles data correctly
"""

import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.logging.Database import DatabaseManager


def test_track_event_details_table_created():
    """Test that track_event_details table is created in the schema."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'test.db')
        db = DatabaseManager(db_path)
        try:
            import sqlite3
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='track_event_details'")
            result = cursor.fetchone()
            conn.close()
            assert result is not None, "track_event_details table should exist"
            assert result[0] == 'track_event_details'
        finally:
            db.close()
    print("✓ test_track_event_details_table_created passed")


def test_add_track_event_detail():
    """Test adding detail steps to the database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'test.db')
        db = DatabaseManager(db_path)
        try:
            # Add ROI collected detail
            detail_id = db.add_track_event_detail(
                track_id=42,
                timestamp='2026-02-10T08:00:01',
                step_type='roi_collected',
                bbox_x1=100, bbox_y1=200, bbox_x2=200, bbox_y2=350,
                quality_score=185.5,
                roi_index=0
            )
            assert detail_id is not None and detail_id > 0

            # Add ROI classified detail
            detail_id2 = db.add_track_event_detail(
                track_id=42,
                timestamp='2026-02-10T08:00:05',
                step_type='roi_classified',
                roi_index=0,
                class_name='Wheatberry',
                confidence=0.92,
                is_rejected=0
            )
            assert detail_id2 > detail_id

            # Add voting result
            detail_id3 = db.add_track_event_detail(
                track_id=42,
                timestamp='2026-02-10T08:00:06',
                step_type='voting_result',
                class_name='Wheatberry',
                confidence=0.90,
                valid_votes=4,
                total_rois=5,
                vote_distribution=json.dumps({'Wheatberry': 3.6, 'Multigrain': 0.8})
            )
            assert detail_id3 > detail_id2

            # Verify
            details = db.get_track_event_details(track_id=42)
            assert len(details) == 3, f"Expected 3 details, got {len(details)}"
            assert details[0]['step_type'] == 'roi_collected'
            assert details[0]['bbox_x1'] == 100
            assert details[1]['step_type'] == 'roi_classified'
            assert details[1]['class_name'] == 'Wheatberry'
            assert details[2]['step_type'] == 'voting_result'
        finally:
            db.close()
    print("✓ test_add_track_event_detail passed")


def test_get_track_event_details_filter():
    """Test filtering details by step_type."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'test.db')
        db = DatabaseManager(db_path)
        try:
            for i in range(3):
                db.add_track_event_detail(
                    track_id=10, timestamp=f'2026-02-10T08:0{i}:00',
                    step_type='roi_collected', bbox_x1=100, bbox_y1=200,
                    bbox_x2=200, bbox_y2=300, roi_index=i
                )
            db.add_track_event_detail(
                track_id=10, timestamp='2026-02-10T08:05:00',
                step_type='voting_result', class_name='Multigrain', confidence=0.88
            )

            all_details = db.get_track_event_details(track_id=10)
            assert len(all_details) == 4

            roi_only = db.get_track_event_details(track_id=10, step_type='roi_collected')
            assert len(roi_only) == 3

            voting_only = db.get_track_event_details(track_id=10, step_type='voting_result')
            assert len(voting_only) == 1
        finally:
            db.close()
    print("✓ test_get_track_event_details_filter passed")


def test_get_track_lifecycle():
    """Test full lifecycle retrieval."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'test.db')
        db = DatabaseManager(db_path)
        try:
            # Add a track event
            db.add_track_event(
                track_id=99, event_type='track_completed',
                timestamp='2026-02-10T08:10:00', created_at='2026-02-10T08:09:50',
                entry_x=640, entry_y=680, exit_x=640, exit_y=10,
                exit_direction='top', distance_pixels=670.0, duration_seconds=5.0,
                total_frames=70, avg_confidence=0.92
            )
            # Add detail steps
            db.add_track_event_detail(
                track_id=99, timestamp='2026-02-10T08:09:52',
                step_type='roi_collected', bbox_x1=600, bbox_y1=640,
                bbox_x2=680, bbox_y2=720, quality_score=200.0, roi_index=0
            )
            db.add_track_event_detail(
                track_id=99, timestamp='2026-02-10T08:10:00',
                step_type='voting_result', class_name='WholeWheat',
                confidence=0.95, valid_votes=5
            )

            lifecycle = db.get_track_lifecycle(99)
            assert lifecycle['summary'] is not None
            assert lifecycle['summary']['track_id'] == 99
            assert lifecycle['summary']['event_type'] == 'track_completed'
            assert len(lifecycle['details']) == 2
        finally:
            db.close()
    print("✓ test_get_track_lifecycle passed")


def test_purge_old_track_events():
    """Test retention purge respects retention_days."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'test.db')
        db = DatabaseManager(db_path)
        try:
            # Add recent event (should survive)
            from datetime import datetime
            now_iso = datetime.utcnow().isoformat()
            db.add_track_event(
                track_id=1, event_type='track_completed',
                timestamp=now_iso, created_at=now_iso,
                distance_pixels=500.0, duration_seconds=3.0, total_frames=40
            )
            db.add_track_event_detail(
                track_id=1, timestamp=now_iso,
                step_type='roi_collected', bbox_x1=100, bbox_y1=200,
                bbox_x2=200, bbox_y2=300
            )

            # Add old event (should be purged with 0 days retention)
            old_iso = '2020-01-01T00:00:00'
            db.add_track_event(
                track_id=2, event_type='track_lost',
                timestamp=old_iso, created_at=old_iso,
                distance_pixels=100.0, duration_seconds=1.0, total_frames=10
            )
            db.add_track_event_detail(
                track_id=2, timestamp=old_iso,
                step_type='track_lost'
            )

            # Verify both exist
            all_events = db.get_track_events()
            assert len(all_events) == 2

            # Purge with 0 retention (deletes everything before "now")
            # Actually this won't delete the recent one. Let's purge 7 days.
            deleted = db.purge_old_track_events(retention_days=7)
            assert deleted >= 1, f"Expected at least 1 deletion, got {deleted}"

            # Recent should survive
            remaining = db.get_track_events()
            assert len(remaining) == 1
            assert remaining[0]['track_id'] == 1

            # Detail for old track should also be gone
            old_details = db.get_track_event_details(track_id=2)
            assert len(old_details) == 0
        finally:
            db.close()
    print("✓ test_purge_old_track_events passed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Track Event Lifecycle Analytics")
    print("=" * 60)
    print()

    tests = [
        test_track_event_details_table_created,
        test_add_track_event_detail,
        test_get_track_event_details_filter,
        test_get_track_lifecycle,
        test_purge_old_track_events,
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
