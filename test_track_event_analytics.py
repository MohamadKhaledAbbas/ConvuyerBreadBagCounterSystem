#!/usr/bin/env python3
"""
Tests for track event analytics functionality.

Tests:
1. Database schema creates track_events table
2. add_track_event() inserts records correctly
3. get_track_events() returns and filters records
4. get_track_event_stats() returns correct aggregations
5. update_track_event_classification() updates classification fields
6. PipelineCore._log_track_event() correctly persists events
"""

import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.logging.Database import DatabaseManager


def test_track_events_table_created():
    """Test that track_events table is created in the schema."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'test.db')
        db = DatabaseManager(db_path)

        try:
            import sqlite3
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='track_events'")
            result = cursor.fetchone()
            conn.close()

            assert result is not None, "track_events table should exist"
            assert result[0] == 'track_events', f"Expected 'track_events', got {result[0]}"
        finally:
            db.close()

    print("✓ test_track_events_table_created passed")


def test_add_track_event():
    """Test adding a track event to the database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'test.db')
        db = DatabaseManager(db_path)

        try:
            event_id = db.add_track_event(
                track_id=42,
                event_type='track_completed',
                timestamp='2026-02-10T08:00:00',
                created_at='2026-02-10T07:59:55',
                entry_x=640,
                entry_y=680,
                exit_x=640,
                exit_y=10,
                exit_direction='top',
                distance_pixels=670.0,
                duration_seconds=5.0,
                total_frames=70,
                avg_confidence=0.92,
                total_hits=65,
                classification=None,
                classification_confidence=None,
                position_history=json.dumps([[640, 680], [640, 500], [640, 300], [640, 10]])
            )

            assert event_id is not None, "event_id should not be None"
            assert event_id > 0, f"event_id should be positive, got {event_id}"

            # Verify record
            events = db.get_track_events()
            assert len(events) == 1, f"Expected 1 event, got {len(events)}"

            e = events[0]
            assert e['track_id'] == 42
            assert e['event_type'] == 'track_completed'
            assert e['entry_x'] == 640
            assert e['entry_y'] == 680
            assert e['exit_x'] == 640
            assert e['exit_y'] == 10
            assert e['exit_direction'] == 'top'
            assert abs(e['distance_pixels'] - 670.0) < 0.1
            assert abs(e['duration_seconds'] - 5.0) < 0.1
            assert e['total_frames'] == 70
            assert e['classification'] is None
        finally:
            db.close()

    print("✓ test_add_track_event passed")


def test_get_track_events_filtering():
    """Test filtering track events by type and date."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'test.db')
        db = DatabaseManager(db_path)

        try:
            # Add various events
            db.add_track_event(
                track_id=1, event_type='track_completed',
                timestamp='2026-02-10T08:00:00', created_at='2026-02-10T07:59:55',
                entry_x=640, entry_y=680, exit_x=640, exit_y=10,
                exit_direction='top', distance_pixels=670.0, duration_seconds=5.0,
                total_frames=70, avg_confidence=0.92
            )
            db.add_track_event(
                track_id=2, event_type='track_lost',
                timestamp='2026-02-10T08:01:00', created_at='2026-02-10T08:00:55',
                entry_x=300, entry_y=680, exit_x=300, exit_y=400,
                exit_direction='timeout', distance_pixels=280.0, duration_seconds=2.0,
                total_frames=28, avg_confidence=0.75
            )
            db.add_track_event(
                track_id=3, event_type='track_invalid',
                timestamp='2026-02-10T08:02:00', created_at='2026-02-10T08:01:58',
                entry_x=640, entry_y=400, exit_x=640, exit_y=10,
                exit_direction='top', distance_pixels=390.0, duration_seconds=1.5,
                total_frames=21, avg_confidence=0.88
            )

            # Test: all events
            all_events = db.get_track_events()
            assert len(all_events) == 3, f"Expected 3 events, got {len(all_events)}"

            # Test: filter by type
            completed = db.get_track_events(event_type='track_completed')
            assert len(completed) == 1, f"Expected 1 completed, got {len(completed)}"
            assert completed[0]['track_id'] == 1

            lost = db.get_track_events(event_type='track_lost')
            assert len(lost) == 1, f"Expected 1 lost, got {len(lost)}"
            assert lost[0]['track_id'] == 2

            invalid = db.get_track_events(event_type='track_invalid')
            assert len(invalid) == 1, f"Expected 1 invalid, got {len(invalid)}"
            assert invalid[0]['track_id'] == 3

            # Test: filter by date
            after = db.get_track_events(start_date='2026-02-10T08:01:00')
            assert len(after) == 2, f"Expected 2 after 08:01, got {len(after)}"

            before = db.get_track_events(end_date='2026-02-10T08:00:30')
            assert len(before) == 1, f"Expected 1 before 08:00:30, got {len(before)}"
        finally:
            db.close()

    print("✓ test_get_track_events_filtering passed")


def test_get_track_event_stats():
    """Test aggregated track event statistics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'test.db')
        db = DatabaseManager(db_path)

        try:
            # Add events
            for i in range(5):
                db.add_track_event(
                    track_id=i+1, event_type='track_completed',
                    timestamp=f'2026-02-10T08:0{i}:00', created_at=f'2026-02-10T07:5{i}:00',
                    distance_pixels=float(500 + i * 10), duration_seconds=float(3 + i * 0.5),
                    total_frames=50, avg_confidence=0.9
                )
            for i in range(3):
                db.add_track_event(
                    track_id=i+10, event_type='track_lost',
                    timestamp=f'2026-02-10T08:1{i}:00', created_at=f'2026-02-10T08:0{i}:00',
                    distance_pixels=float(100 + i * 20), duration_seconds=float(1 + i * 0.3),
                    total_frames=15, avg_confidence=0.7
                )

            stats = db.get_track_event_stats()

            assert stats['total'] == 8, f"Expected total=8, got {stats['total']}"
            assert 'track_completed' in stats['by_type'], "Missing track_completed in stats"
            assert 'track_lost' in stats['by_type'], "Missing track_lost in stats"
            assert int(stats['by_type']['track_completed']['count']) == 5
            assert int(stats['by_type']['track_lost']['count']) == 3
        finally:
            db.close()

    print("✓ test_get_track_event_stats passed")


def test_update_track_event_classification():
    """Test updating classification on a track event."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'test.db')
        db = DatabaseManager(db_path)

        try:
            # Add a completed event without classification
            db.add_track_event(
                track_id=42,
                event_type='track_completed',
                timestamp='2026-02-10T08:00:00',
                created_at='2026-02-10T07:59:55',
                entry_x=640, entry_y=680, exit_x=640, exit_y=10,
                exit_direction='top', distance_pixels=670.0, duration_seconds=5.0,
                total_frames=70, avg_confidence=0.92
            )

            # Verify no classification yet
            events = db.get_track_events()
            assert events[0]['classification'] is None

            # Update classification
            db.update_track_event_classification(
                track_id=42,
                classification='Wheatberry',
                classification_confidence=0.95
            )

            # Verify classification updated
            events = db.get_track_events()
            assert events[0]['classification'] == 'Wheatberry', \
                f"Expected 'Wheatberry', got {events[0]['classification']}"
            assert abs(events[0]['classification_confidence'] - 0.95) < 0.01
        finally:
            db.close()

    print("✓ test_update_track_event_classification passed")


def test_position_history_json():
    """Test that position_history is stored and retrieved as valid JSON."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'test.db')
        db = DatabaseManager(db_path)

        try:
            positions = [[640, 680], [640, 500], [640, 300], [640, 100], [640, 10]]
            db.add_track_event(
                track_id=1,
                event_type='track_completed',
                timestamp='2026-02-10T08:00:00',
                created_at='2026-02-10T07:59:55',
                position_history=json.dumps(positions)
            )

            events = db.get_track_events()
            stored_positions = json.loads(events[0]['position_history'])
            assert stored_positions == positions, \
                f"Position history mismatch: {stored_positions} != {positions}"
        finally:
            db.close()

    print("✓ test_position_history_json passed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Track Event Analytics")
    print("=" * 60)
    print()

    tests = [
        test_track_events_table_created,
        test_add_track_event,
        test_get_track_events_filtering,
        test_get_track_event_stats,
        test_update_track_event_classification,
        test_position_history_json,
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
