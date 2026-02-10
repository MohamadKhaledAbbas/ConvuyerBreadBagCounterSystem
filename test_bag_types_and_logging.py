#!/usr/bin/env python3
"""
Tests for bag_types seeding and log retention cleanup.

Tests:
1. Default bag_types are seeded on schema initialization
2. All 9 expected bag types exist with correct names
3. Re-initialization doesn't duplicate bag_types (INSERT OR IGNORE)
4. Log retention cleanup deletes old files
5. Log retention keeps recent files
6. StructuredLogger is no longer exported
"""

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.logging.Database import DatabaseManager
from src.utils.AppLogging import _cleanup_old_logs


EXPECTED_BAG_TYPES = [
    'Black_Orange',
    'Blue_Yellow',
    'Bran',
    'Brown_Orange_Family',
    'Green_Yellow',
    'Purple_Yellow',
    'Red_Yellow',
    'Rejected',
    'Wheatberry',
]


def test_bag_types_seeded_on_init():
    """Test that all 9 default bag types are created on DB initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'test.db')
        db = DatabaseManager(db_path)
        try:
            bag_types = db.get_all_bag_types()
            names = sorted([bt['name'] for bt in bag_types])
            assert names == EXPECTED_BAG_TYPES, f"Expected {EXPECTED_BAG_TYPES}, got {names}"
            assert len(bag_types) == 9, f"Expected 9 bag types, got {len(bag_types)}"
        finally:
            db.close()
    print("✓ test_bag_types_seeded_on_init passed")


def test_bag_types_have_correct_thumbnails():
    """Test that bag types have correct thumbnail paths."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'test.db')
        db = DatabaseManager(db_path)
        try:
            for name in EXPECTED_BAG_TYPES:
                bt = db.get_bag_type_by_name(name)
                assert bt is not None, f"Bag type '{name}' not found"
                expected_thumb = f"data/classes/{name}/{name}.jpg"
                assert bt['thumb'] == expected_thumb, f"Expected thumb '{expected_thumb}', got '{bt['thumb']}'"
        finally:
            db.close()
    print("✓ test_bag_types_have_correct_thumbnails passed")


def test_bag_types_no_duplicates_on_reinit():
    """Test that re-initializing schema doesn't create duplicates."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'test.db')
        db = DatabaseManager(db_path)
        try:
            # Force re-initialization
            db._initialize_schema()
            bag_types = db.get_all_bag_types()
            assert len(bag_types) == 9, f"Expected 9 bag types after reinit, got {len(bag_types)}"
        finally:
            db.close()
    print("✓ test_bag_types_no_duplicates_on_reinit passed")


def test_black_orange_and_purple_yellow_exist():
    """Test that the two new bag types (Black_Orange, Purple_Yellow) exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'test.db')
        db = DatabaseManager(db_path)
        try:
            black_orange = db.get_bag_type_by_name('Black_Orange')
            assert black_orange is not None, "Black_Orange bag type not found"
            assert black_orange['id'] == 8, f"Expected id=8, got {black_orange['id']}"

            purple_yellow = db.get_bag_type_by_name('Purple_Yellow')
            assert purple_yellow is not None, "Purple_Yellow bag type not found"
            assert purple_yellow['id'] == 9, f"Expected id=9, got {purple_yellow['id']}"
        finally:
            db.close()
    print("✓ test_black_orange_and_purple_yellow_exist passed")


def test_log_retention_deletes_old_files():
    """Test that log retention cleanup deletes old log files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create old log file (10 days old)
        old_log = os.path.join(tmpdir, 'conveyer_counter_20260130_120000.log')
        with open(old_log, 'w') as f:
            f.write("old log")
        # Set modification time to 10 days ago
        old_mtime = time.time() - (10 * 86400)
        os.utime(old_log, (old_mtime, old_mtime))

        # Create recent log file
        recent_log = os.path.join(tmpdir, 'conveyer_counter_20260209_120000.log')
        with open(recent_log, 'w') as f:
            f.write("recent log")

        # Run cleanup with 7-day retention
        _cleanup_old_logs(tmpdir, retention_days=7)

        assert not os.path.exists(old_log), "Old log file should have been deleted"
        assert os.path.exists(recent_log), "Recent log file should be kept"
    print("✓ test_log_retention_deletes_old_files passed")


def test_structured_logger_removed():
    """Test that structured_logger is no longer exported from AppLogging."""
    import src.utils.AppLogging as module
    assert not hasattr(module, 'structured_logger'), "structured_logger should be removed"
    assert not hasattr(module, 'StructuredLogger'), "StructuredLogger class should be removed"
    print("✓ test_structured_logger_removed passed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Bag Types Seeding & Log Retention")
    print("=" * 60)
    print()

    tests = [
        test_bag_types_seeded_on_init,
        test_bag_types_have_correct_thumbnails,
        test_bag_types_no_duplicates_on_reinit,
        test_black_orange_and_purple_yellow_exist,
        test_log_retention_deletes_old_files,
        test_structured_logger_removed,
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
