"""
Tests for monitoring logs DB operations and db_log_handler.

Validates:
1. monitoring_logs table creation via schema
2. insert_monitoring_log() queues writes correctly
3. get_monitoring_logs() returns results with filters
4. purge_old_monitoring_logs() removes old entries
5. get_monitoring_log_summary() aggregates correctly
6. DatabaseLogHandler captures WARNING+ and ignores INFO
7. Level filter works (WARNING only, ERROR only)
8. since_minutes filter works
9. Limit capping works
"""

import sys
import os
import time
import logging

sys.path.insert(0, os.path.dirname(__file__))

from src.logging.Database import DatabaseManager
from src.logging.db_log_handler import DatabaseLogHandler, attach_db_log_handler


# ── helpers ──────────────────────────────────────────────────────────────────

def make_db():
    """Create an in-memory-like temp DB for testing."""
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = DatabaseManager(path)
    return db, path


def cleanup_db(db, path):
    db.close()
    try:
        os.unlink(path)
    except OSError:
        pass


# ── 1. Table creation ────────────────────────────────────────────────────────

def test_monitoring_logs_table_exists():
    db, path = make_db()
    try:
        conn = db._get_connection()
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='monitoring_logs'"
        )
        assert cursor.fetchone() is not None, "monitoring_logs table should exist"
        print("PASS: monitoring_logs table created by schema")
    finally:
        cleanup_db(db, path)


# ── 2. insert_monitoring_log ─────────────────────────────────────────────────

def test_insert_and_retrieve():
    db, path = make_db()
    try:
        db.insert_monitoring_log("WARNING", "test", "Something warned")
        db.insert_monitoring_log("ERROR", "test", "Something broke", '{"key":"value"}')
        db.flush_write_queue(timeout=3.0)

        logs = db.get_monitoring_logs()
        assert len(logs) == 2, f"Expected 2 logs, got {len(logs)}"
        # Newest first
        assert logs[0]["level"] == "ERROR"
        assert logs[1]["level"] == "WARNING"
        assert logs[0]["details"] == '{"key":"value"}'
        assert logs[1]["message"] == "Something warned"
        print("PASS: insert_monitoring_log and get_monitoring_logs work")
    finally:
        cleanup_db(db, path)


# ── 3. Level filter ─────────────────────────────────────────────────────────

def test_level_filter():
    db, path = make_db()
    try:
        db.insert_monitoring_log("WARNING", "test", "warn1")
        db.insert_monitoring_log("ERROR", "test", "err1")
        db.insert_monitoring_log("WARNING", "test", "warn2")
        db.flush_write_queue(timeout=3.0)

        warnings = db.get_monitoring_logs(level="WARNING")
        assert len(warnings) == 2, f"Expected 2 warnings, got {len(warnings)}"
        assert all(l["level"] == "WARNING" for l in warnings)

        errors = db.get_monitoring_logs(level="ERROR")
        assert len(errors) == 1, f"Expected 1 error, got {len(errors)}"
        print("PASS: Level filter works correctly")
    finally:
        cleanup_db(db, path)


# ── 4. Limit capping ────────────────────────────────────────────────────────

def test_limit_capping():
    db, path = make_db()
    try:
        for i in range(10):
            db.insert_monitoring_log("WARNING", "test", f"msg {i}")
        db.flush_write_queue(timeout=3.0)

        logs = db.get_monitoring_logs(limit=3)
        assert len(logs) == 3, f"Expected 3, got {len(logs)}"

        # Max cap at 500
        logs_all = db.get_monitoring_logs(limit=999)
        assert len(logs_all) == 10  # only 10 exist
        print("PASS: Limit capping works")
    finally:
        cleanup_db(db, path)


# ── 5. purge_old_monitoring_logs ─────────────────────────────────────────────

def test_purge():
    db, path = make_db()
    try:
        # Insert with a very old timestamp
        conn = db._get_connection()
        conn.execute(
            "INSERT INTO monitoring_logs (timestamp, level, source, message) VALUES (?, ?, ?, ?)",
            ("2020-01-01 00:00:00", "ERROR", "test", "ancient error")
        )
        conn.commit()

        db.insert_monitoring_log("WARNING", "test", "recent warning")
        db.flush_write_queue(timeout=3.0)

        deleted = db.purge_old_monitoring_logs(retention_days=7)
        assert deleted == 1, f"Expected 1 deleted, got {deleted}"

        remaining = db.get_monitoring_logs()
        assert len(remaining) == 1
        assert remaining[0]["message"] == "recent warning"
        print("PASS: purge_old_monitoring_logs removes old entries")
    finally:
        cleanup_db(db, path)


# ── 6. get_monitoring_log_summary ────────────────────────────────────────────

def test_summary():
    db, path = make_db()
    try:
        db.insert_monitoring_log("WARNING", "test", "w1")
        db.insert_monitoring_log("WARNING", "test", "w2")
        db.insert_monitoring_log("ERROR", "test", "e1")
        db.insert_monitoring_log("CRITICAL", "test", "c1")
        db.flush_write_queue(timeout=3.0)

        summary = db.get_monitoring_log_summary()
        assert summary["total"] == 4
        assert summary["warning_count"] == 2
        assert summary["error_count"] == 1
        assert summary["critical_count"] == 1
        assert summary["latest_timestamp"] is not None
        print("PASS: get_monitoring_log_summary aggregates correctly")
    finally:
        cleanup_db(db, path)


# ── 7. DatabaseLogHandler captures WARNING+ ──────────────────────────────────

def test_db_log_handler():
    db, path = make_db()
    try:
        test_logger = logging.getLogger("test_handler_logger")
        test_logger.setLevel(logging.DEBUG)
        handler = DatabaseLogHandler(db, level=logging.WARNING)
        handler.setFormatter(logging.Formatter("%(message)s"))
        test_logger.addHandler(handler)

        test_logger.info("This should not be stored")
        test_logger.warning("This is a warning")
        test_logger.error("This is an error")
        db.flush_write_queue(timeout=3.0)

        logs = db.get_monitoring_logs()
        assert len(logs) == 2, f"Expected 2 logs (WARNING+), got {len(logs)}"
        messages = [l["message"] for l in logs]
        assert "This is a warning" in messages
        assert "This is an error" in messages
        assert "This should not be stored" not in messages
        print("PASS: DatabaseLogHandler captures WARNING+ and ignores INFO")

        # Cleanup handler
        test_logger.removeHandler(handler)
    finally:
        cleanup_db(db, path)


# ── 8. since_minutes filter ─────────────────────────────────────────────────

def test_since_minutes():
    db, path = make_db()
    try:
        # Insert an old log
        conn = db._get_connection()
        conn.execute(
            "INSERT INTO monitoring_logs (timestamp, level, source, message) VALUES (?, ?, ?, ?)",
            ("2020-01-01 00:00:00", "ERROR", "test", "old error")
        )
        conn.commit()

        db.insert_monitoring_log("WARNING", "test", "recent")
        db.flush_write_queue(timeout=3.0)

        logs = db.get_monitoring_logs(since_minutes=60)
        assert len(logs) == 1, f"Expected 1 recent log, got {len(logs)}"
        assert logs[0]["message"] == "recent"
        print("PASS: since_minutes filter works")
    finally:
        cleanup_db(db, path)


# ── Run all tests ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_monitoring_logs_table_exists,
        test_insert_and_retrieve,
        test_level_filter,
        test_limit_capping,
        test_purge,
        test_summary,
        test_db_log_handler,
        test_since_minutes,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAIL: {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print(f"{'=' * 60}")

    if failed > 0:
        sys.exit(1)
