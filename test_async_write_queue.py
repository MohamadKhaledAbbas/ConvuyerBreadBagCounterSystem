#!/usr/bin/env python3
"""
Tests for async write queue performance improvements.

Tests:
1. enqueue_write() is non-blocking and writes appear in DB after flush
2. enqueue_track_event_detail() is non-blocking and data is correct
3. Write queue drains on close()
4. Queue full drops writes gracefully
5. Concurrent writes from multiple threads don't block each other
6. WAL mode is enabled
"""

import os
import sqlite3
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.logging.Database import DatabaseManager


def test_enqueue_write_nonblocking():
    """Test that enqueue_write returns immediately and data appears after flush."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'test.db')
        db = DatabaseManager(db_path)
        try:
            # Enqueue a write - should return immediately
            start = time.monotonic()
            db.enqueue_write(
                "INSERT INTO config (key, value) VALUES (?, ?)",
                ('test_key', 'test_value')
            )
            elapsed = time.monotonic() - start
            assert elapsed < 0.01, f"enqueue_write should be instant, took {elapsed:.3f}s"

            # Wait for flush
            db.flush_write_queue(timeout=3.0)

            # Verify data appeared
            val = db.get_config('test_key')
            assert val == 'test_value', f"Expected 'test_value', got '{val}'"
        finally:
            db.close()
    print("✓ test_enqueue_write_nonblocking passed")


def test_enqueue_track_event_detail_nonblocking():
    """Test that enqueue_track_event_detail is non-blocking and data is correct."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'test.db')
        db = DatabaseManager(db_path)
        try:
            # Enqueue multiple detail writes
            start = time.monotonic()
            for i in range(10):
                db.enqueue_track_event_detail(
                    track_id=42,
                    timestamp=f'2026-02-10T09:0{i}:00',
                    step_type='roi_collected',
                    bbox_x1=100 + i, bbox_y1=200, bbox_x2=200, bbox_y2=300,
                    quality_score=150.0 + i,
                    roi_index=i
                )
            elapsed = time.monotonic() - start
            assert elapsed < 0.05, f"10 enqueues should be < 50ms, took {elapsed*1000:.1f}ms"

            # Wait for flush
            db.flush_write_queue(timeout=3.0)

            # Verify data
            details = db.get_track_event_details(track_id=42)
            assert len(details) == 10, f"Expected 10 details, got {len(details)}"
            assert details[0]['bbox_x1'] == 100
            assert details[9]['bbox_x1'] == 109
        finally:
            db.close()
    print("✓ test_enqueue_track_event_detail_nonblocking passed")


def test_write_queue_drains_on_close():
    """Test that close() flushes all pending writes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'test.db')
        db = DatabaseManager(db_path)

        # Enqueue writes without waiting
        for i in range(20):
            db.enqueue_track_event_detail(
                track_id=99,
                timestamp=f'2026-02-10T10:00:{i:02d}',
                step_type='roi_classified',
                roi_index=i,
                class_name='Wheatberry',
                confidence=0.9 + i * 0.001
            )

        # Close should drain the queue
        db.close()

        # Re-open and verify
        db2 = DatabaseManager(db_path)
        try:
            details = db2.get_track_event_details(track_id=99)
            assert len(details) == 20, f"Expected 20 details after close, got {len(details)}"
        finally:
            db2.close()
    print("✓ test_write_queue_drains_on_close passed")


def test_queue_full_drops_gracefully():
    """Test that when queue is full, writes are dropped without crashing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'test.db')
        db = DatabaseManager(db_path)
        try:
            # The queue maxsize is 10000 - try to overfill it
            # First stop the write thread so nothing drains
            db._write_stop.set()
            db._write_thread.join(timeout=2.0)

            # Fill beyond capacity - should not raise
            dropped = 0
            for i in range(10500):
                try:
                    db._write_queue.put_nowait(
                        ("INSERT INTO config (key, value) VALUES (?, ?)", (f'k{i}', f'v{i}'))
                    )
                except Exception:
                    dropped += 1

            # enqueue_write on a full queue should not crash
            db.enqueue_write("INSERT INTO config (key, value) VALUES (?, ?)", ('safe', 'ok'))
            assert dropped > 0, "Queue should have overflowed"
        finally:
            # Drain queue before closing to avoid blocking
            while not db._write_queue.empty():
                try:
                    db._write_queue.get_nowait()
                except Exception:
                    break
            db.close()
    print("✓ test_queue_full_drops_gracefully passed")


def test_concurrent_writes_dont_block():
    """Test that concurrent enqueue from multiple threads doesn't block."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'test.db')
        db = DatabaseManager(db_path)
        try:
            results = {'thread_times': []}
            lock = threading.Lock()

            def writer_thread(thread_id, count):
                start = time.monotonic()
                for i in range(count):
                    db.enqueue_track_event_detail(
                        track_id=thread_id * 1000 + i,
                        timestamp=f'2026-02-10T11:{thread_id:02d}:{i:02d}',
                        step_type='roi_collected',
                        bbox_x1=i, bbox_y1=i, bbox_x2=i+50, bbox_y2=i+50,
                        roi_index=i
                    )
                elapsed = time.monotonic() - start
                with lock:
                    results['thread_times'].append(elapsed)

            # 4 threads each writing 50 items
            threads = []
            for t in range(4):
                th = threading.Thread(target=writer_thread, args=(t, 50))
                threads.append(th)
                th.start()

            for th in threads:
                th.join(timeout=5.0)

            # Each thread should complete quickly (< 100ms for 50 enqueues)
            for i, elapsed in enumerate(results['thread_times']):
                assert elapsed < 1.0, f"Thread {i} took {elapsed:.3f}s for 50 enqueues"

            # Wait for flush
            db.flush_write_queue(timeout=5.0)

            # Verify total: 4 * 50 = 200
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM track_event_details")
            count = cursor.fetchone()[0]
            conn.close()
            assert count == 200, f"Expected 200 details, got {count}"
        finally:
            db.close()
    print("✓ test_concurrent_writes_dont_block passed")


def test_wal_mode_enabled():
    """Test that WAL journal mode is active."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'test.db')
        db = DatabaseManager(db_path)
        try:
            conn = db._get_connection()
            cursor = conn.execute("PRAGMA journal_mode")
            mode = cursor.fetchone()[0]
            assert mode == 'wal', f"Expected WAL mode, got '{mode}'"
        finally:
            db.close()
    print("✓ test_wal_mode_enabled passed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Async Write Queue Performance")
    print("=" * 60)
    print()

    tests = [
        test_enqueue_write_nonblocking,
        test_enqueue_track_event_detail_nonblocking,
        test_write_queue_drains_on_close,
        test_queue_full_drops_gracefully,
        test_concurrent_writes_dont_block,
        test_wal_mode_enabled,
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
