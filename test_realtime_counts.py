"""
Test for real-time count visibility features.

Tests:
- Pipeline state read/write (cross-process shared state)
- BidirectionalSmoother.get_pending_summary()
- /api/counts JSON endpoint
- /counts HTML page rendering
- SSE stream endpoint availability
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_pipeline_state_module():
    """Test pipeline_state read/write."""
    from src.endpoint.pipeline_state import write_state, read_state

    fd, tf = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    os.remove(tf)  # Remove so read_state can test missing file
    try:
        # Empty state when missing
        state = read_state(tf)
        assert state["confirmed_total"] == 0
        assert state["window_status"]["size"] == 7
        print("PASS: read_state returns empty state when file missing")

        # Round-trip
        test_state = {
            "confirmed": {"Brown_Orange": 5, "Black_Orange": 2},
            "pending": {"Brown_Orange": 3},
            "just_classified": {"Brown_Orange": 1},
            "confirmed_total": 7,
            "pending_total": 3,
            "just_classified_total": 1,
            "smoothing_rate": 0.286,
            "window_status": {"size": 7, "current_items": 3, "next_confirmation_in": 4},
        }
        assert write_state(test_state, tf) is True
        result = read_state(tf)
        assert result["confirmed"] == test_state["confirmed"]
        assert result["pending_total"] == 3
        assert result["window_status"]["current_items"] == 3
        assert "_updated_at" in result
        print("PASS: write_state/read_state round-trip works")

        # Atomic write (no leftover .tmp file)
        assert not os.path.exists(tf + ".tmp")
        print("PASS: atomic write cleans up tmp file")
    finally:
        if os.path.exists(tf):
            os.remove(tf)


def test_smoother_pending_summary():
    """Test BidirectionalSmoother.get_pending_summary()."""
    from src.tracking.BidirectionalSmoother import BidirectionalSmoother

    smoother = BidirectionalSmoother(window_size=7)

    # Add classifications
    smoother.add_classification(1, "Brown_Orange", 0.95, 0.8, 5)
    smoother.add_classification(2, "Brown_Orange", 0.90, 0.8, 4)
    smoother.add_classification(3, "Black_Orange", 0.85, 0.7, 3)

    summary = smoother.get_pending_summary()
    assert summary == {"Brown_Orange": 2, "Black_Orange": 1}
    print("PASS: get_pending_summary returns correct class counts")

    # After filling window
    for i in range(4, 8):
        smoother.add_classification(i, "Brown_Orange", 0.92, 0.8, 5)

    summary = smoother.get_pending_summary()
    assert sum(summary.values()) == 6  # 7 added, 1 confirmed = 6 pending
    print("PASS: get_pending_summary correct after confirmation")


def test_api_counts_endpoint():
    """Test /api/counts JSON endpoint."""
    fd, state_file = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    old_env = os.environ.get("PIPELINE_STATE_FILE")
    os.environ["PIPELINE_STATE_FILE"] = state_file

    from src.endpoint.pipeline_state import write_state

    write_state(
        {
            "confirmed": {"Brown_Orange": 12, "Black_Orange": 2},
            "pending": {"Brown_Orange": 4},
            "just_classified": {"Brown_Orange": 2},
            "confirmed_total": 14,
            "pending_total": 4,
            "just_classified_total": 2,
            "smoothing_rate": 0.286,
            "window_status": {"size": 7, "current_items": 4, "next_confirmation_in": 3},
        },
    )

    try:
        from fastapi.testclient import TestClient
        from src.endpoint.server import app

        client = TestClient(app)

        resp = client.get("/api/counts")
        assert resp.status_code == 200
        data = resp.json()
        assert data["confirmed"]["Brown_Orange"] == 12
        assert data["pending"]["Brown_Orange"] == 4
        assert data["confirmed_total"] == 14
        assert data["pending_total"] == 4
        assert data["just_classified_total"] == 2
        assert data["smoothing_rate"] == 0.286
        assert data["window_status"]["size"] == 7
        assert data["window_status"]["current_items"] == 4
        assert data["window_status"]["next_confirmation_in"] == 3
        # Internal fields should be stripped
        assert "_updated_at" not in data
        print("PASS: /api/counts returns correct three-tier data")
    finally:
        if old_env is None:
            os.environ.pop("PIPELINE_STATE_FILE", None)
        else:
            os.environ["PIPELINE_STATE_FILE"] = old_env
        if os.path.exists(state_file):
            os.remove(state_file)


def test_counts_html_page():
    """Test /counts HTML page renders with SSE, batch totals, processing bar, and per-class breakdown (Arabic)."""
    from src.endpoint.shared import init_shared_resources, cleanup_shared_resources

    init_shared_resources()
    try:
        from fastapi.testclient import TestClient
        from src.endpoint.server import app

        client = TestClient(app)

        resp = client.get("/counts")
        assert resp.status_code == 200
        body = resp.text
        # Arabic page structure
        assert 'lang="ar"' in body, "Should be Arabic"
        assert 'dir="rtl"' in body, "Should be RTL"
        assert "العد المباشر" in body, "Should have Arabic title"
        assert "الدفعة الحالية" in body, "Should have Arabic batch label"
        assert "مؤكد" in body, "Should have Arabic confirmed label"
        assert "قيد المعالجة" in body, "Should have Arabic processing label"
        assert "العدد حسب النوع" in body, "Should have Arabic section label"
        assert "قيد التشغيل الآن" in body, "Should have Arabic 'Now Processing' label"
        # Core SSE and JS logic preserved
        assert "EventSource" in body
        assert "/api/counts/stream" in body
        assert "/api/bag-types" in body, "Should fetch bag types for thumbnails"
        assert "buildClassCards" in body, "Should build all-type cards on init"
        assert "updateClassCards" in body, "Should update cards with batch counts"
        assert "processing-bar" in body, "Should have compact processing bar"
        assert "connect-arrow" in body, "Should have connecting arrow"
        assert "hero-card" in body, "Should have hero section"
        assert "stat-card" in body, "Should use analytics-style stat cards"
        assert "confirm-flash" in body, "Should have confirmation animation"
        assert "confirm-pulse" in body, "Should have arrow pulse animation"
        assert "heroBatch" in body, "Should have batch total hero element"
        assert "batchTotal" in body, "JS should compute batch total"
        assert "card-batch-info" in body, "Should have per-class batch breakdown"
        assert "confirmed-part" in body, "Should show confirmed portion"
        assert "pending-part" in body, "Should show pending portion"
        assert "allPending" in body, "JS should merge pending + just_classified"
        assert "current_batch_type" in body, "Should reference current_batch_type from data"
        assert "procName" in body, "Should have batch type name element"
        assert "procImg" in body, "Should have batch type image element"
        assert "procCountValue" in body, "Should have per-type count element"
        assert "procBreakdown" in body, "Should have per-type breakdown"
        assert "prevBatchType" in body, "JS should track batch type changes"
        assert "procIdle" in body, "Should have idle state"
        assert "procActive" in body, "Should have active state"
        assert "updateProcessingBar" in body, "Should have processing bar updater"
        # Removed elements should NOT be present
        assert "Live Event Feed" not in body, "Should not have live events feed"
        assert "feed-item" not in body, "Should not have feed item styling"
        assert "Smoothing Window" not in body, "Should not show smoothing window"
        assert "window-fill" not in body, "Should not have smoothing progress bar"
        print("PASS: /counts page renders with Arabic RTL layout, processing bar, batch totals")
    finally:
        cleanup_shared_resources()


def test_api_bag_types_endpoint():
    """Test /api/bag-types returns bag type metadata."""
    from src.endpoint.shared import init_shared_resources, cleanup_shared_resources

    # Expected default bag types seeded by schema.sql
    DEFAULT_BAG_TYPES = [
        "Brown_Orange", "Red_Yellow", "Wheatberry", "Blue_Yellow",
        "Green_Yellow", "Bran", "Black_Orange", "Purple_Yellow", "Rejected"
    ]

    init_shared_resources()
    try:
        from fastapi.testclient import TestClient
        from src.endpoint.server import app

        client = TestClient(app)

        resp = client.get("/api/bag-types")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list), "Should return a list"
        assert len(data) >= len(DEFAULT_BAG_TYPES), \
            f"Should have at least {len(DEFAULT_BAG_TYPES)} default bag types, got {len(data)}"

        # Check expected bag types are present
        names = {bt["name"] for bt in data}
        for expected in DEFAULT_BAG_TYPES:
            assert expected in names, f"Should include {expected}"

        # Check thumb paths are normalized to web paths
        for bt in data:
            if bt.get("thumb"):
                assert "known_classes/" in bt["thumb"], \
                    f"Thumb path should be normalized: {bt['thumb']}"
                assert "data/classes/" not in bt["thumb"], \
                    f"Should not have filesystem path: {bt['thumb']}"
        print("PASS: /api/bag-types returns normalized bag type metadata")
    finally:
        cleanup_shared_resources()


def test_pipeline_state_with_events():
    """Test pipeline state includes recent_events field."""
    from src.endpoint.pipeline_state import write_state, read_state, _empty_state

    # Empty state should have recent_events and current_batch_type
    empty = _empty_state()
    assert "recent_events" in empty
    assert empty["recent_events"] == []
    assert "current_batch_type" in empty
    assert empty["current_batch_type"] is None
    assert "last_classified_type" in empty
    assert empty["last_classified_type"] is None
    print("PASS: empty state includes recent_events, current_batch_type, and last_classified_type")

    # Round-trip with events
    fd, tf = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    os.remove(tf)
    try:
        import time
        state = {
            "confirmed": {"Brown_Orange": 5},
            "pending": {},
            "just_classified": {},
            "confirmed_total": 5,
            "pending_total": 0,
            "just_classified_total": 0,
            "smoothing_rate": 0.0,
            "window_status": {"size": 7, "current_items": 0, "next_confirmation_in": 7},
            "recent_events": [
                {"ts": time.time(), "msg": "CONFIRMED T1:Brown_Orange"},
                {"ts": time.time(), "msg": "TENTATIVE T2:Brown_Orange (pending smoothing)"},
            ]
        }
        write_state(state, tf)
        result = read_state(tf)
        assert len(result["recent_events"]) == 2
        assert result["recent_events"][0]["msg"] == "CONFIRMED T1:Brown_Orange"
        print("PASS: pipeline state round-trips recent_events")
    finally:
        if os.path.exists(tf):
            os.remove(tf)


def test_sse_stream_endpoint():
    """Test /api/counts/stream SSE endpoint exists and returns correct media type."""
    # The SSE endpoint returns an infinite stream, so we can't fully test it
    # in a sync test context. Verify it's registered as a route.
    from src.endpoint.server import app

    route_paths = [route.path for route in app.routes]
    assert "/api/counts/stream" in route_paths
    assert "/api/bag-types" in route_paths
    print("PASS: /api/counts/stream and /api/bag-types routes are registered")


def test_high_confidence_outlier_smoothing():
    """Test that a single high-confidence outlier is smoothed when batch dominance is strong."""
    from src.tracking.BidirectionalSmoother import BidirectionalSmoother

    smoother = BidirectionalSmoother(window_size=7, min_window_dominance=0.7)

    # Fill window with 6 Black_Orange, then 1 Blue_Yellow with high confidence
    # Simulates: black, black, black, blue(1.00), black, black, black
    results = []
    for i in range(1, 4):
        r = smoother.add_classification(i, "Black_Orange", 0.95, 0.9, 5)
        if r:
            results.append(r)

    # Add a high-confidence outlier
    r = smoother.add_classification(4, "Blue_Yellow", 1.00, 1.0, 5)
    if r:
        results.append(r)

    for i in range(5, 8):
        r = smoother.add_classification(i, "Black_Orange", 0.95, 0.9, 5)
        if r:
            results.append(r)

    # The first confirmed item (index 0) should be Black_Orange
    assert len(results) >= 1, "Should have at least one confirmed result"
    assert results[0].class_name == "Black_Orange", \
        f"First confirmed should be Black_Orange, got {results[0].class_name}"

    # Flush remaining to check if the blue outlier (track 4) was smoothed
    flushed = smoother.flush_remaining()
    all_records = results + flushed

    # Find the record for track_id=4 (the blue outlier)
    track4 = [r for r in all_records if r.track_id == 4]
    assert len(track4) == 1, "Should have exactly one record for track 4"
    assert track4[0].class_name == "Black_Orange", \
        f"High-confidence outlier should be smoothed to Black_Orange, got {track4[0].class_name}"
    assert track4[0].smoothed is True, "Outlier should be marked as smoothed"
    assert track4[0].original_class == "Blue_Yellow", "Original class should be preserved"
    print("PASS: high-confidence single outlier is smoothed when batch dominance is strong")


def test_smoother_get_dominant_class():
    """Test BidirectionalSmoother.get_dominant_class() returns the majority class."""
    from src.tracking.BidirectionalSmoother import BidirectionalSmoother

    smoother = BidirectionalSmoother(window_size=7)

    # Empty window
    assert smoother.get_dominant_class() is None
    print("PASS: get_dominant_class returns None for empty window")

    # Add items
    smoother.add_classification(1, "Red_Yellow", 0.95, 0.9, 5)
    smoother.add_classification(2, "Red_Yellow", 0.90, 0.8, 4)
    smoother.add_classification(3, "Blue_Yellow", 0.85, 0.7, 3)

    dominant = smoother.get_dominant_class()
    assert dominant == "Red_Yellow", f"Dominant should be Red_Yellow, got {dominant}"
    print("PASS: get_dominant_class returns correct majority class")

    # Rejected items should be excluded
    smoother2 = BidirectionalSmoother(window_size=7)
    smoother2.add_classification(1, "Rejected", 0.10, 0.1, 0)
    smoother2.add_classification(2, "Rejected", 0.10, 0.1, 0)
    smoother2.add_classification(3, "Green_Yellow", 0.85, 0.7, 3)

    dominant2 = smoother2.get_dominant_class()
    assert dominant2 == "Green_Yellow", f"Dominant should be Green_Yellow (Rejected excluded), got {dominant2}"
    print("PASS: get_dominant_class excludes Rejected class")


def test_large_window_outlier_smoothing():
    """Test that multiple outliers are smoothed in larger windows (e.g., window=15).

    With window_size=15, up to 3 items (20% of 15) of a non-dominant class
    should be smoothed when the dominant class has ≥70% dominance.
    This prevents 'leaking' between batches during transitions.
    """
    from src.tracking.BidirectionalSmoother import BidirectionalSmoother

    smoother = BidirectionalSmoother(window_size=15, min_window_dominance=0.7)

    # Simulate: 13 Black_Orange with 2 Brown_Orange outliers scattered in the middle
    # Pattern: B,B,B,B,B,Br,B,B,B,Br,B,B,B,B,B  (B=Black, Br=Brown)
    # After filling window (15 items), 1 is confirmed; then add 2 more B
    # to push the Brown items through the full window context.
    classes = (
        ["Black_Orange"] * 5 + ["Brown_Orange"] +
        ["Black_Orange"] * 4 + ["Brown_Orange"] +
        ["Black_Orange"] * 4
    )

    results = []
    for i, cls in enumerate(classes, 1):
        conf = 0.99 if cls == "Brown_Orange" else 0.95
        r = smoother.add_classification(i, cls, conf, 0.9, 5)
        if r:
            results.append(r)

    # Add 2 more Black items to push Brown outliers through full window context
    for i in range(16, 18):
        r = smoother.add_classification(i, "Black_Orange", 0.95, 0.9, 5)
        if r:
            results.append(r)

    # Flush remaining
    flushed = smoother.flush_remaining()
    all_records = results + flushed

    assert len(all_records) == 17, f"Expected 17, got {len(all_records)}"

    # The 2 Brown outliers (tracks 6, 11) should be smoothed to Black
    for tid in [6, 11]:
        rec = [r for r in all_records if r.track_id == tid][0]
        assert rec.class_name == "Black_Orange", (
            f"T{tid} should be smoothed to Black_Orange, got {rec.class_name}"
        )
        assert rec.smoothed is True, f"T{tid} should be marked as smoothed"
        assert rec.original_class == "Brown_Orange", f"T{tid} original should be Brown_Orange"

    # All other items should be Black_Orange (unsmoothed)
    for rec in all_records:
        if rec.track_id not in [6, 11]:
            assert rec.class_name == "Black_Orange", (
                f"T{rec.track_id} should be Black_Orange, got {rec.class_name}"
            )

    print("PASS: large window (15) smooths multiple outliers (2 Brown in Black batch)")


def test_sql_subquery_for_update():
    """Test that the UPDATE query works in SQLite (uses subquery, not ORDER BY on UPDATE)."""
    import sqlite3
    import tempfile

    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("""CREATE TABLE track_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            track_id INTEGER,
            classification TEXT,
            classification_confidence REAL
        )""")
        # Insert two rows for same track_id
        conn.execute("INSERT INTO track_events (track_id, classification, classification_confidence) VALUES (1, NULL, NULL)")
        conn.execute("INSERT INTO track_events (track_id, classification, classification_confidence) VALUES (1, NULL, NULL)")
        conn.commit()

        # Execute the exact query pattern used in pipeline_core.py
        conn.execute(
            """UPDATE track_events
               SET classification = ?, classification_confidence = ?
               WHERE id = (
                   SELECT id FROM track_events
                   WHERE track_id = ? AND classification IS NULL
                   ORDER BY id DESC LIMIT 1
               )""",
            ("Brown_Orange", 0.95, 1)
        )
        conn.commit()

        # Verify: only the latest row (id=2) was updated
        rows = conn.execute("SELECT id, classification FROM track_events ORDER BY id").fetchall()
        assert rows[0] == (1, None), f"First row should remain NULL, got {rows[0]}"
        assert rows[1] == (2, "Brown_Orange"), f"Second row should be updated, got {rows[1]}"
        conn.close()
        print("PASS: UPDATE with subquery works in SQLite (no syntax error)")
    finally:
        os.remove(db_path)


if __name__ == "__main__":
    test_pipeline_state_module()
    test_smoother_pending_summary()
    test_api_counts_endpoint()
    test_counts_html_page()
    test_api_bag_types_endpoint()
    test_pipeline_state_with_events()
    test_sse_stream_endpoint()
    test_high_confidence_outlier_smoothing()
    test_smoother_get_dominant_class()
    test_large_window_outlier_smoothing()
    test_sql_subquery_for_update()
    print("\n=== All tests PASSED ===")
