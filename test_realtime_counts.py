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
    """Test /counts HTML page renders with SSE and simplified UX."""
    from src.endpoint.shared import init_shared_resources, cleanup_shared_resources

    init_shared_resources()
    try:
        from fastapi.testclient import TestClient
        from src.endpoint.server import app

        client = TestClient(app)

        resp = client.get("/counts")
        assert resp.status_code == 200
        body = resp.text
        assert "Live Counts" in body
        assert "EventSource" in body
        assert "/api/counts/stream" in body
        assert "Confirmed" in body
        # Simplified UX elements
        assert "/api/bag-types" in body, "Should fetch bag types for thumbnails"
        assert "buildClassCards" in body, "Should build all-type cards on init"
        assert "updateClassCards" in body, "Should update confirmed counts"
        assert "processing-card" in body, "Should have processing bridge card"
        assert "connect-line" in body, "Should have connecting line"
        assert "hero-card" in body, "Should have hero section"
        assert "stat-card" in body, "Should use analytics-style stat cards"
        assert "confirm-flash" in body, "Should have confirmation animation"
        assert "confirm-pulse" in body, "Should have line pulse animation"
        # Removed elements should NOT be present
        assert "Live Event Feed" not in body, "Should not have live events feed"
        assert "feed-item" not in body, "Should not have feed item styling"
        assert "Smoothing Window" not in body, "Should not show smoothing window"
        assert "window-fill" not in body, "Should not have smoothing progress bar"
        assert "renderBatch" not in body, "Should not have batch item rendering"
        assert "count-breakdown" not in body, "Should not show per-class pending breakdown"
        print("PASS: /counts page renders with simplified UX and processing bridge")
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

    # Empty state should have recent_events
    empty = _empty_state()
    assert "recent_events" in empty
    assert empty["recent_events"] == []
    print("PASS: empty state includes recent_events")

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


if __name__ == "__main__":
    test_pipeline_state_module()
    test_smoother_pending_summary()
    test_api_counts_endpoint()
    test_counts_html_page()
    test_api_bag_types_endpoint()
    test_pipeline_state_with_events()
    test_sse_stream_endpoint()
    print("\n=== All tests PASSED ===")
