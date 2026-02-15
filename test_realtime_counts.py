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
    """Test /counts HTML page renders with SSE code."""
    from src.endpoint.shared import init_shared_resources, cleanup_shared_resources

    init_shared_resources()
    try:
        from fastapi.testclient import TestClient
        from src.endpoint.server import app

        client = TestClient(app)

        resp = client.get("/counts")
        assert resp.status_code == 200
        body = resp.text
        assert "Live Pipeline Counts" in body
        assert "EventSource" in body
        assert "Smoothing Window" in body
        assert "/api/counts/stream" in body
        assert "Confirmed" in body
        assert "Pending" in body
        assert "Just Classified" in body
        print("PASS: /counts page renders with SSE and three-tier display")
    finally:
        cleanup_shared_resources()


def test_sse_stream_endpoint():
    """Test /api/counts/stream SSE endpoint exists and returns correct media type."""
    # The SSE endpoint returns an infinite stream, so we can't fully test it
    # in a sync test context. Verify it's registered as a route.
    from src.endpoint.server import app

    route_paths = [route.path for route in app.routes]
    assert "/api/counts/stream" in route_paths
    print("PASS: /api/counts/stream route is registered")


if __name__ == "__main__":
    test_pipeline_state_module()
    test_smoother_pending_summary()
    test_api_counts_endpoint()
    test_counts_html_page()
    test_sse_stream_endpoint()
    print("\n=== All tests PASSED ===")
