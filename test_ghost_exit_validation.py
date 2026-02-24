"""
Tests for Ghost Companion Completion and Near-Top Diagnostics.

Ghost companion completion: when a track exits the top, any concurrent ghost
tracks (that were traveling alongside it near the top) are completed too —
the survivor's exit is hard evidence that companions made it.

Ghost near-top diagnostics: ghosts that expire without a survivor exiting
are flagged as ghost_exit_promoted=True on track_lost events (monitoring only).

Tests validate:
1. Ghost companions are completed when their concurrent track exits top
2. Companions must meet strict criteria (bottom entry, enough hits, near top)
3. Ghosts without a survivor exiting stay track_lost (diagnostic flag only)
4. No double counting possible
5. Ordering: lost tracks moved to ghost buffer BEFORE completions processed
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

from src.config.tracking_config import TrackingConfig
from src.tracking.ConveyorTracker import ConveyorTracker
from src.detection.BaseDetection import Detection

# Sleep time must exceed ghost_track_max_age_seconds (0.5s) to ensure expiry
GHOST_EXPIRY_WAIT = 0.7


# ── helpers ──��───────────────────────────────────────────────────────────────

def make_tracker(**overrides) -> ConveyorTracker:
    """Create a tracker with test-friendly defaults."""
    config = TrackingConfig()
    config.ghost_track_max_age_seconds = 0.5
    config.max_frames_without_detection = 3
    config.min_track_duration_frames = 2
    config.ghost_exit_validation_enabled = True
    config.ghost_exit_near_top_ratio = 0.35
    config.ghost_exit_min_travel_ratio = 0.40
    config.ghost_exit_min_hits = 3
    config.ghost_exit_predicted_top_ratio = 0.25
    config.exit_margin_pixels = 30
    config.target_fps = 17.0
    config.min_travel_duration_seconds = 0.0  # Tests run instantly, disable time check
    for key, val in overrides.items():
        setattr(config, key, val)
    return ConveyorTracker(config=config)


def make_detection(x1, y1, x2, y2, conf=0.9):
    return Detection(bbox=(x1, y1, x2, y2), confidence=conf)


def _box(x_center, y_center, w=180, h=250):
    """Return (x1, y1, x2, y2) centered on (x_center, y_center)."""
    return (x_center - w // 2, y_center - h // 2,
            x_center + w // 2, y_center + h // 2)


# ── Ghost Companion Completion Tests ─────────────────────────────────────────

def test_companion_completed_when_survivor_exits_top():
    """
    Core scenario: two tracks travel side-by-side, one loses detection
    near top (goes ghost), then the other exits top normally.
    The ghost companion should be completed as track_completed.

    Timing requirements:
    - Track B must be unmatched for > max_frames_without_detection (3) frames
      to be moved to ghost buffer via _check_completed_tracks.
    - Track A must reach y < exit_margin_pixels (30) to trigger top exit.
    - Ghost buffer move happens BEFORE completions, so B is in ghost buffer
      when A exits.
    """
    tracker = make_tracker()
    frame_shape = (720, 1280)

    # Track A: will exit top. Track B: stops detection at y~220 (near top).
    # Both travel together for 10 frames, then B disappears.
    # Use small steps (~30px) so velocity is moderate and B doesn't drift into exit zone.
    # A continues for 8 more frames (>3 needed to ghost B) until it exits top.
    positions_a = [600, 570, 540, 510, 480, 450, 400, 350, 300, 250, 200, 160, 120, 80, 50, 30, 20, 10]
    positions_b = [610, 580, 550, 520, 490, 460, 410, 360, 310, 240]  # 10 frames, last pos y=240

    # Move both together for first N frames
    for i in range(len(positions_b)):
        tracker.update([
            make_detection(*_box(350, positions_a[i])),
            make_detection(*_box(500, positions_b[i])),
        ], frame_shape=frame_shape)

    # Now only track A continues. B is unmatched and will go ghost after 4 frames.
    for i in range(len(positions_b), len(positions_a)):
        tracker.update([
            make_detection(*_box(350, positions_a[i])),
        ], frame_shape=frame_shape)

    # Collect events — A should have exited top, and B should be companion-completed
    events = tracker.get_completed_events()

    completed = [e for e in events if e.event_type == 'track_completed']
    assert len(completed) >= 2, (
        f"Expected >=2 completed (A exits + B companion), got {len(completed)}: "
        f"{[(e.track_id, e.event_type, e.ghost_exit_promoted) for e in events]}"
    )

    # One should be the normal exit, one should be ghost_exit_promoted (companion)
    promoted = [e for e in completed if e.ghost_exit_promoted]
    normal = [e for e in completed if not e.ghost_exit_promoted]
    assert len(normal) >= 1, "Expected at least 1 normal completion"
    assert len(promoted) >= 1, (
        f"Expected at least 1 companion promoted, got {len(promoted)}: "
        f"{[(e.track_id, e.ghost_exit_promoted) for e in completed]}"
    )
    print("PASS test_companion_completed_when_survivor_exits_top")


def test_companion_not_completed_if_too_far_from_top():
    """
    Ghost companion should NOT be completed if its last position was
    too far from the top exit zone (below near_top_threshold = 35% of 720 = 252).
    """
    tracker = make_tracker()
    frame_shape = (720, 1280)

    # Track A exits top. Track B only made it to y=430 (mid-frame, far from top).
    # near_top_threshold = 35% of 720 = 252, so B at y=430 is rejected.
    positions_a = [600, 570, 540, 510, 480, 450, 400, 350, 300, 250, 200, 160, 120, 80, 50, 30, 20, 10]
    positions_b = [600, 570, 540, 510, 480, 460, 440]  # stops at y=440, far from top

    for i in range(len(positions_b)):
        tracker.update([
            make_detection(*_box(350, positions_a[i])),
            make_detection(*_box(500, positions_b[i])),
        ], frame_shape=frame_shape)

    for i in range(len(positions_b), len(positions_a)):
        tracker.update([
            make_detection(*_box(350, positions_a[i])),
        ], frame_shape=frame_shape)

    events = tracker.get_completed_events()
    promoted = [e for e in events if e.event_type == 'track_completed' and e.ghost_exit_promoted]
    assert len(promoted) == 0, (
        f"Expected 0 companion promotions (B too far from top), got {len(promoted)}"
    )
    print("PASS test_companion_not_completed_if_too_far_from_top")


def test_companion_not_completed_too_few_hits():
    """Ghost companion with too few hits should not be completed."""
    tracker = make_tracker(ghost_exit_min_hits=10)
    frame_shape = (720, 1280)

    positions_a = [600, 570, 540, 510, 480, 450, 400, 350, 300, 250, 200, 160, 120, 80, 50, 30, 20, 10]
    positions_b = [620, 590, 560, 530]  # only 4 hits, then disappears

    for i in range(len(positions_b)):
        tracker.update([
            make_detection(*_box(350, positions_a[i])),
            make_detection(*_box(500, positions_b[i])),
        ], frame_shape=frame_shape)

    for i in range(len(positions_b), len(positions_a)):
        tracker.update([
            make_detection(*_box(350, positions_a[i])),
        ], frame_shape=frame_shape)

    events = tracker.get_completed_events()
    promoted = [e for e in events if e.event_type == 'track_completed' and e.ghost_exit_promoted]
    assert len(promoted) == 0, (
        f"Expected 0 companion promotions (too few hits), got {len(promoted)}"
    )
    print("PASS test_companion_not_completed_too_few_hits")


def test_no_companion_if_no_concurrent_ids():
    """
    A ghost without concurrent_track_ids should never be companion-completed.
    """
    tracker = make_tracker()
    frame_shape = (720, 1280)

    # Track A alone (no concurrent partner), exits top
    positions = [600, 540, 480, 420, 360, 300, 240, 180, 120, 60, 20]
    for y in positions:
        tracker.update([make_detection(*_box(400, y))], frame_shape=frame_shape)

    events = tracker.get_completed_events()
    promoted = [e for e in events if e.ghost_exit_promoted]
    assert len(promoted) == 0
    print("PASS test_no_companion_if_no_concurrent_ids")


# ── Ghost Near-Top Diagnostic Tests ──────────────────────────────────────────

def test_ghost_near_top_flagged_when_no_survivor_exits():
    """
    When both tracks go ghost and neither is recovered/exits,
    near-top ghosts get the diagnostic flag but stay track_lost.
    """
    tracker = make_tracker()
    frame_shape = (720, 1280)

    positions_a = [600, 520, 440, 360, 280, 230, 200, 180]
    positions_b = [620, 540, 460, 380, 300, 250, 220, 200]

    for ya, yb in zip(positions_a, positions_b):
        tracker.update([make_detection(*_box(350, ya)),
                        make_detection(*_box(450, yb))], frame_shape=frame_shape)

    # Both go ghost
    for _ in range(tracker.config.max_frames_without_detection + 1):
        tracker.update([], frame_shape=frame_shape)

    time.sleep(GHOST_EXPIRY_WAIT)
    tracker.update([], frame_shape=frame_shape)

    events = tracker.get_completed_events()
    completed = [e for e in events if e.event_type == 'track_completed']
    assert len(completed) == 0, "Ghosts without survivor should NOT be counted"

    lost = [e for e in events if e.event_type == 'track_lost']
    flagged = [e for e in lost if e.ghost_exit_promoted]
    assert len(flagged) == 2, f"Expected 2 flagged, got {len(flagged)}"
    print("PASS test_ghost_near_top_flagged_when_no_survivor_exits")


def test_ghost_no_concurrent_not_flagged():
    """Single track (no neighbours) near top should NOT be flagged."""
    tracker = make_tracker()
    frame_shape = (720, 1280)

    positions = [600, 520, 440, 360, 280, 230, 200, 180]
    for y in positions:
        tracker.update([make_detection(*_box(400, y))], frame_shape=frame_shape)

    for _ in range(tracker.config.max_frames_without_detection + 1):
        tracker.update([], frame_shape=frame_shape)

    time.sleep(GHOST_EXPIRY_WAIT)
    tracker.update([], frame_shape=frame_shape)

    events = tracker.get_completed_events()
    lost = [e for e in events if e.event_type == 'track_lost']
    assert len(lost) == 1
    assert lost[0].ghost_exit_promoted is False
    print("PASS test_ghost_no_concurrent_not_flagged")


def test_no_double_count_on_bag_replacement():
    """
    After ghost expires (track_lost, not counted), placing bag back
    creates a new track that can be counted — no double counting.
    """
    tracker = make_tracker()
    frame_shape = (720, 1280)

    positions_a = [600, 520, 440, 360, 280, 230, 200, 180]
    positions_b = [620, 540, 460, 380, 300, 250, 220, 200]
    for ya, yb in zip(positions_a, positions_b):
        tracker.update([make_detection(*_box(350, ya)),
                        make_detection(*_box(450, yb))], frame_shape=frame_shape)

    for _ in range(tracker.config.max_frames_without_detection + 1):
        tracker.update([], frame_shape=frame_shape)

    time.sleep(GHOST_EXPIRY_WAIT)
    tracker.update([], frame_shape=frame_shape)

    events = tracker.get_completed_events()
    completed = [e for e in events if e.event_type == 'track_completed']
    assert len(completed) == 0  # no counting happened

    # Bag placed back — should create new track freely
    tracker.update([make_detection(*_box(350, 600))], frame_shape=frame_shape)
    assert len(tracker.tracks) == 1
    print("PASS test_no_double_count_on_bag_replacement")


def test_ghost_validation_disabled():
    """Ghost exit validation disabled means no flags, no companions."""
    tracker = make_tracker(ghost_exit_validation_enabled=False)
    frame_shape = (720, 1280)

    positions_a = [600, 520, 440, 360, 280, 230, 200, 180]
    positions_b = [620, 540, 460, 380, 300, 250, 220, 200]
    for ya, yb in zip(positions_a, positions_b):
        tracker.update([make_detection(*_box(350, ya)),
                        make_detection(*_box(450, yb))], frame_shape=frame_shape)

    for _ in range(tracker.config.max_frames_without_detection + 1):
        tracker.update([], frame_shape=frame_shape)

    time.sleep(GHOST_EXPIRY_WAIT)
    tracker.update([], frame_shape=frame_shape)

    events = tracker.get_completed_events()
    flagged = [e for e in events if e.ghost_exit_promoted]
    assert len(flagged) == 0
    print("PASS test_ghost_validation_disabled")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    test_companion_completed_when_survivor_exits_top()
    test_companion_not_completed_if_too_far_from_top()
    test_companion_not_completed_too_few_hits()
    test_no_companion_if_no_concurrent_ids()
    test_ghost_near_top_flagged_when_no_survivor_exits()
    test_ghost_no_concurrent_not_flagged()
    test_no_double_count_on_bag_replacement()
    test_ghost_validation_disabled()
    print("\nAll ghost companion + diagnostic tests passed!")








