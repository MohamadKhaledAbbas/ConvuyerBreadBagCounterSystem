#!/usr/bin/env python3
"""
set_configs.py — Interactive configuration script for ConveyorBreadBagCounterSystem.

Run this on the RDK board (or dev machine) to seed / update the SQLite config table.
Each section has sensible defaults pre-filled; press Enter to keep the current DB value
or the built-in default when the key doesn't exist yet.

Usage:
    python scripts/set_configs.py [--db PATH] [--non-interactive]

    --db PATH           Override the DB path (default: data/db/bag_events.db)
    --non-interactive   Write all defaults without prompting (useful for first-boot).
"""

import argparse
import os
import sqlite3
import sys


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DB_DEFAULT = os.path.join(os.path.dirname(__file__), "..", "data", "db", "bag_events.db")


def _connect(db_path: str) -> sqlite3.Connection:
    db_path = os.path.abspath(db_path)
    if not os.path.exists(db_path):
        print(f"[ERROR] DB not found: {db_path}")
        sys.exit(1)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _get(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM config WHERE key=?", (key,)).fetchone()
    return row[0] if row else None


def _set(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO config (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )


def _ask(conn: sqlite3.Connection, key: str, default: str, label: str,
         non_interactive: bool) -> str:
    current = _get(conn, key)
    display_default = current if current is not None else default
    if non_interactive:
        value = display_default
    else:
        raw = input(f"  {label} [{display_default}]: ").strip()
        value = raw if raw else display_default
    return value


def _section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ---------------------------------------------------------------------------
# Config sections
# ---------------------------------------------------------------------------

def configure_qr_camera(conn, ni):
    _section("QR Camera (RTSP — container tracking)")
    configs = [
        ("container_rtsp_host",     "192.168.2.118", "IP address"),
        ("container_rtsp_port",     "554",           "RTSP port"),
        ("container_rtsp_username", "admin",         "Username"),
        ("container_rtsp_password", "",              "Password"),
    ]
    for key, default, label in configs:
        _set(conn, key, _ask(conn, key, default, label, ni))


def configure_content_camera(conn, ni):
    _section("Content Camera (optional 3D-angle recorder)")
    enabled = _ask(conn, "content_recording_enabled", "0",
                   "Enable content recording? (1=yes / 0=no)", ni)
    _set(conn, "content_recording_enabled", enabled)
    if enabled == "1" or not ni:
        configs = [
            ("content_rtsp_host",     "192.168.2.128", "Content camera IP"),
            ("content_rtsp_port",     "554",           "RTSP port"),
            ("content_rtsp_username", "admin",         "Username"),
            ("content_rtsp_password", "",              "Password"),
            ("content_rtsp_subtype",  "0",             "Subtype (0=main, 1=sub)"),
        ]
        for key, default, label in configs:
            _set(conn, key, _ask(conn, key, default, label, ni))


def configure_tracking(conn, ni):
    _section("Container Tracking Behaviour")
    configs = [
        ("container_exit_zone_ratio",          "0.05", "Exit-zone width ratio (0–0.5)"),
        ("container_lost_timeout",             "2.0",  "Lost-track timeout (seconds)"),
        ("container_detect_interval",          "3",    "Run QR detection every N frames"),
        ("container_qr_engine",               "auto", "QR engine: auto (prefer wechat), wechat, legacy"),
        ("container_min_detections_for_event", "3",    "Min detections to confirm an event"),
    ]
    for key, default, label in configs:
        _set(conn, key, _ask(conn, key, default, label, ni))


def configure_snapshots(conn, ni):
    _section("Snapshot Ring Buffer")
    configs = [
        ("container_pre_event_seconds",  "5.0", "Pre-event buffer (seconds)"),
        ("container_post_event_seconds", "5.0", "Post-event capture (seconds)"),
    ]
    for key, default, label in configs:
        _set(conn, key, _ask(conn, key, default, label, ni))


def configure_event_video(conn, ni):
    _section("Event Video (QR-camera clip saved on each container event)")
    configs = [
        ("container_event_video_source",        "qr",  "Video source (qr / content)"),
        ("container_event_video_fps",           "20",  "Output FPS"),
        ("container_event_video_max_seconds",   "5.0", "Max clip length (seconds)"),
        ("container_event_video_stationary_px", "5",   "Stationary pixel threshold"),
        ("content_pre_event_seconds",           "3.0", "Content pre-event (seconds)"),
        ("content_post_event_seconds",          "2.0", "Content post-event (seconds)"),
        ("content_buffer_seconds",              "5.0", "Content ring buffer size (seconds)"),
        ("content_video_fps",                   "10",  "Content video FPS"),
        ("content_max_recording_seconds",       "15.0","Content max clip length (seconds)"),
    ]
    for key, default, label in configs:
        _set(conn, key, _ask(conn, key, default, label, ni))


def configure_purge(conn, ni):
    _section("Data Retention / Auto-Purge")
    configs = [
        ("container_snapshots_retention_hours",      "72.0",  "Snapshots retention (hours)"),
        ("container_snapshots_max_count",            "500",   "Snapshots max count"),
        ("container_content_videos_retention_hours", "72.0",  "Content videos retention (hours)"),
        ("container_content_videos_max_count",       "200",   "Content videos max count"),
        ("container_db_events_retention_hours",      "168.0", "DB events retention (hours)"),
        ("container_purge_interval_minutes",         "60.0",  "Purge check interval (minutes)"),
    ]
    for key, default, label in configs:
        _set(conn, key, _ask(conn, key, default, label, ni))


def configure_ui_cards(conn, ni):
    _section("Dashboard Card Visibility (1=visible, 0=hidden)")
    configs = [
        ("container_ui_card_visible",       "0", "Container QR tracking card"),
        ("ui_card_counts_visible",          "1", "Counts card"),
        ("ui_card_analytics_visible",       "1", "Analytics card"),
        ("ui_card_analytics_daily_visible", "1", "Daily analytics card"),
        ("ui_card_lost_tracks_visible",     "1", "Lost tracks card"),
        ("ui_card_track_events_visible",    "1", "Track events card"),
        ("ui_card_snapshot_visible",        "1", "Snapshot card"),
        ("ui_card_endpoints_visible",       "1", "Endpoints card"),
        ("ui_card_guidelines_visible",      "1", "Guidelines card"),
    ]
    for key, default, label in configs:
        _set(conn, key, _ask(conn, key, default, label, ni))


def configure_display(conn, ni):
    _section("Display / Debug")
    _set(conn, "container_enable_display",
         _ask(conn, "container_enable_display", "0",
              "Enable display window on board? (1=yes / 0=no)", ni))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SECTIONS = [
    configure_qr_camera,
    configure_content_camera,
    configure_tracking,
    configure_snapshots,
    configure_event_video,
    configure_purge,
    configure_ui_cards,
    configure_display,
]


def main():
    parser = argparse.ArgumentParser(description="Seed / update app config in the SQLite DB.")
    parser.add_argument("--db", default=DB_DEFAULT, help="Path to bag_events.db")
    parser.add_argument("--non-interactive", action="store_true",
                        help="Write defaults without prompting")
    args = parser.parse_args()

    ni = args.non_interactive
    conn = _connect(args.db)

    print("=" * 60)
    print("  ConveyorBreadBagCounterSystem — Config Setup")
    print(f"  DB: {os.path.abspath(args.db)}")
    if ni:
        print("  Mode: non-interactive (writing defaults)")
    else:
        print("  Press Enter to keep the value shown in [brackets].")
    print("=" * 60)

    try:
        for section_fn in SECTIONS:
            section_fn(conn, ni)
        conn.commit()
        print("\n[OK] Config saved successfully.\n")
    except KeyboardInterrupt:
        print("\n[ABORTED] No changes committed.")
        conn.rollback()
    finally:
        conn.close()


if __name__ == "__main__":
    main()
