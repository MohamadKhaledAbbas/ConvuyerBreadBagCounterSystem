#!/usr/bin/env python3
"""
configure_sim.py — Configure the app DB for content-camera simulation
(dev machine) or restore production settings (RDK board).

Usage
-----
  # Enable simulation mode (local MediaMTX at 127.0.0.1:8554)
  python tools/configure_sim.py --enable

  # Choose video source: 'content' or 'qr' (default: content)
  python tools/configure_sim.py --enable --video-source qr

  # Show current content-camera config
  python tools/configure_sim.py --show

  # Restore production defaults (real camera at 192.168.2.128:554)
  python tools/configure_sim.py --production \
      --host 192.168.2.128 --port 554 \
      --user admin --password YourPassword

  # Disable content recording entirely (go back to QR-only)
  python tools/configure_sim.py --disable
"""

import argparse
import os
import sqlite3
import sys

DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "db", "bag_events.db",
)

# Keys that live in the config table
CONTENT_KEYS = [
    "content_recording_enabled",
    "content_rtsp_host",
    "content_rtsp_port",
    "content_rtsp_username",
    "content_rtsp_password",
    "container_event_video_source",
]


def upsert(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO config(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )


def show(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        "SELECT key, value FROM config WHERE key IN ({})".format(
            ", ".join("?" * len(CONTENT_KEYS))
        ),
        CONTENT_KEYS,
    ).fetchall()
    kv = dict(rows)
    print("\nCurrent content-camera DB config:")
    print("-" * 45)
    for k in CONTENT_KEYS:
        v = kv.get(k, "<not set>")
        # Mask password
        display = "***" if "password" in k and v not in ("<not set>", "") else v
        print(f"  {k:<40s} = {display}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--enable",     action="store_true",
                      help="Enable sim mode (local MediaMTX)")
    mode.add_argument("--disable",    action="store_true",
                      help="Disable content recording (QR only)")
    mode.add_argument("--production", action="store_true",
                      help="Set production camera credentials")
    mode.add_argument("--show",       action="store_true",
                      help="Show current config and exit")

    parser.add_argument("--video-source", choices=["content", "qr"], default="content",
                        help="Which camera produces event clips (default: content)")
    parser.add_argument("--host",     default="192.168.2.128",
                        help="Production RTSP host (--production only)")
    parser.add_argument("--port",     default="554",
                        help="Production RTSP port (--production only)")
    parser.add_argument("--user",     default="admin",
                        help="Production RTSP username (--production only)")
    parser.add_argument("--password", default="",
                        help="Production RTSP password (--production only)")

    args = parser.parse_args()

    if not os.path.exists(DB_PATH):
        print(f"ERROR: DB not found at {DB_PATH}", file=sys.stderr)
        print("Run the app once first to create the DB.", file=sys.stderr)
        sys.exit(1)

    with sqlite3.connect(DB_PATH) as conn:
        if args.show:
            show(conn)
            return

        if args.enable:
            upsert(conn, "content_recording_enabled", "1")
            upsert(conn, "content_rtsp_host",         "127.0.0.1")
            upsert(conn, "content_rtsp_port",         "8554")
            upsert(conn, "content_rtsp_username",     "sim")
            upsert(conn, "content_rtsp_password",     "sim")
            upsert(conn, "container_event_video_source", args.video_source)
            conn.commit()
            print("✓ Simulation mode enabled.")
            print("  RTSP source : rtsp://sim:sim@127.0.0.1:8554/cam/realmonitor")
            print(f"  Video source: {args.video_source}")
            print()
            print("Next steps:")
            print("  1. In a separate terminal:")
            print("       bash tools/sim_content_camera.sh")
            print("  2. Start the app normally:")
            print("       python container_main.py")

        elif args.disable:
            upsert(conn, "content_recording_enabled",    "0")
            upsert(conn, "container_event_video_source", "qr")
            conn.commit()
            print("✓ Content recording disabled. App will use QR camera only.")

        elif args.production:
            upsert(conn, "content_recording_enabled", "1")
            upsert(conn, "content_rtsp_host",         args.host)
            upsert(conn, "content_rtsp_port",         args.port)
            upsert(conn, "content_rtsp_username",     args.user)
            upsert(conn, "content_rtsp_password",     args.password)
            upsert(conn, "container_event_video_source", args.video_source)
            conn.commit()
            print("✓ Production config set.")
            print(f"  RTSP source : rtsp://{args.user}:***@{args.host}:{args.port}/cam/realmonitor?channel=1&subtype=0")
            print(f"  Video source: {args.video_source}")

        show(conn)


if __name__ == "__main__":
    main()
