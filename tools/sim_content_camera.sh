#!/usr/bin/env bash
# sim_content_camera.sh — Simulate the second (content) camera on a Linux
# developer machine by serving a looping video file over RTSP.
#
# Usage:
#   ./tools/sim_content_camera.sh [video_file]
#
# Default video:  data/test_round_tip.avi  (the existing QR test clip)
# Custom example: ./tools/sim_content_camera.sh /path/to/my_clip.mp4
#
# What it does:
#   1. Downloads the MediaMTX single binary if not already present.
#   2. Starts MediaMTX as a background RTSP server (port 8554).
#   3. Uses ffmpeg to loop the video and push it to MediaMTX via RTSP.
#
# The content recorder in the app then connects to:
#   rtsp://sim:sim@127.0.0.1:8554/cam/realmonitor
# (credentials are accepted by MediaMTX because auth is disabled in
# tools/mediamtx.yml)
#
# To configure the app DB to use this sim camera, run:
#   python tools/configure_sim.py --enable
#
# Stop:
#   Press Ctrl-C  (both ffmpeg and mediamtx are killed via trap)
#
# Production note:
#   Nothing in this file touches production config.  On the RDK board the
#   real camera at 192.168.2.128:554 is used directly — no MediaMTX needed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
TOOLS_DIR="$SCRIPT_DIR"
MEDIAMTX_BIN="$TOOLS_DIR/mediamtx"
MEDIAMTX_CFG="$TOOLS_DIR/mediamtx.yml"

# ── Video source ────────────────────────────────────────────────────────────
VIDEO="${1:-$REPO_ROOT/data/test_round_tip.avi}"
if [[ ! -f "$VIDEO" ]]; then
    echo "[sim] ERROR: Video file not found: $VIDEO"
    exit 1
fi

# ── Download MediaMTX if needed ─────────────────────────────────────────────
MEDIAMTX_VERSION="v1.9.3"
MEDIAMTX_ARCHIVE="$TOOLS_DIR/mediamtx.tar.gz"

if [[ ! -x "$MEDIAMTX_BIN" ]]; then
    ARCH="$(uname -m)"
    case "$ARCH" in
        x86_64)  MTX_ARCH="amd64" ;;
        aarch64) MTX_ARCH="arm64v8" ;;
        armv7l)  MTX_ARCH="armv7" ;;
        *)
            echo "[sim] ERROR: Unsupported architecture: $ARCH"
            exit 1
            ;;
    esac

    URL="https://github.com/bluenviron/mediamtx/releases/download/${MEDIAMTX_VERSION}/mediamtx_${MEDIAMTX_VERSION}_linux_${MTX_ARCH}.tar.gz"
    echo "[sim] Downloading MediaMTX ${MEDIAMTX_VERSION} for ${MTX_ARCH}..."
    curl -L --progress-bar "$URL" -o "$MEDIAMTX_ARCHIVE"
    tar -xzf "$MEDIAMTX_ARCHIVE" -C "$TOOLS_DIR" mediamtx
    rm -f "$MEDIAMTX_ARCHIVE"
    chmod +x "$MEDIAMTX_BIN"
    echo "[sim] MediaMTX downloaded to $MEDIAMTX_BIN"
fi

# ── Cleanup on exit ──────────────────────────────────────────────────────────
MTX_PID=""
FFMPEG_PID=""
cleanup() {
    echo ""
    echo "[sim] Shutting down..."
    [[ -n "$FFMPEG_PID" ]] && kill "$FFMPEG_PID" 2>/dev/null || true
    [[ -n "$MTX_PID"    ]] && kill "$MTX_PID"    2>/dev/null || true
    wait
    echo "[sim] Done."
}
trap cleanup EXIT INT TERM

# ── Start MediaMTX ───────────────────────────────────────────────────────────
echo "[sim] Starting MediaMTX RTSP server on :8554..."
"$MEDIAMTX_BIN" "$MEDIAMTX_CFG" &
MTX_PID=$!

# Wait until MediaMTX is actually accepting connections (up to 10s).
echo -n "[sim] Waiting for MediaMTX to bind..."
for i in $(seq 1 20); do
    if nc -z 127.0.0.1 8554 2>/dev/null; then
        echo " ready."
        break
    fi
    sleep 0.5
done
if ! nc -z 127.0.0.1 8554 2>/dev/null; then
    echo ""
    echo "[sim] ERROR: MediaMTX did not start within 10s. Check the config."
    exit 1
fi

# ── Push video via ffmpeg ────────────────────────────────────────────────────
RTSP_PUSH="rtsp://127.0.0.1:8554/cam/realmonitor"
echo "[sim] Streaming $VIDEO → $RTSP_PUSH (looping)"
echo "[sim] Content recorder should connect to:"
echo "      rtsp://sim:sim@127.0.0.1:8554/cam/realmonitor"
echo ""
echo "[sim] Press Ctrl-C to stop."
echo ""

ffmpeg \
    -re \
    -stream_loop -1 \
    -i "$VIDEO" \
    -c:v libx264 \
    -preset ultrafast \
    -tune zerolatency \
    -pix_fmt yuv420p \
    -b:v 1000k \
    -an \
    -f rtsp \
    -rtsp_transport tcp \
    "$RTSP_PUSH" &
FFMPEG_PID=$!

wait "$FFMPEG_PID"
