#!/usr/bin/env bash
# sim_content_camera2.sh — Simulate the third (content2) camera on a Linux
# developer machine by serving a looping video file over RTSP on port 8555.
#
# Usage:
#   ./tools/sim_content_camera2.sh [video_file]
#
# Default video:  /home/khaled/Downloads/cam_content.mp4
# Custom example: ./tools/sim_content_camera2.sh /path/to/my_clip.mp4
#
# What it does:
#   1. Uses the existing MediaMTX binary (downloads if needed).
#   2. Starts MediaMTX as a background RTSP server (port 8555).
#   3. Uses ffmpeg to loop the video and push it to MediaMTX via RTSP.
#
# The content2 recorder in the app then connects to:
#   rtsp://sim:sim@127.0.0.1:8555/cam/realmonitor
#
# ``container_main.py`` now auto-detects this local simulator in
# development mode when it is reachable at 127.0.0.1:8555, so the
# normal local workflow is simply:
#   1. ./tools/sim_content_camera.sh  (for content1 on port 8554)
#   2. ./tools/sim_content_camera2.sh (for content2 on port 8555)
#   3. python container_main.py
#
# Stop:
#   Press Ctrl-C  (both ffmpeg and mediamtx are killed via trap)
#
# Production note:
#   Nothing in this file touches production config.  On the RDK board the
#   real camera at 192.168.2.138:554 is used directly — no MediaMTX needed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
TOOLS_DIR="$SCRIPT_DIR"
MEDIAMTX_BIN="$TOOLS_DIR/mediamtx"
MEDIAMTX_CFG="$TOOLS_DIR/mediamtx.yml"

# ── Video source ────────────────────────────────────────────────────────────
VIDEO="${1:-/home/khaled/Downloads/cam_content2.mp4}"
if [[ ! -f "$VIDEO" ]]; then
    echo "[sim2] ERROR: Video file not found: $VIDEO"
    exit 1
fi

# ── Download MediaMTX if needed (shared with content1 sim) ───────────────────
MEDIAMTX_VERSION="v1.9.3"
MEDIAMTX_ARCHIVE="$TOOLS_DIR/mediamtx.tar.gz"

if [[ ! -x "$MEDIAMTX_BIN" ]]; then
    ARCH="$(uname -m)"
    case "$ARCH" in
        x86_64)  MTX_ARCH="amd64" ;;
        aarch64) MTX_ARCH="arm64v8" ;;
        armv7l)  MTX_ARCH="armv7" ;;
        *)
            echo "[sim2] ERROR: Unsupported architecture: $ARCH"
            exit 1
            ;;
    esac

    URL="https://github.com/bluenviron/mediamtx/releases/download/${MEDIAMTX_VERSION}/mediamtx_${MEDIAMTX_VERSION}_linux_${MTX_ARCH}.tar.gz"
    echo "[sim2] Downloading MediaMTX ${MEDIAMTX_VERSION} for ${MTX_ARCH}..."
    curl -L --progress-bar "$URL" -o "$MEDIAMTX_ARCHIVE"
    tar -xzf "$MEDIAMTX_ARCHIVE" -C "$TOOLS_DIR" mediamtx
    rm -f "$MEDIAMTX_ARCHIVE"
    chmod +x "$MEDIAMTX_BIN"
    echo "[sim2] MediaMTX downloaded to $MEDIAMTX_BIN"
fi

# ── Cleanup on exit ──────────────────────────────────────────────────────────
FFMPEG_PID=""
cleanup() {
    echo ""
    echo "[sim2] Shutting down..."
    [[ -n "$FFMPEG_PID" ]] && kill "$FFMPEG_PID" 2>/dev/null || true
    wait
    echo "[sim2] Done."
}
trap cleanup EXIT INT TERM

# ── Check if MediaMTX is already running (from sim_content_camera.sh) ───────
echo -n "[sim2] Checking for MediaMTX on port 8554..."
if ! nc -z 127.0.0.1 8554 2>/dev/null; then
    echo ""
    echo "[sim2] ERROR: MediaMTX not running on port 8554."
    echo "[sim2] Please start it first with: ./tools/sim_content_camera.sh"
    exit 1
fi
echo " running."

# ── Push video via ffmpeg ────────────────────────────────────────────────────
RTSP_PUSH="rtsp://127.0.0.1:8554/cam/realmonitor2"
echo "[sim2] Streaming $VIDEO → $RTSP_PUSH (looping)"
echo "[sim2] Content recorder 2 should connect to:"
echo "      rtsp://sim:sim@127.0.0.1:8554/cam/realmonitor2"
echo "[sim2] Note: This uses the same MediaMTX instance as content1 (port 8554)."
echo ""
echo "[sim2] Press Ctrl-C to stop."
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
