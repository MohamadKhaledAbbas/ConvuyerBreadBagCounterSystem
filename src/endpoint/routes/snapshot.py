"""
Snapshot Routes - On-demand camera frame snapshot endpoint.

Provides browser-accessible endpoints for viewing the current camera feed:
- /snapshot - Request and get the latest frame as JPEG
- /snapshot?overlay=true - Frame with detection overlays
- /snapshot/view - HTML page with manual/auto-refresh

On-Demand Architecture:
- Browser requests /snapshot
- Endpoint sets snapshot_requested flag in database to "1"
- Main app (ConveyorCounterApp) checks flag, captures frame, writes to disk
- Main app sets flag back to "0"
- Endpoint polls for the new snapshot file and returns it

This approach minimizes disk I/O - frames are only written when requested.
"""

import json
import os
import time
from typing import Optional

from fastapi import APIRouter, Query
from fastapi.responses import Response, HTMLResponse

from src.constants import snapshot_requested_key
from src.endpoint.shared import get_db
from src.utils.AppLogging import logger

router = APIRouter(tags=["snapshot"])

# Default snapshot directory
SNAPSHOT_DIR = os.getenv("SNAPSHOT_DIR", "data/snapshot")
SNAPSHOT_RAW_PATH = os.path.join(SNAPSHOT_DIR, "latest_raw.jpg")
SNAPSHOT_OVERLAY_PATH = os.path.join(SNAPSHOT_DIR, "latest_overlay.jpg")
SNAPSHOT_META_PATH = os.path.join(SNAPSHOT_DIR, "latest_meta.json")


class SnapshotWriter:
    """
    Writes frames to disk on-demand.

    Used by ConveyorCounterApp when snapshot_requested flag is set.
    """

    def __init__(self, snapshot_dir: str = SNAPSHOT_DIR):
        self._snapshot_dir = snapshot_dir
        self._raw_path = os.path.join(snapshot_dir, "latest_raw.jpg")
        self._overlay_path = os.path.join(snapshot_dir, "latest_overlay.jpg")
        self._meta_path = os.path.join(snapshot_dir, "latest_meta.json")

        # Ensure directory exists
        os.makedirs(snapshot_dir, exist_ok=True)

    def write_snapshot(
        self,
        frame,
        frame_with_overlay=None,
        frame_number: int = 0,
        quality: int = 85
    ) -> bool:
        """
        Write snapshot to disk.

        Args:
            frame: Raw BGR frame (numpy array)
            frame_with_overlay: Frame with detection/tracking overlays
            frame_number: Current frame number
            quality: JPEG quality (1-100)

        Returns:
            True if written successfully
        """
        try:
            import cv2
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]

            # Write raw frame
            cv2.imwrite(self._raw_path, frame, encode_params)

            # Write overlay frame if provided
            if frame_with_overlay is not None:
                cv2.imwrite(self._overlay_path, frame_with_overlay, encode_params)

            # Write metadata
            meta = {
                "frame_number": frame_number,
                "timestamp": time.time(),
                "width": frame.shape[1],
                "height": frame.shape[0]
            }
            with open(self._meta_path, 'w') as f:
                json.dump(meta, f)

            return True

        except Exception as e:
            logger.error(f"[SnapshotWriter] Failed to write snapshot: {e}")
            return False


# Global writer instance
_snapshot_writer: Optional[SnapshotWriter] = None


def get_snapshot_writer() -> SnapshotWriter:
    """Get or create the global snapshot writer instance."""
    global _snapshot_writer
    if _snapshot_writer is None:
        _snapshot_writer = SnapshotWriter()
    return _snapshot_writer


def _read_snapshot(overlay: bool = False) -> tuple[Optional[bytes], dict]:
    """
    Read the latest snapshot from disk.

    Args:
        overlay: If True, read the overlay version

    Returns:
        Tuple of (jpeg_bytes, metadata) or (None, {}) if not available
    """
    path = SNAPSHOT_OVERLAY_PATH if overlay else SNAPSHOT_RAW_PATH

    # Check if overlay exists, fall back to raw
    if overlay and not os.path.exists(path):
        path = SNAPSHOT_RAW_PATH

    if not os.path.exists(path):
        return None, {}

    try:
        with open(path, 'rb') as f:
            jpeg_bytes = f.read()

        meta = {}
        if os.path.exists(SNAPSHOT_META_PATH):
            with open(SNAPSHOT_META_PATH, 'r') as f:
                meta = json.load(f)

        return jpeg_bytes, meta

    except Exception:
        return None, {}


def _request_snapshot() -> bool:
    """
    Set the snapshot_requested flag in database to trigger capture.

    Returns:
        True if flag was set successfully
    """
    try:
        db = get_db()
        db.set_config(snapshot_requested_key, "1")
        return True
    except Exception as e:
        logger.error(f"[Snapshot] Failed to set snapshot request flag: {e}")
        return False


def _get_snapshot_timestamp() -> float:
    """Get the timestamp of the current snapshot file."""
    if os.path.exists(SNAPSHOT_META_PATH):
        try:
            with open(SNAPSHOT_META_PATH, 'r') as f:
                meta = json.load(f)
            return meta.get("timestamp", 0)
        except Exception:
            pass
    return 0


@router.get("/snapshot")
async def snapshot(
    overlay: bool = Query(False, description="Include detection overlays"),
    timeout: float = Query(3.0, ge=0.5, le=10.0, description="Max wait time for new snapshot")
) -> Response:
    """
    Request and get the latest camera frame as a JPEG image.

    This triggers the main app to capture a new frame. The endpoint waits
    for the new frame to be written (up to timeout seconds).

    Args:
        overlay: Include detection/tracking overlays
        timeout: Maximum time to wait for new snapshot (seconds)

    Returns:
        JPEG image response or 503 if capture failed
    """
    # Get current snapshot timestamp before requesting new one
    old_timestamp = _get_snapshot_timestamp()

    # Request new snapshot
    if not _request_snapshot():
        return Response(
            content="Failed to request snapshot - database error",
            status_code=500,
            media_type="text/plain"
        )

    # Wait for new snapshot to be written
    start_time = time.time()
    while time.time() - start_time < timeout:
        new_timestamp = _get_snapshot_timestamp()
        if new_timestamp > old_timestamp:
            # New snapshot available
            break
        time.sleep(0.05)  # Poll every 50ms

    # Read the snapshot
    jpeg_bytes, meta = _read_snapshot(overlay=overlay)

    if jpeg_bytes is None:
        return Response(
            content="No snapshot available - main.py may not be running",
            status_code=503,
            media_type="text/plain"
        )

    timestamp = meta.get("timestamp", 0)
    age_ms = (time.time() - timestamp) * 1000 if timestamp else 0
    frame_num = meta.get("frame_number", 0)

    # Check if we got a fresh snapshot
    is_fresh = timestamp > old_timestamp

    return Response(
        content=jpeg_bytes,
        media_type="image/jpeg",
        headers={
            "X-Frame-Number": str(frame_num),
            "X-Frame-Age-Ms": f"{age_ms:.0f}",
            "X-Frame-Fresh": "true" if is_fresh else "false",
            "Cache-Control": "no-cache, no-store, must-revalidate",
        }
    )


@router.get("/snapshot/info")
async def snapshot_info() -> dict:
    """Get information about the current snapshot."""
    _, meta = _read_snapshot()

    if not meta:
        return {
            "available": False,
            "frame_number": 0,
            "timestamp": 0,
            "age_seconds": None,
        }

    timestamp = meta.get("timestamp", 0)

    return {
        "available": True,
        "frame_number": meta.get("frame_number", 0),
        "timestamp": timestamp,
        "age_seconds": time.time() - timestamp if timestamp else None,
        "width": meta.get("width"),
        "height": meta.get("height")
    }


@router.get("/snapshot/view", response_class=HTMLResponse)
async def snapshot_view(
    refresh: float = Query(0, ge=0, le=60.0, description="Auto-refresh interval (0=manual)"),
    overlay: bool = Query(True, description="Include detection overlays")
) -> HTMLResponse:
    """
    HTML page for viewing snapshots with manual or auto-refresh.

    Args:
        refresh: Auto-refresh interval in seconds (0 = manual only)
        overlay: Include detection overlays
    """
    overlay_param = "true" if overlay else "false"
    auto_refresh_js = f"setInterval(refreshNow, {int(refresh * 1000)});" if refresh > 0 else ""

    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Conveyor Camera Snapshot</title>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            background: #1a1a2e;
            color: #eee;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ color: #00d9ff; margin-bottom: 10px; }}
        .info-bar {{
            display: flex; gap: 20px; margin-bottom: 15px; padding: 10px 15px;
            background: #16213e; border-radius: 8px; font-size: 14px; flex-wrap: wrap;
        }}
        .info-item {{ display: flex; align-items: center; gap: 5px; }}
        .info-label {{ color: #888; }}
        .info-value {{ color: #00d9ff; font-weight: 600; }}
        .status-dot {{
            width: 10px; height: 10px; border-radius: 50%;
            background: #888; transition: background 0.3s;
        }}
        .status-dot.live {{ background: #00ff88; }}
        .status-dot.stale {{ background: #ff4444; }}
        .status-dot.loading {{ background: #ffaa00; animation: pulse 0.5s infinite; }}
        @keyframes pulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.5; }} }}
        .frame-container {{
            position: relative; background: #0f0f23; border-radius: 8px;
            overflow: hidden; box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            min-height: 400px;
        }}
        #snapshot {{ display: block; width: 100%; height: auto; }}
        .loading-overlay {{
            position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
            background: rgba(0,0,0,0.7); padding: 20px 40px; border-radius: 8px;
            display: none;
        }}
        .controls {{ margin-top: 15px; display: flex; gap: 10px; flex-wrap: wrap; }}
        button {{
            padding: 12px 24px; border: none; border-radius: 6px;
            cursor: pointer; font-size: 14px; font-weight: 600;
            transition: transform 0.1s, box-shadow 0.1s;
        }}
        button:hover {{ transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,0,0,0.3); }}
        button:disabled {{ opacity: 0.5; cursor: not-allowed; transform: none; }}
        .btn-primary {{ background: #00d9ff; color: #1a1a2e; }}
        .btn-secondary {{ background: #16213e; color: #eee; border: 1px solid #333; }}
        select {{
            padding: 12px; border-radius: 6px; border: 1px solid #333;
            background: #16213e; color: #eee; font-size: 14px;
        }}
        .note {{
            margin-top: 20px; padding: 15px; background: #16213e;
            border-radius: 8px; border-left: 4px solid #00d9ff;
            font-size: 13px; color: #aaa;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📸 Conveyor Camera Snapshot</h1>
        
        <div class="info-bar">
            <div class="info-item">
                <div class="status-dot" id="statusDot"></div>
                <span class="info-label">Status:</span>
                <span class="info-value" id="status">Ready</span>
            </div>
            <div class="info-item">
                <span class="info-label">Frame:</span>
                <span class="info-value" id="frameNum">-</span>
            </div>
            <div class="info-item">
                <span class="info-label">Age:</span>
                <span class="info-value" id="frameAge">-</span>
            </div>
            <div class="info-item">
                <span class="info-label">Load time:</span>
                <span class="info-value" id="loadTime">-</span>
            </div>
        </div>
        
        <div class="frame-container">
            <img id="snapshot" src="" alt="Click 'Capture Snapshot' to get image">
            <div class="loading-overlay" id="loadingOverlay">
                ⏳ Capturing...
            </div>
        </div>
        
        <div class="controls">
            <button class="btn-primary" id="captureBtn" onclick="refreshNow()">
                📷 Capture Snapshot
            </button>
            <button class="btn-secondary" onclick="toggleOverlay()">
                {"👁️ Hide Overlay" if overlay else "👁️ Show Overlay"}
            </button>
            <select id="autoRefresh" onchange="updateAutoRefresh()">
                <option value="0" {"selected" if refresh == 0 else ""}>Manual</option>
                <option value="1" {"selected" if refresh == 1 else ""}>Auto 1s</option>
                <option value="2" {"selected" if refresh == 2 else ""}>Auto 2s</option>
                <option value="5" {"selected" if refresh == 5 else ""}>Auto 5s</option>
            </select>
        </div>
        
        <div class="note">
            <strong>On-Demand Mode:</strong> Snapshots are captured only when you click the button
            or enable auto-refresh. This minimizes system load.
        </div>
    </div>
    
    <script>
        let overlay = {str(overlay).lower()};
        let isLoading = false;
        
        const img = document.getElementById('snapshot');
        const statusDot = document.getElementById('statusDot');
        const statusText = document.getElementById('status');
        const frameNumText = document.getElementById('frameNum');
        const frameAgeText = document.getElementById('frameAge');
        const loadTimeText = document.getElementById('loadTime');
        const loadingOverlay = document.getElementById('loadingOverlay');
        const captureBtn = document.getElementById('captureBtn');
        
        async function refreshNow() {{
            if (isLoading) return;
            
            isLoading = true;
            captureBtn.disabled = true;
            statusDot.className = 'status-dot loading';
            statusText.textContent = 'Capturing...';
            loadingOverlay.style.display = 'block';
            
            const startTime = performance.now();
            const timestamp = new Date().getTime();
            
            try {{
                const response = await fetch('/snapshot?overlay=' + overlay + '&_t=' + timestamp);
                
                if (response.ok) {{
                    const blob = await response.blob();
                    const url = URL.createObjectURL(blob);
                    img.src = url;
                    
                    const loadTime = Math.round(performance.now() - startTime);
                    loadTimeText.textContent = loadTime + 'ms';
                    
                    const frameNum = response.headers.get('X-Frame-Number') || '-';
                    const ageMs = response.headers.get('X-Frame-Age-Ms') || '0';
                    const isFresh = response.headers.get('X-Frame-Fresh') === 'true';
                    
                    frameNumText.textContent = '#' + frameNum;
                    frameAgeText.textContent = Math.round(parseFloat(ageMs)) + 'ms';
                    
                    statusDot.className = isFresh ? 'status-dot live' : 'status-dot stale';
                    statusText.textContent = isFresh ? 'Fresh' : 'Cached';
                }} else {{
                    statusDot.className = 'status-dot stale';
                    statusText.textContent = 'Error: ' + response.status;
                }}
            }} catch (e) {{
                statusDot.className = 'status-dot stale';
                statusText.textContent = 'Disconnected';
            }}
            
            loadingOverlay.style.display = 'none';
            captureBtn.disabled = false;
            isLoading = false;
        }}
        
        function toggleOverlay() {{
            overlay = !overlay;
            const url = new URL(window.location);
            url.searchParams.set('overlay', overlay);
            window.location.href = url.toString();
        }}
        
        function updateAutoRefresh() {{
            const rate = document.getElementById('autoRefresh').value;
            const url = new URL(window.location);
            url.searchParams.set('refresh', rate);
            window.location.href = url.toString();
        }}
        
        // Auto-refresh if configured
        {auto_refresh_js}
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)
