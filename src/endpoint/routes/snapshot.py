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


@router.get("/snapshot/background")
async def get_background_frame() -> Response:
    """
    Get the static background frame for track visualization.

    Returns the background_frame.jpg from the snapshot directory.
    This is a fixed reference image showing the camera view without any bags.
    """
    background_path = os.path.join(SNAPSHOT_DIR, "background_frame.jpg")

    if not os.path.exists(background_path):
        # Fallback to latest_raw.jpg if background_frame doesn't exist
        background_path = SNAPSHOT_RAW_PATH

    if not os.path.exists(background_path):
        return Response(
            content=b"",
            status_code=404,
            media_type="text/plain"
        )

    try:
        with open(background_path, "rb") as f:
            image_data = f.read()

        return Response(
            content=image_data,
            media_type="image/jpeg",
            headers={
                "Cache-Control": "public, max-age=3600",  # Cache for 1 hour
            }
        )
    except Exception as e:
        logger.error(f"[Snapshot] Failed to read background frame: {e}")
        return Response(
            content=b"",
            status_code=500,
            media_type="text/plain"
        )


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
<html lang="ar">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>البث المباشر — منظومة إحصاء أكياس الخبز</title>
    <link href="https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;800&family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        :root {{
            --bg-deep: #0f172a;
            --glass-bg: rgba(30, 41, 59, 0.7);
            --glass-border: rgba(255, 255, 255, 0.08);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --accent-primary: #38bdf8;
            --accent-success: #2dd4bf;
            --accent-warning: #fbbf24;
            --accent-danger: #f87171;
            --accent-purple: #a78bfa;
            --radius-lg: 16px;
            --radius-md: 12px;
            --radius-sm: 8px;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'Tajawal', 'Inter', system-ui, sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
            color: var(--text-primary);
            min-height: 100vh;
            line-height: 1.6;
        }}
        /* Force English numbers */
        .num, #frameNum, #frameAge, #loadTime {{
            font-family: 'Inter', sans-serif !important;
            font-feature-settings: "tnum" 1;
        }}
        main {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem 1.5rem;
        }}
        .back-home {{
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            color: var(--accent-primary);
            text-decoration: none;
            font-size: 0.9rem;
            margin-bottom: 1.5rem;
            transition: color 0.2s;
        }}
        .back-home:hover {{ color: var(--accent-success); }}
        .page-header {{
            text-align: center;
            margin-bottom: 2rem;
        }}
        .page-header h1 {{
            font-size: 2rem;
            font-weight: 800;
            background: linear-gradient(135deg, var(--accent-primary), var(--accent-purple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.5rem;
        }}
        .page-header .subtitle {{
            color: var(--text-secondary);
            font-size: 1rem;
        }}
        .info-bar {{
            display: flex;
            gap: 1.5rem;
            margin-bottom: 1.5rem;
            padding: 1rem 1.5rem;
            background: var(--glass-bg);
            border: 1px solid var(--glass-border);
            border-radius: var(--radius-md);
            flex-wrap: wrap;
            backdrop-filter: blur(10px);
        }}
        .info-item {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        .info-label {{
            color: var(--text-muted);
            font-size: 0.85rem;
        }}
        .info-value {{
            color: var(--accent-primary);
            font-weight: 600;
            font-size: 0.9rem;
        }}
        .status-dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: var(--text-muted);
            transition: background 0.3s;
        }}
        .status-dot.live {{ background: var(--accent-success); box-shadow: 0 0 10px var(--accent-success); }}
        .status-dot.stale {{ background: var(--accent-danger); }}
        .status-dot.loading {{ background: var(--accent-warning); animation: pulse 0.5s infinite; }}
        @keyframes pulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.5; }} }}
        .frame-container {{
            position: relative;
            background: var(--glass-bg);
            border: 1px solid var(--glass-border);
            border-radius: var(--radius-lg);
            overflow: hidden;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            min-height: 400px;
        }}
        #snapshot {{
            display: block;
            width: 100%;
            height: auto;
            border-radius: var(--radius-lg);
        }}
        .loading-overlay {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: rgba(15, 23, 42, 0.9);
            padding: 1.5rem 2.5rem;
            border-radius: var(--radius-md);
            display: none;
            color: var(--accent-primary);
            font-weight: 600;
        }}
        .controls {{
            margin-top: 1.5rem;
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
        }}
        .btn {{
            padding: 0.75rem 1.5rem;
            border: none;
            border-radius: var(--radius-sm);
            cursor: pointer;
            font-size: 0.9rem;
            font-weight: 600;
            font-family: 'Tajawal', sans-serif;
            transition: all 0.2s;
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
        }}
        .btn:hover {{ transform: translateY(-2px); }}
        .btn:disabled {{ opacity: 0.5; cursor: not-allowed; transform: none; }}
        .btn-primary {{
            background: linear-gradient(135deg, var(--accent-primary), var(--accent-purple));
            color: white;
        }}
        .btn-primary:hover {{ box-shadow: 0 8px 25px rgba(56, 189, 248, 0.3); }}
        .btn-secondary {{
            background: var(--glass-bg);
            border: 1px solid var(--glass-border);
            color: var(--text-primary);
        }}
        .btn-secondary:hover {{ border-color: var(--accent-primary); background: rgba(56, 189, 248, 0.1); }}
        select {{
            padding: 0.75rem 1rem;
            border-radius: var(--radius-sm);
            border: 1px solid var(--glass-border);
            background: var(--glass-bg);
            color: var(--text-primary);
            font-size: 0.9rem;
            font-family: 'Tajawal', sans-serif;
            cursor: pointer;
        }}
        select:focus {{
            outline: none;
            border-color: var(--accent-primary);
        }}
        .note {{
            margin-top: 1.5rem;
            padding: 1rem 1.5rem;
            background: var(--glass-bg);
            border: 1px solid var(--glass-border);
            border-radius: var(--radius-md);
            border-right: 4px solid var(--accent-primary);
            font-size: 0.9rem;
            color: var(--text-secondary);
        }}
        .note strong {{ color: var(--accent-primary); }}
        .page-footer {{
            text-align: center;
            padding: 2rem;
            color: var(--text-muted);
            font-size: 0.85rem;
            margin-top: 2rem;
        }}
        .page-footer a {{
            color: var(--accent-primary);
            text-decoration: none;
        }}
        .page-footer a:hover {{ color: var(--accent-success); }}
    </style>
</head>
<body>
    <main>
        <a href="/" class="back-home"><i class="fa-solid fa-home"></i> الرئيسية</a>
        
        <div class="page-header">
            <h1><i class="fa-solid fa-camera"></i> البث المباشر</h1>
            <p class="subtitle">عرض حي للكاميرا مع إطارات الكشف والتعرف</p>
        </div>
        
        <div class="info-bar">
            <div class="info-item">
                <div class="status-dot" id="statusDot"></div>
                <span class="info-label">الحالة:</span>
                <span class="info-value" id="status">جاهز</span>
            </div>
            <div class="info-item">
                <span class="info-label">رقم الإطار:</span>
                <span class="info-value" id="frameNum">-</span>
            </div>
            <div class="info-item">
                <span class="info-label">العمر:</span>
                <span class="info-value" id="frameAge">-</span>
            </div>
            <div class="info-item">
                <span class="info-label">وقت التحميل:</span>
                <span class="info-value" id="loadTime">-</span>
            </div>
        </div>
        
        <div class="frame-container">
            <img id="snapshot" src="" alt="اضغط على 'التقاط صورة' للحصول على الصورة">
            <div class="loading-overlay" id="loadingOverlay">
                <i class="fa-solid fa-spinner fa-spin"></i> جارٍ الالتقاط...
            </div>
        </div>
        
        <div class="controls">
            <button class="btn btn-primary" id="captureBtn" onclick="refreshNow()">
                <i class="fa-solid fa-camera"></i> التقاط صورة
            </button>
            <button class="btn btn-secondary" onclick="toggleOverlay()">
                <i class="fa-solid fa-eye"></i> {"إخفاء التوضيحات" if overlay else "إظهار التوضيحات"}
            </button>
            <select id="autoRefresh" onchange="updateAutoRefresh()">
                <option value="0" {"selected" if refresh == 0 else ""}>يدوي</option>
                <option value="1" {"selected" if refresh == 1 else ""}>تلقائي 1 ثانية</option>
                <option value="2" {"selected" if refresh == 2 else ""}>تلقائي 2 ثانية</option>
                <option value="5" {"selected" if refresh == 5 else ""}>تلقائي 5 ثوانٍ</option>
            </select>
        </div>
        
        <div class="note">
            <strong><i class="fa-solid fa-info-circle"></i> وضع الالتقاط عند الطلب:</strong>
            يتم التقاط الصور فقط عند الضغط على الزر أو تفعيل التحديث التلقائي، مما يقلل الحمل على النظام.
        </div>
        
        <div class="page-footer">
            <a href="/">الرئيسية</a> • <a href="/counts">الإحصاء اللحظي</a> • <a href="/analytics">لوحة المؤشرات</a>
        </div>
    </main>
    
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
        
        function formatAge(ms) {{
            const sec = Math.floor(ms / 1000);
            if (sec < 60) return sec + ' ثانية';
            const min = Math.floor(sec / 60);
            if (min < 60) return min + ' دقيقة';
            const hr = Math.floor(min / 60);
            return hr + ' ساعة';
        }}
        
        async function refreshNow() {{
            if (isLoading) return;
            
            isLoading = true;
            captureBtn.disabled = true;
            statusDot.className = 'status-dot loading';
            statusText.textContent = 'جارٍ الالتقاط...';
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
                    loadTimeText.textContent = loadTime < 1000 ? loadTime + 'ms' : (loadTime / 1000).toFixed(1) + 's';
                    
                    const frameNum = response.headers.get('X-Frame-Number') || '-';
                    const ageMs = response.headers.get('X-Frame-Age-Ms') || '0';
                    const isFresh = response.headers.get('X-Frame-Fresh') === 'true';
                    
                    frameNumText.textContent = '#' + frameNum;
                    frameAgeText.textContent = formatAge(parseFloat(ageMs));
                    
                    statusDot.className = isFresh ? 'status-dot live' : 'status-dot stale';
                    statusText.textContent = isFresh ? 'مباشر' : 'مخزن مؤقتاً';
                }} else {{
                    statusDot.className = 'status-dot stale';
                    statusText.textContent = 'خطأ: ' + response.status;
                }}
            }} catch (e) {{
                statusDot.className = 'status-dot stale';
                statusText.textContent = 'غير متصل';
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
