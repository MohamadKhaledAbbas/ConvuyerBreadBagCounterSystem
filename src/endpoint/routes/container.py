"""
Container tracking routes for FastAPI.

Provides endpoints for:
- Container dashboard page
- Container events API
- Container statistics API
- Container snapshot viewer
- Real-time SSE updates

Route prefix: /container
"""

import json
import os
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, RedirectResponse, Response

from src.endpoint.shared import get_db, get_templates, render_template
from src.endpoint.repositories.container_repository import ContainerRepository
from src.logging.Database import DatabaseManager
from src.utils.AppLogging import logger
from src.config.paths import (
    DATA_DIR,
    SNAPSHOT_DIR,
    CONTAINER_PIPELINE_STATE_FILE as CONTAINER_STATE_FILE,
    CONTAINER_SNAPSHOT_DIR,
    CONTAINER_CONTENT_VIDEOS_DIR,
)


router = APIRouter(prefix="/container", tags=["container"])


def get_container_repo(db: DatabaseManager = Depends(get_db)) -> ContainerRepository:
    """Dependency to get container repository."""
    return ContainerRepository(db)


def _event_id_from_path(path: Optional[str]) -> Optional[str]:
    """Extract the event id from a stored relative path."""
    if not path:
        return None

    parts = [part for part in str(path).replace("\\", "/").split("/") if part]
    if not parts:
        return None

    last = parts[-1]
    if last == "video.mp4" and len(parts) >= 2:
        return parts[-2]
    if last.endswith(".mp4") and len(last) > 4:
        return last[:-4]
    return last


def _attach_event_video_info(event: dict) -> dict:
    """Annotate an event payload with actual clip availability."""
    metadata = event.get("metadata") or {}
    event_id = (
        _event_id_from_path(metadata.get("video_relpath"))
        or _event_id_from_path(event.get("snapshot_path"))
    )
    video_path = _resolve_event_video_path(event_id) if event_id else None

    enriched = dict(event)
    enriched["video_event_id"] = event_id
    enriched["has_video"] = bool(video_path)
    enriched["video_url"] = (
        f"/container/event/{event_id}/video" if video_path and event_id else None
    )
    return enriched


# =============================================================================
# HTML Pages
# =============================================================================

@router.get("", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
async def container_dashboard(
    request: Request,
    templates=Depends(get_templates),
):
    """
    Container monitoring dashboard page (صالة).
    
    Displays:
    - Real-time positive/negative counts
    - Per-QR breakdown (containers 1-5)
    - Direction mismatch alerts
    - Recent container events
    - Timeline chart
    """
    return render_template(
        templates,
        request,
        "container.html",
        {
            "title": "صالة - مراقبة الحاويات",
            "page": "container",
        }
    )


@router.get("/events", response_class=HTMLResponse)
async def container_events_page(
    request: Request,
    templates=Depends(get_templates),
):
    """Container events list page."""
    return render_template(
        templates,
        request,
        "container_events.html",
        {
            "title": "سجل الحاويات",
            "page": "container_events",
        }
    )


# =============================================================================
# API Endpoints
# =============================================================================

@router.get("/api/stats")
async def get_container_stats(
    start_time: Optional[str] = Query(None, description="ISO 8601 start time"),
    end_time: Optional[str] = Query(None, description="ISO 8601 end time"),
    direction: Optional[str] = Query(None),
    qr_code_value: Optional[int] = Query(None, ge=1, le=5),
    is_lost: Optional[bool] = Query(None),
    repo: ContainerRepository = Depends(get_container_repo),
):
    """
    Get aggregated container statistics.
    
    Returns counts and breakdowns for the specified time range.
    If no time range specified, returns today's stats.
    """
    # Default to today
    if not start_time:
        start_time = datetime.now().replace(
            hour=0, minute=0, second=0, microsecond=0
        ).isoformat()
    
    if not end_time:
        end_time = datetime.now().isoformat()
    
    stats = repo.get_aggregated_stats(
        start_time=start_time,
        end_time=end_time,
        direction=direction,
        qr_code_value=qr_code_value,
        is_lost=is_lost,
    )
    
    return JSONResponse(content=stats)


@router.get("/api/events")
async def get_container_events(
    start_time: Optional[str] = Query(None),
    end_time: Optional[str] = Query(None),
    direction: Optional[str] = Query(None),
    qr_code_value: Optional[int] = Query(None, ge=1, le=5),
    is_lost: Optional[bool] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    repo: ContainerRepository = Depends(get_container_repo),
):
    """
    Get container events with filtering and pagination.
    
    Query parameters:
    - start_time: ISO 8601 start time filter
    - end_time: ISO 8601 end time filter
    - direction: Filter by direction (positive/negative/unknown)
    - qr_code_value: Filter by container QR code (1-5)
    - is_lost: Filter by lost status (true/false)
    - limit: Max events to return (default 100)
    - offset: Pagination offset
    """
    events = repo.get_events(
        start_time=start_time,
        end_time=end_time,
        direction=direction,
        qr_code_value=qr_code_value,
        is_lost=is_lost,
        limit=limit,
        offset=offset,
    )
    events = [_attach_event_video_info(event) for event in events]
    
    total = repo.get_event_count(
        start_time=start_time,
        end_time=end_time,
        direction=direction,
        qr_code_value=qr_code_value,
        is_lost=is_lost,
    )
    
    return JSONResponse(content={
        "events": events,
        "total": total,
        "limit": limit,
        "offset": offset,
    })


@router.get("/api/events/{event_id}")
async def get_container_event(
    event_id: int,
    repo: ContainerRepository = Depends(get_container_repo),
):
    """Get a single container event by ID."""
    event = repo.get_event_by_id(event_id)
    
    if not event:
        return JSONResponse(
            status_code=404,
            content={"error": "Event not found"}
        )
    
    return JSONResponse(content=_attach_event_video_info(event))


@router.get("/api/hourly")
async def get_hourly_stats(
    start_time: Optional[str] = Query(None),
    end_time: Optional[str] = Query(None),
    repo: ContainerRepository = Depends(get_container_repo),
):
    """
    Get hourly statistics for timeline charts.
    
    Default: last 24 hours.
    """
    if not end_time:
        end_time = datetime.now().isoformat()
    
    if not start_time:
        start_time = (datetime.now() - timedelta(hours=24)).isoformat()
    
    hourly = repo.get_hourly_stats(start_time, end_time)
    
    return JSONResponse(content={
        "hourly": hourly,
        "start_time": start_time,
        "end_time": end_time,
    })


@router.get("/api/alerts")
async def get_mismatch_alerts(
    threshold: int = Query(5, ge=1, le=100),
    window_hours: float = Query(1.0, ge=0.1, le=24.0),
    repo: ContainerRepository = Depends(get_container_repo),
):
    """
    Check for direction mismatch alerts.
    
    Returns alerts when positive and negative counts differ
    significantly within the specified time window.
    
    Query parameters:
    - threshold: Minimum difference to trigger alert (default 5)
    - window_hours: Time window in hours to check (default 1.0)
    """
    alerts = repo.get_mismatch_alerts(
        threshold=threshold,
        window_hours=window_hours,
    )
    
    return JSONResponse(content={
        "alerts": alerts,
        "has_alerts": len(alerts) > 0,
        "threshold": threshold,
        "window_hours": window_hours,
    })


@router.get("/api/state")
async def get_pipeline_state():
    """
    Get current container pipeline state.
    
    Reads from the state file published by ContainerCounterApp.
    Used for real-time dashboard updates.
    """
    try:
        if os.path.exists(CONTAINER_STATE_FILE):
            with open(CONTAINER_STATE_FILE, 'r') as f:
                state = json.load(f)
            
            # Check staleness
            state_time = datetime.fromisoformat(state.get('timestamp', ''))
            age_seconds = (datetime.now() - state_time).total_seconds()
            state['stale'] = age_seconds > 60
            state['age_seconds'] = age_seconds
            
            return JSONResponse(content=state)
        else:
            return JSONResponse(content={
                "error": "State file not found",
                "stale": True,
                "state": None,
            })
    except Exception as e:
        logger.error(f"[ContainerRoutes] Failed to read state: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


# =============================================================================
# Snapshot Endpoints
# =============================================================================

@router.get("/snapshot/view")
async def container_snapshot_view():
    """Redirect to shared snapshot viewer with container camera selected."""
    return RedirectResponse(url="/snapshot/view?camera=container")


@router.get("/snapshot/latest")
async def get_container_snapshot():
    """
    Get the latest snapshot from container camera.
    
    Returns the most recent frame as JPEG.
    """
    snapshot_path = os.path.join(SNAPSHOT_DIR, "container_latest_raw.jpg")
    
    if os.path.exists(snapshot_path):
        return FileResponse(
            snapshot_path,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
        )
    else:
        return JSONResponse(
            status_code=404,
            content={"error": "No snapshot available"}
        )


@router.get("/snapshot/event/{event_id}")
async def get_event_snapshots(
    event_id: str,
    frame_type: str = Query("frames", pattern="^(pre|post|frames)$"),
    frame_index: int = Query(0, ge=0),
):
    """
    Get a specific frame from an event's snapshot capture.
    
    Path parameters:
    - event_id: The event capture ID (directory name)
    
    Query parameters:
    - frame_type: "frames" (track-lifetime), or legacy "pre"/"post"
    - frame_index: Frame index (0-based)
    """
    frame_path = os.path.join(
        CONTAINER_SNAPSHOT_DIR,
        event_id,
        frame_type,
        f"frame_{frame_index:04d}.jpg"
    )
    
    if os.path.exists(frame_path):
        return FileResponse(frame_path, media_type="image/jpeg")
    else:
        return JSONResponse(
            status_code=404,
            content={"error": "Frame not found"}
        )


@router.get("/snapshot/event/{event_id}/metadata")
async def get_event_metadata(event_id: str):
    """Get metadata for an event's snapshot capture."""
    metadata_path = os.path.join(CONTAINER_SNAPSHOT_DIR, event_id, "metadata.json")
    
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return JSONResponse(content=metadata)
    else:
        return JSONResponse(
            status_code=404,
            content={"error": "Metadata not found"}
        )


@router.get("/snapshot/event/{event_id}/video")
async def get_event_video(event_id: str):
    """Legacy alias — see :func:`get_event_clip`.

    Kept so older clients keep working; prefer ``/container/event/{id}/video``.
    """
    return await get_event_clip(event_id)


# =============================================================================
# Unified event-video endpoint (new)
# =============================================================================

# event_id format: qr{N}_{direction}_{YYYYMMDD_HHMMSS_ffffff}
_EVENT_ID_RE = __import__('re').compile(r'^[A-Za-z0-9_\-]+$')


def _resolve_event_video_path(event_id: str) -> Optional[str]:
    """Return the absolute path of the MP4 for ``event_id`` if it exists.

    Tries the QR-camera location first (``container_snapshots/{id}/video.mp4``),
    then the content-camera location (``container_content_videos/{id}.mp4``).
    Returns ``None`` when neither exists.  Also verifies the resolved path
    stays inside the expected roots (defence in depth against traversal).
    """
    if not _EVENT_ID_RE.match(event_id):
        return None

    candidates = [
        (CONTAINER_SNAPSHOT_DIR, os.path.join(CONTAINER_SNAPSHOT_DIR, event_id, "video.mp4")),
        (CONTAINER_CONTENT_VIDEOS_DIR, os.path.join(CONTAINER_CONTENT_VIDEOS_DIR, f"{event_id}.mp4")),
    ]
    for root, candidate in candidates:
        try:
            abs_root = os.path.realpath(root)
            abs_file = os.path.realpath(candidate)
        except OSError:
            continue
        if not abs_file.startswith(abs_root + os.sep):
            continue
        if os.path.isfile(abs_file):
            return abs_file
    return None


@router.get("/event/{event_id}/video")
async def get_event_clip(event_id: str, request: Request):
    """Stream the event MP4 regardless of which camera produced it.

    Resolves the clip by checking both the QR-camera and content-camera
    output directories, returning the first match.  Supports HTTP Range
    requests so ``<video>`` tags can seek without full downloads.
    """
    path = _resolve_event_video_path(event_id)
    if path is None:
        return JSONResponse(status_code=404, content={"error": "Video not found"})

    file_size = os.path.getsize(path)
    range_header = request.headers.get("range")

    if range_header:
        # Parse "bytes=START-END"
        range_spec = range_header.strip().lower()
        if range_spec.startswith("bytes="):
            range_spec = range_spec[6:]
        parts = range_spec.split("-", 1)
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if len(parts) > 1 and parts[1] else file_size - 1
        end = min(end, file_size - 1)
        length = end - start + 1

        with open(path, "rb") as f:
            f.seek(start)
            data = f.read(length)

        return Response(
            content=data,
            status_code=206,
            media_type="video/mp4",
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(length),
                "Cache-Control": "no-cache",
            },
        )

    return FileResponse(
        path,
        media_type="video/mp4",
        headers={
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size),
        },
    )



# =============================================================================
# Content Camera (3D-angle container contents recorder)
# =============================================================================

# Restrict event_id to alphanumerics, underscore, dash — prevents path traversal.
_EVENT_ID_SAFE = _EVENT_ID_RE


@router.get("/content", response_class=HTMLResponse)
async def container_content_page(
    request: Request,
    templates=Depends(get_templates),
):
    """Content camera playback page.

    Lists recent content-camera recordings with embedded video players.
    """
    return render_template(
        templates,
        request,
        "container_content.html",
        {
            "title": "محتويات الحاويات - تسجيلات",
            "page": "container_content",
        },
    )


@router.get("/api/content/list")
async def list_content_videos(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    qr_code_value: Optional[int] = Query(None, ge=1, le=5),
    repo: ContainerRepository = Depends(get_container_repo),
):
    """List content videos (newest first), joined with their container events.

    Each item: event_id, qr_value, direction, timestamp, size_bytes,
    duration_seconds (approximate), db event (if matched).
    """
    if not os.path.isdir(CONTAINER_CONTENT_VIDEOS_DIR):
        return JSONResponse(content={"videos": [], "total": 0})

    try:
        entries = []
        for name in os.listdir(CONTAINER_CONTENT_VIDEOS_DIR):
            if not name.endswith(".mp4"):
                continue
            event_id = name[:-4]
            if not _EVENT_ID_SAFE.match(event_id):
                continue
            # Filter by QR if requested (event_id format: qrN_dir_timestamp).
            if qr_code_value is not None:
                if not event_id.startswith(f"qr{qr_code_value}_"):
                    continue
            full = os.path.join(CONTAINER_CONTENT_VIDEOS_DIR, name)
            try:
                st = os.stat(full)
            except OSError:
                continue
            entries.append({
                "event_id": event_id,
                "filename": name,
                "size_bytes": st.st_size,
                "mtime": st.st_mtime,
            })
        entries.sort(key=lambda e: e["mtime"], reverse=True)
        total = len(entries)
        page = entries[offset:offset + limit]

        # Try to attach event metadata (qr_value, direction, timestamp) by
        # matching the naming convention qrN_dir_YYYYMMDD_HHMMSS_ffffff.
        for item in page:
            parts = item["event_id"].split("_")
            if len(parts) >= 2 and parts[0].startswith("qr"):
                try:
                    item["qr_value"] = int(parts[0][2:])
                except ValueError:
                    item["qr_value"] = None
                item["direction"] = parts[1]
            else:
                item["qr_value"] = None
                item["direction"] = None
            # Try DB lookup by snapshot_path basename.
            try:
                ev = repo.get_event_by_snapshot_event_id(item["event_id"]) \
                    if hasattr(repo, "get_event_by_snapshot_event_id") else None
            except Exception:
                ev = None
            if ev:
                item["db_event"] = ev

        return JSONResponse(content={"videos": page, "total": total})

    except Exception as e:
        logger.error(f"[ContainerRoutes] list_content_videos failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/content/video/{event_id}")
async def get_content_video(event_id: str):
    """Stream a content-camera MP4 for the given event id.

    FastAPI's FileResponse supports HTTP Range requests automatically,
    so the ``<video>`` element can seek without downloading the whole file.
    """
    if not _EVENT_ID_SAFE.match(event_id):
        return JSONResponse(status_code=400, content={"error": "Invalid event id"})

    video_path = os.path.join(CONTAINER_CONTENT_VIDEOS_DIR, f"{event_id}.mp4")
    # Resolve & ensure we didn't escape the videos dir.
    abs_root = os.path.realpath(CONTAINER_CONTENT_VIDEOS_DIR)
    abs_file = os.path.realpath(video_path)
    if not abs_file.startswith(abs_root + os.sep):
        return JSONResponse(status_code=400, content={"error": "Invalid path"})

    if not os.path.isfile(abs_file):
        return JSONResponse(status_code=404, content={"error": "Video not found"})

    return FileResponse(
        abs_file,
        media_type="video/mp4",
        headers={"Accept-Ranges": "bytes"},
    )
