"""
Counts Routes - Real-time count visibility with three-tier display.

Provides:
- GET /api/counts      - JSON endpoint with confirmed/pending/just_classified data
- GET /api/counts/stream - SSE endpoint for real-time updates
- GET /api/bag-types   - Bag type metadata (name, thumb path) for UI images
- GET /counts          - HTML dashboard with live pipeline visualization
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Any, List

from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, StreamingResponse

from src.config.paths import KNOWN_CLASSES_DIR

from src.endpoint.pipeline_state import read_state
from src.endpoint.shared import get_db, get_templates
from src.utils.AppLogging import logger

router = APIRouter(tags=["counts"])

# 2-hour idle threshold in seconds
_IDLE_RESET_SECONDS = 2 * 60 * 60
# Work "active" threshold – last bag within this window means the line is still running
_WORK_ACTIVE_THRESHOLD = 5 * 60  # 5 minutes


def _apply_idle_reset_and_work_info(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Post-process pipeline state for the counts API:

    1. **2-hour idle reset** – If the last counted bag was more than 2 hours
       ago, zero out confirmed / pending / just_classified counts in the
       response so the UI shows a clean slate.  The underlying pipeline state
       file is NOT modified (cosmetic reset only).

    2. **Work-started info** – Adds ``work_started_ago`` (human-readable
       "Xh Ym") and ``work_started_display`` (HH:MM) so the UI can show
       when the current work session began.  When the line has stopped (last
       bag > 5 min ago), ``work_started_ago`` reflects the *actual* work
       duration (start → last bag), not the ever-growing time-since-start.

    3. **Enhanced idle display** – Formats idle time in HH:MM format and
       exposes ``last_count_datetime_formatted`` (the datetime of the last
       counted bag) so the UI can show precisely when the line went quiet.
    """
    now = time.time()

    last_count_ts = state.get("last_count_timestamp", 0) or 0
    work_started_ts = state.get("work_started_timestamp", 0) or 0

    # ── 2-hour idle reset ──
    idle_seconds = (now - last_count_ts) if last_count_ts > 0 else 0
    is_idle_reset = last_count_ts > 0 and idle_seconds >= _IDLE_RESET_SECONDS

    if is_idle_reset:
        state["confirmed"] = {}
        state["pending"] = {}
        state["just_classified"] = {}
        state["confirmed_total"] = 0
        state["pending_total"] = 0
        state["just_classified_total"] = 0
        state["idle_reset"] = True
        state["idle_minutes"] = round(idle_seconds / 60)

        # Format idle time in HH:MM format
        idle_hours = int(state["idle_minutes"] // 60)
        idle_mins = int(state["idle_minutes"] % 60)
        state["idle_time_formatted"] = f"{idle_hours:02d}:{idle_mins:02d}"

        # Clear batch/SM display fields so the batch timeline and state-machine
        # panel reset on screen (counts were zeroed above but type names were not,
        # causing the previous batch name to persist until the next production run).
        state["current_batch_type"] = None
        state["previous_batch_type"] = None
        state["last_classified_type"] = None
        if "state_machine" in state and isinstance(state["state_machine"], dict):
            state["state_machine"] = {
                **state["state_machine"],
                "confirmed_batch_class": None,
                "current_run_class": None,
                "current_run_length": 0,
                "last_decision": None,
            }
    else:
        state["idle_reset"] = False
        state["idle_minutes"] = round(idle_seconds / 60) if last_count_ts > 0 else 0
        state["idle_time_formatted"] = None

    # ── Last-bag datetime (always available when at least one bag has been counted) ──
    if last_count_ts > 0:
        state["last_count_datetime_formatted"] = datetime.fromtimestamp(last_count_ts).strftime("%Y/%m/%d - %H:%M")
    else:
        state["last_count_datetime_formatted"] = None

    # ── Work-started info ──
    # Determine whether the line is still actively running (last bag < 5 min ago).
    # When stopped, show the *actual* work duration (start → last bag) instead of
    # the ever-growing (start → now), which was misleading after the line had stopped.
    work_is_active = last_count_ts > 0 and idle_seconds < _WORK_ACTIVE_THRESHOLD

    if work_started_ts > 0 and not is_idle_reset:
        if work_is_active:
            # Line running: show live elapsed time from session start to NOW
            elapsed = now - work_started_ts
        else:
            # Line stopped: show fixed duration = session start → last bag counted
            elapsed = max(0.0, last_count_ts - work_started_ts) if last_count_ts > 0 else (now - work_started_ts)

        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        state["work_started_ago"] = f" {hours} ساعة {minutes} دقيقة " if hours > 0 else f" {minutes} دقيقة"
        state["work_started_display"] = datetime.fromtimestamp(work_started_ts).strftime("%H:%M")
        state["work_started_datetime_formatted"] = datetime.fromtimestamp(work_started_ts).strftime("%Y/%m/%d - %H:%M")
        state["work_is_active"] = work_is_active
    else:
        state["work_started_ago"] = None
        state["work_started_display"] = None
        state["work_started_datetime_formatted"] = None
        state["work_is_active"] = False

    return state


@router.get("/api/counts")
async def api_counts() -> Dict[str, Any]:
    """
    Enhanced analytics endpoint with three-tier count data.

    Returns granular pipeline state:
    - confirmed: Final, persisted counts (after smoothing)
    - pending: Items in the smoothing window awaiting batch validation
    - just_classified: Real-time classification results (before smoothing)
    - smoothing_rate: Fraction of items that were corrected by smoothing
    - window_status: Smoothing window fill state
    - recent_events: Last 10 pipeline events for live feed
    - idle_reset: True if counts were zeroed due to 2-hour inactivity
    - work_started_ago: Human-readable elapsed time since work started
    - work_started_display: HH:MM when work started
    """
    state = read_state()

    # Remove internal fields
    result = {k: v for k, v in state.items() if not k.startswith("_")}

    # Apply idle reset and work-started calculations
    result = _apply_idle_reset_and_work_info(result)

    return result


@router.get("/api/bag-types")
async def api_bag_types() -> List[Dict[str, Any]]:
    """
    Bag type metadata endpoint for UI image rendering.

    Returns list of bag types with normalized web paths for thumbnails.
    Used by the counts dashboard to show classification images.
    """
    db = get_db()
    bag_types = await run_in_threadpool(db.get_all_bag_types)

    # Normalize thumb paths: data/classes/X → known_classes/X
    for bt in bag_types:
        thumb = bt.get("thumb", "") or ""
        bt["thumb"] = thumb.replace(KNOWN_CLASSES_DIR + "/", "known_classes/")

    return bag_types


@router.get("/api/counts/stream")
async def api_counts_stream(request: Request) -> StreamingResponse:
    """
    Server-Sent Events (SSE) endpoint for real-time count updates.

    Pushes state updates at ~1 second intervals when data changes.
    Stops when the client disconnects so the server thread is freed.

    Clients connect via EventSource:
        const es = new EventSource('/api/counts/stream');
        es.onmessage = (e) => { const data = JSON.parse(e.data); ... };
    """
    return StreamingResponse(
        _sse_generator(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _sse_generator(request: Request):
    """Generate SSE events by polling pipeline state file.

    Stops when the client disconnects (request.is_disconnected()) to avoid
    leaking background tasks that block other server requests.
    """
    last_updated_at = 0.0

    while True:
        # Stop generating when the client has disconnected
        if await request.is_disconnected():
            logger.debug("[SSE] Client disconnected, stopping stream")
            return

        try:
            state = read_state()
            current_updated = state.get("_updated_at", 0)

            # Only send when state has changed
            if current_updated > last_updated_at:
                last_updated_at = current_updated
                payload = {k: v for k, v in state.items() if not k.startswith("_")}
                payload = _apply_idle_reset_and_work_info(payload)
                yield f"data: {json.dumps(payload)}\n\n"
            else:
                # Send keepalive comment to prevent connection timeout
                yield ": keepalive\n\n"

        except Exception as e:
            logger.debug(f"[SSE] Error reading state: {e}")
            yield ": error\n\n"

        await asyncio.sleep(1.0)


@router.get("/counts", response_class=HTMLResponse)
async def counts_page(request: Request) -> HTMLResponse:
    """
    HTML dashboard for real-time pipeline count visibility.

    Shows three-tier counts with SSE-powered live updates:
    - Confirmed counts (after smoothing) ✓
    - Pending counts (in smoothing window) ⏳
    - Just classified (tentative) 🔄
    - Visual pipeline progress indicator
    """
    templates = get_templates()
    return templates.TemplateResponse("counts.html", {"request": request})
