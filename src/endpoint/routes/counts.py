"""
Counts Routes - Real-time count visibility with three-tier display.

Provides:
- GET /api/counts      - JSON endpoint with confirmed/pending/just_classified data
- GET /api/counts/stream - SSE endpoint for real-time updates
- GET /counts          - HTML dashboard with live pipeline visualization
"""

import asyncio
import json
import time
from typing import Dict, Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from src.endpoint.pipeline_state import read_state
from src.endpoint.shared import get_templates
from src.utils.AppLogging import logger

router = APIRouter(tags=["counts"])


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
    """
    state = read_state()

    # Remove internal fields
    result = {k: v for k, v in state.items() if not k.startswith("_")}
    return result


@router.get("/api/counts/stream")
async def api_counts_stream() -> StreamingResponse:
    """
    Server-Sent Events (SSE) endpoint for real-time count updates.

    Pushes state updates at ~1 second intervals when data changes.
    Clients connect via EventSource:
        const es = new EventSource('/api/counts/stream');
        es.onmessage = (e) => { const data = JSON.parse(e.data); ... };
    """
    return StreamingResponse(
        _sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _sse_generator():
    """Generate SSE events by polling pipeline state file."""
    last_updated_at = 0.0

    while True:
        try:
            state = read_state()
            current_updated = state.get("_updated_at", 0)

            # Only send when state has changed
            if current_updated > last_updated_at:
                last_updated_at = current_updated
                payload = {k: v for k, v in state.items() if not k.startswith("_")}
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
