"""
Counts Routes - Real-time count visibility with three-tier display.

Provides:
- GET  /api/counts           - JSON endpoint with confirmed/pending/just_classified data
- WS   /ws/counts/stream     - WebSocket endpoint for real-time updates
- GET  /api/bag-types        - Bag type metadata (name, thumb path) for UI images
- GET  /counts               - HTML dashboard with live pipeline visualization
"""

import asyncio
from typing import Dict, Any, List

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse

from src.endpoint.pipeline_state import read_state
from src.endpoint.shared import get_db, get_templates
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
    - recent_events: Last 10 pipeline events for live feed
    """
    state = read_state()

    # Remove internal fields
    result = {k: v for k, v in state.items() if not k.startswith("_")}
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
        bt["thumb"] = thumb.replace("data/classes/", "known_classes/")

    return bag_types


@router.websocket("/ws/counts/stream")
async def ws_counts_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time count updates.

    Pushes JSON state updates at ~1-second intervals when data changes.
    Sends a lightweight ``{"type":"ping"}`` keep-alive every idle cycle so
    proxies and firewalls do not drop the connection.

    Clients connect via:
        const ws = new WebSocket(`ws://${location.host}/ws/counts/stream`);
        ws.onmessage = (e) => { const data = JSON.parse(e.data); ... };
    """
    await websocket.accept()
    logger.debug("[WS] Client connected to /ws/counts/stream")

    last_updated_at = 0.0

    try:
        while True:
            try:
                state = read_state()
                current_updated = state.get("_updated_at", 0)

                if current_updated > last_updated_at:
                    last_updated_at = current_updated
                    payload = {k: v for k, v in state.items() if not k.startswith("_")}
                    await websocket.send_json(payload)
                else:
                    # Keep-alive so proxies don't time out the idle connection
                    await websocket.send_json({"type": "ping"})

            except WebSocketDisconnect:
                raise  # re-raise so the outer handler logs it cleanly
            except Exception as exc:
                logger.debug(f"[WS] Error reading state: {exc}")
                try:
                    await websocket.send_json({"type": "error", "detail": str(exc)})
                except Exception:
                    break  # connection is dead

            await asyncio.sleep(1.0)

    except WebSocketDisconnect:
        logger.debug("[WS] Client disconnected from /ws/counts/stream")
    except Exception as exc:
        logger.debug(f"[WS] Unexpected error: {exc}")
    finally:
        # Graceful close – ignore if already closed by the client
        try:
            await websocket.close()
        except Exception:
            pass


@router.get("/counts", response_class=HTMLResponse)
async def counts_page(request: Request) -> HTMLResponse:
    """
    HTML dashboard for real-time pipeline count visibility.

    Shows three-tier counts with WebSocket-powered live updates:
    - Confirmed counts (after smoothing) ✓
    - Pending counts (in smoothing window) ⏳
    - Just classified (tentative) 🔄
    - Visual pipeline progress indicator
    """
    templates = get_templates()
    return templates.TemplateResponse("counts.html", {"request": request})
