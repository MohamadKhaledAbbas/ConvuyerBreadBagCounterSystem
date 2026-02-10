"""Track Lifecycle Routes - Track event lifecycle analytics endpoint."""

from typing import Optional

from fastapi import APIRouter, Request, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse

from src.endpoint.repositories.track_lifecycle_repository import TrackLifecycleRepository
from src.endpoint.services.track_lifecycle_service import TrackLifecycleService
from src.endpoint.shared import get_db, get_templates
from src.utils.AppLogging import logger

router = APIRouter()


def _get_service() -> TrackLifecycleService:
    db = get_db()
    repo = TrackLifecycleRepository(db)
    return TrackLifecycleService(repo)


@router.get("/track-events", response_class=HTMLResponse)
async def track_events_page(
    request: Request,
    start_time: Optional[str] = Query(None),
    end_time: Optional[str] = Query(None),
    event_type: Optional[str] = Query(None)
):
    """
    Track event lifecycle analytics page.

    Shows all track events with their full lifecycle details:
    - When each track was created and at what position
    - ROI collections with bbox coordinates
    - Per-ROI classification results
    - Final voting result
    - Track completion/lost/invalid status

    Query params:
        start_time: ISO datetime start (optional, defaults to last 24h)
        end_time: ISO datetime end (optional, defaults to now)
        event_type: Filter by type: track_completed, track_lost, track_invalid
    """
    templates = get_templates()
    service = _get_service()

    try:
        if start_time and end_time:
            start_dt = service.parse_datetime(start_time)
            end_dt = service.parse_datetime(end_time)
            if start_dt >= end_dt:
                raise HTTPException(422, 'Start time must be before end time')
        else:
            start_dt, end_dt = service.get_default_time_range()

        data = service.get_lifecycle_data(start_dt, end_dt, event_type=event_type)

        context = {
            'request': request,
            'meta': data['meta'],
            'stats': data['stats'],
            'events': data['events']
        }

        logger.info(f"[TrackEvents] Rendering {len(data['events'])} events")
        return templates.TemplateResponse('track_events.html', context)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'[TrackEvents] Error: {e}', exc_info=True)
        raise HTTPException(500, str(e))


@router.get("/track-events/{track_id}", response_class=JSONResponse)
async def track_lifecycle_detail(track_id: int):
    """
    Get full lifecycle for a single track as JSON.

    Returns summary + all detail steps for debugging.
    """
    service = _get_service()
    try:
        data = service.repo.get_track_lifecycle(track_id)
        if data['summary'] is None:
            raise HTTPException(404, f"Track {track_id} not found")
        return data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'[TrackEvents] Detail error: {e}', exc_info=True)
        raise HTTPException(500, str(e))
