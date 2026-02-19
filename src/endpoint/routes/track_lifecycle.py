"""Track Lifecycle Routes - Track event lifecycle analytics endpoint.

Enhanced with:
- Advanced filtering (classification, confidence, duration, entry/exit type)
- Pagination support
- JSON API endpoints
- Track animation data endpoint
"""

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
    event_type: Optional[str] = Query(None),
    classification: Optional[str] = Query(None),
    min_confidence: Optional[float] = Query(None, ge=0, le=1),
    min_duration: Optional[float] = Query(None, ge=0),
    max_duration: Optional[float] = Query(None, ge=0),
    entry_type: Optional[str] = Query(None),
    exit_direction: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=10, le=200)
):
    """
    Track event lifecycle analytics page with advanced filtering.

    Shows all track events with their full lifecycle details:
    - When each track was created and at what position
    - ROI collections with bbox coordinates
    - Per-ROI classification results
    - Final voting result
    - Track completion/lost/invalid status
    - Animated trajectory visualization

    Query params:
        start_time: ISO datetime start (optional, defaults to last 24h)
        end_time: ISO datetime end (optional, defaults to now)
        event_type: Filter by type: track_completed, track_lost, track_invalid
        classification: Filter by classification result (e.g., "Wheatberry")
        min_confidence: Minimum detection confidence (0.0-1.0)
        min_duration: Minimum track duration in seconds
        max_duration: Maximum track duration in seconds
        entry_type: Filter by entry type: bottom_entry, thrown_entry, midway_entry
        exit_direction: Filter by exit: top, bottom, left, right, timeout
        page: Page number (1-based)
        page_size: Events per page (10-200)
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

        data = service.get_lifecycle_data(
            start_dt, end_dt,
            event_type=event_type,
            classification=classification,
            min_confidence=min_confidence,
            min_duration=min_duration,
            max_duration=max_duration,
            entry_type=entry_type,
            exit_direction=exit_direction,
            page=page,
            page_size=page_size
        )

        context = {
            'request': request,
            'meta': data['meta'],
            'stats': data['stats'],
            'events': data['events'],
            'pagination': data['pagination'],
            'filter_options': data['filter_options']
        }

        logger.info(f"[TrackEvents] Rendering {len(data['events'])} events (page {page})")
        return templates.TemplateResponse('track_events_ar.html', context)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'[TrackEvents] Error: {e}', exc_info=True)
        raise HTTPException(500, str(e))


@router.get("/api/track-events", response_class=JSONResponse)
async def track_events_api(
    start_time: Optional[str] = Query(None),
    end_time: Optional[str] = Query(None),
    event_type: Optional[str] = Query(None),
    classification: Optional[str] = Query(None),
    min_confidence: Optional[float] = Query(None, ge=0, le=1),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=10, le=200)
):
    """
    JSON API endpoint for track events.

    Returns paginated track events with parsed JSON fields.
    Use this for programmatic access or custom frontends.
    """
    service = _get_service()

    try:
        if start_time and end_time:
            start_dt = service.parse_datetime(start_time)
            end_dt = service.parse_datetime(end_time)
            if start_dt >= end_dt:
                raise HTTPException(422, 'Start time must be before end time')
        else:
            start_dt, end_dt = service.get_default_time_range()

        data = service.get_events_json(
            start_dt, end_dt,
            event_type=event_type,
            classification=classification,
            min_confidence=min_confidence,
            page=page,
            page_size=page_size
        )

        return data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'[TrackEvents API] Error: {e}', exc_info=True)
        raise HTTPException(500, str(e))


@router.get("/api/track-events/stats", response_class=JSONResponse)
async def track_events_stats_api(
    start_time: Optional[str] = Query(None),
    end_time: Optional[str] = Query(None)
):
    """
    JSON API endpoint for enhanced track event statistics.

    Returns:
    - Event type breakdown with avg metrics
    - Classification distribution
    - Entry type distribution
    - Exit direction distribution
    - Ghost recovery statistics
    - Duration histogram
    - Confidence histogram
    """
    service = _get_service()

    try:
        if start_time and end_time:
            start_dt = service.parse_datetime(start_time)
            end_dt = service.parse_datetime(end_time)
            if start_dt >= end_dt:
                raise HTTPException(422, 'Start time must be before end time')
        else:
            start_dt, end_dt = service.get_default_time_range()

        stats = service.repo.get_enhanced_stats(start_dt, end_dt)

        return {
            'time_range': {
                'start': start_dt.isoformat(),
                'end': end_dt.isoformat()
            },
            'stats': stats
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'[TrackEvents Stats API] Error: {e}', exc_info=True)
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


@router.get("/track-events/{track_id}/animation", response_class=JSONResponse)
async def track_animation_data(track_id: int):
    """
    Get animation data for visualizing a track's lifecycle journey.

    Returns:
    - Position history (keyframes for animation)
    - ROI collection events with bounding boxes
    - Classification events
    - Lifecycle milestone events
    - Occlusion and merge events
    - Animation properties (suggested duration, frame count)

    Use this endpoint to render an SVG/Canvas animation of the track's
    journey from creation to completion/loss.
    """
    service = _get_service()
    try:
        data = service.get_track_animation(track_id)
        if data is None:
            raise HTTPException(404, f"Track {track_id} not found")
        return data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'[TrackEvents] Animation error: {e}', exc_info=True)
        raise HTTPException(500, str(e))


@router.get("/track-events/{track_id}/visualize", response_class=HTMLResponse)
async def track_visualization_page(request: Request, track_id: int):
    """
    Render a standalone page with animated track lifecycle visualization.

    Shows an SVG animation of the track's journey with:
    - Trajectory path
    - ROI collection points
    - Classification events
    - Entry/exit markers
    - Timeline of events
    """
    templates = get_templates()
    service = _get_service()

    try:
        animation_data = service.get_track_animation(track_id)
        if animation_data is None:
            raise HTTPException(404, f"Track {track_id} not found")

        context = {
            'request': request,
            'track_id': track_id,
            'data': animation_data
        }

        return templates.TemplateResponse('track_visualization_ar.html', context)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'[TrackEvents] Visualization error: {e}', exc_info=True)
        raise HTTPException(500, str(e))

