"""Track Lifecycle Routes - Track event lifecycle analytics endpoint.

Enhanced with:
- Advanced filtering (classification, confidence, duration, entry/exit type)
- Pagination support
- JSON API endpoints
- Track animation data endpoint
"""

import re
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


def _parse_optional_float(val: Optional[str]) -> Optional[float]:
    """Parse optional float from query param. Returns None for empty/blank strings."""
    if val is None or val.strip() == '':
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _parse_optional_int(val: Optional[str]) -> Optional[int]:
    """Parse optional int from query param. Returns None for empty/blank strings."""
    if val is None or val.strip() == '':
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _clean_str(val: Optional[str]) -> Optional[str]:
    """Return None for empty/blank strings."""
    if val is None or val.strip() == '':
        return None
    return val.strip()


def _parse_track_search(val: Optional[str]):
    """
    Parse track search input into track_ids list or track_id_range tuple.

    Supports:
      - Single ID:     "10" or "T10"
      - Comma list:    "10,20,30" or "T10, T20, T30"
      - Range:         "10-20" or "T10-T20"

    Returns:
        (track_ids, track_id_range) - one will be set, other None
    """
    if val is None or val.strip() == '':
        return None, None

    cleaned = val.strip().upper().replace('T', '')

    # Check for range pattern: "10-20"
    range_match = re.match(r'^(\d+)\s*-\s*(\d+)$', cleaned)
    if range_match:
        start = int(range_match.group(1))
        end = int(range_match.group(2))
        if start > end:
            start, end = end, start
        return None, (start, end)

    # Check for comma-separated list: "10,20,30"
    parts = [p.strip() for p in cleaned.split(',') if p.strip()]
    ids = []
    for p in parts:
        try:
            ids.append(int(p))
        except ValueError:
            continue
    if ids:
        return ids, None

    return None, None


@router.get("/track-events", response_class=HTMLResponse)
async def track_events_page(
    request: Request,
    start_time: Optional[str] = Query(None),
    end_time: Optional[str] = Query(None),
    event_type: Optional[str] = Query(None),
    classification: Optional[str] = Query(None),
    min_confidence: Optional[str] = Query(None),
    min_duration: Optional[str] = Query(None),
    max_duration: Optional[str] = Query(None),
    entry_type: Optional[str] = Query(None),
    exit_direction: Optional[str] = Query(None),
    min_distance: Optional[str] = Query(None),
    max_distance: Optional[str] = Query(None),
    min_hits: Optional[str] = Query(None),
    max_hits: Optional[str] = Query(None),
    min_frames: Optional[str] = Query(None),
    has_ghost_recovery: Optional[str] = Query(None),
    show_noise: Optional[str] = Query(None),
    track_search: Optional[str] = Query(None),
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
        track_search: Search by track ID. Supports: single (T10), comma list (T10,T20),
                      range (T10-T20). The T prefix is optional.
        start_time: ISO datetime start (optional, defaults to last 24h)
        end_time: ISO datetime end (optional, defaults to now)
        event_type: Filter by type: track_completed, track_lost, track_invalid
        classification: Filter by classification result (e.g., "Wheatberry")
        min_confidence: Minimum detection confidence (0.0-1.0)
        min_duration: Minimum track duration in seconds
        max_duration: Maximum track duration in seconds
        entry_type: Filter by entry type: bottom_entry, thrown_entry, midway_entry
        exit_direction: Filter by exit: top, bottom, left, right, timeout
        min_distance: Minimum travel distance in pixels
        max_distance: Maximum travel distance in pixels
        min_hits: Minimum detection hits count
        max_hits: Maximum detection hits count
        min_frames: Minimum total frames
        has_ghost_recovery: 'yes' or 'no' - filter by ghost recovery
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

        # Parse numeric filters (HTML forms send "" for empty fields)
        p_min_confidence = _parse_optional_float(min_confidence)
        p_min_duration = _parse_optional_float(min_duration)
        p_max_duration = _parse_optional_float(max_duration)
        p_min_distance = _parse_optional_float(min_distance)
        p_max_distance = _parse_optional_float(max_distance)
        p_min_hits = _parse_optional_int(min_hits)
        p_max_hits = _parse_optional_int(max_hits)
        p_min_frames = _parse_optional_int(min_frames)
        p_event_type = _clean_str(event_type)
        p_classification = _clean_str(classification)
        p_entry_type = _clean_str(entry_type)
        p_exit_direction = _clean_str(exit_direction)

        # Parse track search (single ID, comma list, or range)
        p_track_ids, p_track_id_range = _parse_track_search(track_search)
        track_search_str = _clean_str(track_search)

        # Convert has_ghost_recovery string to bool
        ghost_recovery_bool = None
        has_ghost_str = _clean_str(has_ghost_recovery)
        if has_ghost_str == 'yes':
            ghost_recovery_bool = True
        elif has_ghost_str == 'no':
            ghost_recovery_bool = False

        # ── Default noise filtering ──────────────────────────────────────
        # When user lands on the page without touching advanced filters,
        # we apply sensible defaults that hide noise (edge-of-frame false
        # detections, single-frame flashes, etc.).
        #   • min_hits  = 3   → hides ≤2 hit tracks (single-frame noise)
        #   • min_distance = 30 px → hides 0 px stationary false detections
        #   • min_duration = 0.3 s → hides ultra-short flashes (<0.3s)
        # The user can adjust any of these values in the Advanced Filters
        # panel (they appear pre-filled) or click "Show Noise" to remove
        # all defaults at once.
        # ─────────────────────────────────────────────────────────────────
        NOISE_DEFAULT_MIN_HITS = 3
        NOISE_DEFAULT_MIN_DISTANCE = 30.0
        NOISE_DEFAULT_MIN_DURATION = 0.3

        noise_filtering_active = False
        user_touched_advanced = (
            p_min_hits is not None or p_max_hits is not None or
            p_min_distance is not None or p_max_distance is not None or
            p_min_duration is not None or p_max_duration is not None or
            p_min_frames is not None or p_min_confidence is not None
        )

        # When searching specific tracks, disable noise filtering so they always appear
        searching_tracks = p_track_ids is not None or p_track_id_range is not None

        effective_min_hits = p_min_hits
        effective_min_distance = p_min_distance
        effective_min_duration = p_min_duration

        if show_noise != '1' and not user_touched_advanced and not searching_tracks:
            # No explicit advanced filter set and user hasn't clicked "show noise"
            effective_min_hits = NOISE_DEFAULT_MIN_HITS
            effective_min_distance = NOISE_DEFAULT_MIN_DISTANCE
            effective_min_duration = NOISE_DEFAULT_MIN_DURATION
            noise_filtering_active = True

        data = service.get_lifecycle_data(
            start_dt, end_dt,
            event_type=p_event_type,
            classification=p_classification,
            min_confidence=p_min_confidence,
            min_duration=effective_min_duration,
            max_duration=p_max_duration,
            entry_type=p_entry_type,
            exit_direction=p_exit_direction,
            min_distance=effective_min_distance,
            max_distance=p_max_distance,
            min_hits=effective_min_hits,
            max_hits=p_max_hits,
            min_frames=p_min_frames,
            has_ghost_recovery=ghost_recovery_bool,
            track_ids=p_track_ids,
            track_id_range=p_track_id_range,
            page=page,
            page_size=page_size
        )

        # Count noise tracks (<=2 hits) for the banner
        noise_count = 0
        if noise_filtering_active:
            noise_count = service.count_noise_tracks(start_dt, end_dt)

        context = {
            'request': request,
            'meta': data['meta'],
            'stats': data['stats'],
            'events': data['events'],
            'pagination': data['pagination'],
            'filter_options': data['filter_options'],
            'noise_filtering_active': noise_filtering_active,
            'noise_count': noise_count,
            'track_search': track_search_str or ''
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
    min_confidence: Optional[str] = Query(None),
    min_duration: Optional[str] = Query(None),
    max_duration: Optional[str] = Query(None),
    entry_type: Optional[str] = Query(None),
    exit_direction: Optional[str] = Query(None),
    min_distance: Optional[str] = Query(None),
    max_distance: Optional[str] = Query(None),
    min_hits: Optional[str] = Query(None),
    max_hits: Optional[str] = Query(None),
    min_frames: Optional[str] = Query(None),
    has_ghost_recovery: Optional[str] = Query(None),
    track_search: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=10, le=200)
):
    """
    JSON API endpoint for track events.

    Returns paginated track events with parsed JSON fields.
    Use this for programmatic access or custom frontends.

    Supports all filters: event_type, classification, min_confidence,
    min_duration, max_duration, entry_type, exit_direction,
    min_distance, max_distance, min_hits, max_hits, min_frames,
    has_ghost_recovery (yes/no), track_search (e.g. T10, T10-T20, T10,T20,T30).
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

        # Parse numeric filters (may be empty strings)
        p_min_confidence = _parse_optional_float(min_confidence)
        p_min_duration = _parse_optional_float(min_duration)
        p_max_duration = _parse_optional_float(max_duration)
        p_min_distance = _parse_optional_float(min_distance)
        p_max_distance = _parse_optional_float(max_distance)
        p_min_hits = _parse_optional_int(min_hits)
        p_max_hits = _parse_optional_int(max_hits)
        p_min_frames = _parse_optional_int(min_frames)
        p_event_type = _clean_str(event_type)
        p_classification = _clean_str(classification)
        p_entry_type = _clean_str(entry_type)
        p_exit_direction = _clean_str(exit_direction)

        # Parse track search
        p_track_ids, p_track_id_range = _parse_track_search(track_search)

        # Convert has_ghost_recovery string to bool
        ghost_recovery_bool = None
        has_ghost_str = _clean_str(has_ghost_recovery)
        if has_ghost_str == 'yes':
            ghost_recovery_bool = True
        elif has_ghost_str == 'no':
            ghost_recovery_bool = False

        data = service.get_events_json(
            start_dt, end_dt,
            event_type=p_event_type,
            classification=p_classification,
            min_confidence=p_min_confidence,
            min_duration=p_min_duration,
            max_duration=p_max_duration,
            entry_type=p_entry_type,
            exit_direction=p_exit_direction,
            min_distance=p_min_distance,
            max_distance=p_max_distance,
            min_hits=p_min_hits,
            max_hits=p_max_hits,
            min_frames=p_min_frames,
            has_ghost_recovery=ghost_recovery_bool,
            track_ids=p_track_ids,
            track_id_range=p_track_id_range,
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


@router.get("/track-events/by-event/{event_id}", response_class=JSONResponse)
async def track_lifecycle_by_event_id(event_id: int):
    """
    Get full lifecycle for a track by its UNIQUE event_id.

    This is the recommended endpoint for unambiguous track lookups since
    track_id resets on each app restart. The event_id is the unique
    primary key from the track_events table.

    Args:
        event_id: Unique ID from track_events.id (shown in the events table)

    Returns:
        Summary + all detail steps for this specific track event
    """
    db = get_db()
    try:
        data = db.get_track_lifecycle_by_event_id(event_id)
        if data['summary'] is None:
            raise HTTPException(404, f"Event {event_id} not found")
        return data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'[TrackEvents] Event detail error: {e}', exc_info=True)
        raise HTTPException(500, str(e))


@router.get("/track-events/{track_id}", response_class=JSONResponse)
async def track_lifecycle_detail(track_id: int):
    """
    Get full lifecycle for a single track as JSON.

    WARNING: track_id is NOT unique across sessions (resets on app restart).
    Use /track-events/by-event/{event_id} for unambiguous lookups.

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

