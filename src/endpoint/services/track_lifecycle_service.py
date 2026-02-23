"""
Track Lifecycle Service - Business Logic Layer.

Assembles track lifecycle data for the endpoint, including:
- Summary statistics (event type breakdown)
- Track event list with lifecycle detail steps
- Time range handling with timezone offset
- Animation data for track visualization

Enhanced with:
- Advanced filtering (classification, confidence, duration)
- Pagination support
- Enhanced statistics
- Animation data preparation
"""

import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from fastapi import HTTPException

from src.config.config_manager import get_config
from src.endpoint.repositories.track_lifecycle_repository import TrackLifecycleRepository
from src.utils.AppLogging import logger


class TrackLifecycleService:
    """Service for track event lifecycle analytics."""

    def __init__(self, repository: TrackLifecycleRepository):
        self.repo = repository
        self.config = get_config()

    @staticmethod
    def parse_datetime(val: Optional[str]) -> Optional[datetime]:
        """Parse ISO 8601 datetime string."""
        if not val:
            return None
        val = val.replace(' ', 'T').strip()
        if len(val) == 16:
            val += ":00"
        elif len(val) != 19:
            raise HTTPException(400, f"Invalid time format: {val}")
        try:
            return datetime.fromisoformat(val)
        except ValueError:
            raise HTTPException(400, f"Invalid datetime: {val}")

    def get_default_time_range(self) -> tuple:
        """Get default time range: last 24 hours."""
        # Use local system time directly (events are stored in local time)
        now = datetime.now()
        start = now - timedelta(hours=24)
        return start, now

    def get_lifecycle_data(
        self,
        start_time: datetime,
        end_time: datetime,
        event_type: Optional[str] = None,
        classification: Optional[str] = None,
        min_confidence: Optional[float] = None,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        entry_type: Optional[str] = None,
        exit_direction: Optional[str] = None,
        has_classification: Optional[bool] = None,
        min_distance: Optional[float] = None,
        max_distance: Optional[float] = None,
        min_hits: Optional[int] = None,
        max_hits: Optional[int] = None,
        min_frames: Optional[int] = None,
        has_ghost_recovery: Optional[bool] = None,
        page: int = 1,
        page_size: int = 100
    ) -> Dict[str, Any]:
        """
        Get complete lifecycle analytics data with pagination.

        Args:
            start_time: Local start time
            end_time: Local end time
            event_type: Optional filter by event type
            classification: Optional filter by classification result
            min_confidence: Minimum avg detection confidence
            min_duration: Minimum track duration in seconds
            max_duration: Maximum track duration in seconds
            entry_type: Filter by entry type
            exit_direction: Filter by exit direction
            has_classification: Filter by whether classification exists
            page: Page number (1-based)
            page_size: Events per page

        Returns:
            Dictionary with meta, stats, events, pagination, and filter options
        """
        db_start = start_time
        db_end = end_time

        offset = (page - 1) * page_size

        logger.info(
            f"[TrackLifecycle] Query: {db_start.isoformat()} to {db_end.isoformat()}, "
            f"type={event_type}, class={classification}, page={page}"
        )

        # Get enhanced stats (respects all active filters)
        stats = self.repo.get_enhanced_stats(
            db_start, db_end,
            event_type=event_type,
            classification=classification,
            min_confidence=min_confidence,
            min_duration=min_duration,
            max_duration=max_duration,
            entry_type=entry_type,
            exit_direction=exit_direction,
            has_classification=has_classification,
            min_distance=min_distance,
            max_distance=max_distance,
            min_hits=min_hits,
            max_hits=max_hits,
            min_frames=min_frames,
            has_ghost_recovery=has_ghost_recovery,
        )

        # Get paginated events with filters
        events, total_count = self.repo.get_track_events_page(
            db_start, db_end,
            event_type=event_type,
            classification=classification,
            min_confidence=min_confidence,
            min_duration=min_duration,
            max_duration=max_duration,
            entry_type=entry_type,
            exit_direction=exit_direction,
            has_classification=has_classification,
            min_distance=min_distance,
            max_distance=max_distance,
            min_hits=min_hits,
            max_hits=max_hits,
            min_frames=min_frames,
            has_ghost_recovery=has_ghost_recovery,
            limit=page_size,
            offset=offset
        )

        # Get detail steps for each track (batch query)
        # IMPORTANT: Pass time range to filter by current window (track_id is not unique across sessions)
        track_ids = [e['track_id'] for e in events]
        details_map = {}
        if track_ids:
            details_map = self.repo.get_track_event_details_for_tracks(
                track_ids,
                start_time=db_start,
                end_time=db_end
            )

        # Enrich events with their detail steps and parsed data
        for event in events:
            tid = event['track_id']
            event['details'] = details_map.get(tid, [])

            # Parse position_history for template use
            if event.get('position_history'):
                try:
                    event['_positions'] = json.loads(event['position_history'])
                except (json.JSONDecodeError, TypeError):
                    event['_positions'] = []
            else:
                event['_positions'] = []

            # Parse occlusion_events
            if event.get('occlusion_events'):
                try:
                    event['_occlusion_events'] = json.loads(event['occlusion_events'])
                except (json.JSONDecodeError, TypeError):
                    event['_occlusion_events'] = []
            else:
                event['_occlusion_events'] = []

            # Parse merge_events
            if event.get('merge_events'):
                try:
                    event['_merge_events'] = json.loads(event['merge_events'])
                except (json.JSONDecodeError, TypeError):
                    event['_merge_events'] = []
            else:
                event['_merge_events'] = []

        # Get distinct classifications for filter dropdown
        distinct_classifications = self.repo.get_distinct_classifications(db_start, db_end)

        # Calculate pagination info
        total_pages = (total_count + page_size - 1) // page_size if total_count > 0 else 1

        logger.info(
            f"[TrackLifecycle] Found {len(events)} events (page {page}/{total_pages}), "
            f"total={total_count}, stats total={stats.get('total', 0)}"
        )

        return {
            'meta': {
                'start': start_time,
                'end': end_time,
                'request_time': datetime.now().strftime("%Y/%m/%d - %H:%M:%S"),
                'event_type_filter': event_type,
                'classification_filter': classification,
                'min_confidence_filter': min_confidence,
                'min_duration_filter': min_duration,
                'max_duration_filter': max_duration,
                'entry_type_filter': entry_type,
                'exit_direction_filter': exit_direction,
                'min_distance_filter': min_distance,
                'max_distance_filter': max_distance,
                'min_hits_filter': min_hits,
                'max_hits_filter': max_hits,
                'min_frames_filter': min_frames,
                'has_ghost_recovery_filter': 'yes' if has_ghost_recovery is True else ('no' if has_ghost_recovery is False else None)
            },
            'stats': stats,
            'events': events,
            'pagination': {
                'page': page,
                'page_size': page_size,
                'total_count': total_count,
                'total_pages': total_pages,
                'has_prev': page > 1,
                'has_next': page < total_pages
            },
            'filter_options': {
                'classifications': distinct_classifications,
                'event_types': ['track_completed', 'track_lost', 'track_invalid'],
                'entry_types': ['bottom_entry', 'thrown_entry', 'midway_entry'],
                'exit_directions': ['top', 'bottom', 'left', 'right', 'timeout']
            }
        }

    def get_track_animation(self, track_id: int) -> Dict[str, Any]:
        """
        Get animation data for a track's lifecycle visualization.

        Returns:
            Dictionary with all data needed for SVG animation
        """
        animation_data = self.repo.get_track_animation_data(track_id)

        if not animation_data:
            return None

        # Add computed animation properties
        positions = animation_data.get('position_history', [])

        if len(positions) >= 2:
            # Calculate animation keyframes from position history
            total_distance = 0
            keyframes = []

            for i, pos in enumerate(positions):
                if i > 0:
                    dx = pos[0] - positions[i-1][0]
                    dy = pos[1] - positions[i-1][1]
                    total_distance += (dx**2 + dy**2) ** 0.5

                keyframes.append({
                    'index': i,
                    'x': pos[0],
                    'y': pos[1],
                    'progress': i / (len(positions) - 1) if len(positions) > 1 else 1
                })

            animation_data['animation'] = {
                'keyframes': keyframes,
                'total_distance': total_distance,
                'frame_count': len(positions),
                'suggested_duration_ms': min(
                    max(len(positions) * 100, 2000),  # At least 2s, 100ms per frame
                    10000  # Max 10s
                )
            }
        else:
            animation_data['animation'] = {
                'keyframes': [],
                'total_distance': 0,
                'frame_count': 0,
                'suggested_duration_ms': 2000
            }

        return animation_data

    def count_noise_tracks(self, start_time: datetime, end_time: datetime) -> int:
        """Count noise tracks (<=2 hits) in time range for the noise banner."""
        return self.repo.count_noise_tracks(start_time, end_time)

    def get_events_json(
        self,
        start_time: datetime,
        end_time: datetime,
        event_type: Optional[str] = None,
        classification: Optional[str] = None,
        min_confidence: Optional[float] = None,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        entry_type: Optional[str] = None,
        exit_direction: Optional[str] = None,
        min_distance: Optional[float] = None,
        max_distance: Optional[float] = None,
        min_hits: Optional[int] = None,
        max_hits: Optional[int] = None,
        min_frames: Optional[int] = None,
        has_ghost_recovery: Optional[bool] = None,
        page: int = 1,
        page_size: int = 100
    ) -> Dict[str, Any]:
        """
        Get track events as JSON (for API consumption).

        Returns a lighter payload without template-specific processing.
        Supports all filters including distance, hits, frames, and ghost recovery.
        """
        events, total_count = self.repo.get_track_events_page(
            start_time, end_time,
            event_type=event_type,
            classification=classification,
            min_confidence=min_confidence,
            min_duration=min_duration,
            max_duration=max_duration,
            entry_type=entry_type,
            exit_direction=exit_direction,
            min_distance=min_distance,
            max_distance=max_distance,
            min_hits=min_hits,
            max_hits=max_hits,
            min_frames=min_frames,
            has_ghost_recovery=has_ghost_recovery,
            limit=page_size,
            offset=(page - 1) * page_size
        )

        # Parse JSON fields
        for event in events:
            if event.get('position_history'):
                try:
                    event['position_history'] = json.loads(event['position_history'])
                except (json.JSONDecodeError, TypeError):
                    event['position_history'] = []
            if event.get('occlusion_events'):
                try:
                    event['occlusion_events'] = json.loads(event['occlusion_events'])
                except (json.JSONDecodeError, TypeError):
                    event['occlusion_events'] = []
            if event.get('merge_events'):
                try:
                    event['merge_events'] = json.loads(event['merge_events'])
                except (json.JSONDecodeError, TypeError):
                    event['merge_events'] = []

        total_pages = (total_count + page_size - 1) // page_size if total_count > 0 else 1

        return {
            'events': events,
            'pagination': {
                'page': page,
                'page_size': page_size,
                'total_count': total_count,
                'total_pages': total_pages
            }
        }
