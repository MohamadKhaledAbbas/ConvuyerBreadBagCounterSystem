"""
Track Lifecycle Service - Business Logic Layer.

Assembles track lifecycle data for the endpoint, including:
- Summary statistics (event type breakdown)
- Track event list with lifecycle detail steps
- Time range handling with timezone offset
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
        now = datetime.now() + timedelta(hours=self.config.timezone_offset_hours)
        start = now - timedelta(hours=24)
        return start, now

    def get_lifecycle_data(
        self,
        start_time: datetime,
        end_time: datetime,
        event_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get complete lifecycle analytics data.

        Args:
            start_time: Local start time
            end_time: Local end time
            event_type: Optional filter by event type

        Returns:
            Dictionary with meta, stats, events, and details
        """
        # Convert to UTC for DB queries
        db_start = start_time - timedelta(hours=self.config.timezone_offset_hours)
        db_end = end_time - timedelta(hours=self.config.timezone_offset_hours)

        logger.info(f"[TrackLifecycle] Query: {db_start.isoformat()} to {db_end.isoformat()}, type={event_type}")

        # Get stats
        stats = self.repo.get_track_event_stats(db_start, db_end)

        # Get events
        events = self.repo.get_track_events_page(db_start, db_end, event_type=event_type, limit=500)

        # Get detail steps for each track
        track_ids = [e['track_id'] for e in events]
        details_map = {}
        if track_ids:
            details_map = self.repo.get_track_event_details_for_tracks(track_ids)

        # Enrich events with their detail steps
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

        # Reverse to show newest first
        events.reverse()

        logger.info(f"[TrackLifecycle] Found {len(events)} events, stats total={stats.get('total', 0)}")

        return {
            'meta': {
                'start': start_time,
                'end': end_time,
                'request_time': datetime.now().strftime("%Y/%m/%d - %H:%M:%S"),
                'event_type_filter': event_type
            },
            'stats': stats,
            'events': events
        }
