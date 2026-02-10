"""
Track Lifecycle Repository - Data Access Layer.

Provides database queries for track event lifecycle analytics,
including summary stats, per-track details, and ROI lifecycle steps.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional

from src.logging.Database import DatabaseManager
from src.utils.AppLogging import logger


class TrackLifecycleRepository:
    """Repository for track event lifecycle data access."""

    def __init__(self, db: DatabaseManager):
        self.db = db

    def get_track_events_page(
        self,
        start_time: datetime,
        end_time: datetime,
        event_type: Optional[str] = None,
        limit: int = 500
    ) -> List[Dict[str, Any]]:
        """
        Get track events for a time range, ordered by most recent first.

        Args:
            start_time: Start datetime (UTC)
            end_time: End datetime (UTC)
            event_type: Optional filter ('track_completed', 'track_lost', 'track_invalid')
            limit: Maximum results

        Returns:
            List of track event dictionaries
        """
        return self.db.get_track_events(
            event_type=event_type,
            start_date=start_time.isoformat(),
            end_date=end_time.isoformat(),
            limit=limit
        )

    def get_track_event_stats(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """Get aggregated stats for time range."""
        return self.db.get_track_event_stats(
            start_date=start_time.isoformat(),
            end_date=end_time.isoformat()
        )

    def get_track_lifecycle(self, track_id: int) -> Dict[str, Any]:
        """Get full lifecycle for a single track."""
        return self.db.get_track_lifecycle(track_id)

    def get_track_event_details_for_tracks(
        self,
        track_ids: List[int]
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Get detail steps for a list of track IDs.

        Returns:
            Dictionary mapping track_id -> list of detail steps
        """
        result = {}
        for tid in track_ids:
            result[tid] = self.db.get_track_event_details(track_id=tid)
        return result
