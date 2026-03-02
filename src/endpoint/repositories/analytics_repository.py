"""
Analytics Repository - Data Access Layer.

Abstracts complex database queries for analytics use cases.
Provides high-level methods that hide SQL details from services.

Architecture: Service → Repository → Database
"""

from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Any

from src.logging.Database import DatabaseManager
from src.utils.AppLogging import logger


class AnalyticsRepository:
    """
    Repository for analytics data access.

    Provides domain-specific query methods that return ready-to-use data
    for analytics services.

    Benefits:
    - Encapsulates complex queries
    - Testable (can mock DatabaseManager)
    - Single responsibility (data access only)
    - Reusable across services
    """

    def __init__(self, db: DatabaseManager):
        """
        Initialize repository with database manager.

        Args:
            db: DatabaseManager instance
        """
        self.db = db

    def get_time_range_analytics(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """
        Get complete analytics data for time range.

        Returns all data needed for analytics display:
        - Aggregated statistics (total, by type)
        - Events with full bag_type metadata
        - Per-class time windows (first/last seen)

        Args:
            start_time: Start datetime (UTC)
            end_time: End datetime (UTC)

        Returns:
            Dictionary with:
            - stats: Aggregated statistics
            - events: List of events with metadata
            - per_class_windows: Dict of per-class summaries
        """
        start_iso = start_time.isoformat()
        end_iso = end_time.isoformat()

        logger.info(f"[Repository] Fetching analytics: {start_iso} to {end_iso}")

        # Get aggregated statistics with high/low confidence breakdown
        stats = self.db.get_aggregated_stats(start_iso, end_iso)

        # Get all events with joined bag_type metadata
        events = self.db.get_events_with_bag_types(start_iso, end_iso, limit=10000)

        # Build per-class time windows
        per_class = self._build_per_class_windows(events)

        logger.info(
            f"[Repository] Retrieved {stats['total']['count']} events, "
            f"{len(stats['by_type'])} classes"
        )

        return {
            'stats': stats,
            'events': events,
            'per_class_windows': per_class
        }

    def _build_per_class_windows(self, events: List[Dict]) -> Dict[str, Any]:
        """
        Build first/last seen windows per class.

        For each bag type, tracks:
        - First time seen
        - Last time seen
        - Total event count
        - Metadata (name, thumb, etc.)

        Args:
            events: List of events with bag_type metadata

        Returns:
            Dictionary keyed by bag_type_id
        """
        windows = defaultdict(lambda: {
            'first_seen': None,
            'last_seen': None,
            'event_count': 0,
            'bag_type_id': None,
            'name': '',
            'arabic_name': '',
            'thumb': '',
            'weight': 0
        })

        for event in events:
            bag_type_id = event['bag_type_id']
            timestamp = event['timestamp']

            # Initialize on first occurrence
            if windows[bag_type_id]['first_seen'] is None:
                windows[bag_type_id].update({
                    'bag_type_id': bag_type_id,
                    'name': event['bag_type'],
                    'arabic_name': event['arabic_name'],
                    'thumb': event['thumb'],
                    'weight': event['weight'],
                    'first_seen': timestamp
                })

            # Always update last_seen
            windows[bag_type_id]['last_seen'] = timestamp
            windows[bag_type_id]['event_count'] += 1

        return dict(windows)

    def get_all_bag_types(self) -> List[Dict[str, Any]]:
        """
        Get all registered bag types.

        Returns:
            List of bag_type dictionaries with all metadata
        """
        return self.db.get_all_bag_types()

    def get_bag_type_summary(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict[str, Any]]:
        """
        Get summary statistics per bag type for time range.

        Args:
            start_time: Start datetime (UTC)
            end_time: End datetime (UTC)

        Returns:
            List of bag type summaries with counts
        """
        stats = self.db.get_aggregated_stats(
            start_time.isoformat(),
            end_time.isoformat()
        )

        summaries = []
        for bag_type_id, data in stats['by_type'].items():
            summaries.append({
                'bag_type_id': bag_type_id,
                'name': data['name'],
                'arabic_name': data['arabic_name'],
                'count': data['count'],
                'high_count': data['high_count'],
                'low_count': data['low_count'],
                'weight': data['weight'],
                'total_weight': data['count'] * data['weight'],
                'thumb': data['thumb']
            })

        return sorted(summaries, key=lambda x: x['count'], reverse=True)

    def get_shift_report_data(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """
        Query track_events for shift performance metrics in a time range.

        Returns counts for: completed, rejected, lost (by cause), invalid paths.
        """
        s = start_time.isoformat()
        e = end_time.isoformat()
        time_filter = "timestamp >= ? AND timestamp <= ?"

        results: Dict[str, Any] = {}
        try:
            with self.db._cursor() as cursor:
                # Total completed tracks
                cursor.execute(
                    f"SELECT COUNT(*) FROM track_events WHERE {time_filter} AND event_type='track_completed'",
                    (s, e)
                )
                results['completed'] = cursor.fetchone()[0] or 0

                # Rejected bags (completed but classified as Rejected)
                cursor.execute(
                    f"SELECT COUNT(*) FROM track_events WHERE {time_filter} AND event_type='track_completed' AND classification='Rejected'",
                    (s, e)
                )
                results['rejected'] = cursor.fetchone()[0] or 0

                # Lost tracks (total + breakdown by exit_direction)
                cursor.execute(
                    f"SELECT COUNT(*) FROM track_events WHERE {time_filter} AND event_type='track_lost'",
                    (s, e)
                )
                results['lost_total'] = cursor.fetchone()[0] or 0

                cursor.execute(
                    f"SELECT exit_direction, COUNT(*) as cnt FROM track_events WHERE {time_filter} AND event_type='track_lost' GROUP BY exit_direction",
                    (s, e)
                )
                results['lost_by_cause'] = {row[0]: row[1] for row in cursor.fetchall()}

                # Lost tracks with occlusion events (camera was blocked)
                cursor.execute(
                    f"SELECT COUNT(*) FROM track_events WHERE {time_filter} AND event_type='track_lost' AND occlusion_events IS NOT NULL AND occlusion_events != '[]'",
                    (s, e)
                )
                results['lost_with_occlusion'] = cursor.fetchone()[0] or 0

                # Invalid path tracks
                cursor.execute(
                    f"SELECT COUNT(*) FROM track_events WHERE {time_filter} AND event_type='track_invalid'",
                    (s, e)
                )
                results['invalid'] = cursor.fetchone()[0] or 0

                # Total all tracks (for denominator)
                cursor.execute(
                    f"SELECT COUNT(*) FROM track_events WHERE {time_filter}",
                    (s, e)
                )
                results['total_tracks'] = cursor.fetchone()[0] or 0

        except Exception as exc:
            logger.error(f"[Repository] shift report query error: {exc}")
            results = {
                'completed': 0, 'rejected': 0, 'lost_total': 0,
                'lost_by_cause': {}, 'lost_with_occlusion': 0,
                'invalid': 0, 'total_tracks': 0,
            }

        return results

