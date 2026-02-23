"""
Track Lifecycle Repository - Data Access Layer.

Provides database queries for track event lifecycle analytics,
including summary stats, per-track details, and ROI lifecycle steps.

Enhanced with:
- Batch queries for better performance
- Advanced filtering (classification, confidence, duration)
- Pagination support
- Animation data extraction
"""

from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from src.logging.Database import DatabaseManager
from src.utils.AppLogging import logger


class TrackLifecycleRepository:
    """Repository for track event lifecycle data access."""

    def __init__(self, db: DatabaseManager):
        self.db = db

    def _build_filter_clause(
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
        track_ids: Optional[List[int]] = None,
        track_id_range: Optional[Tuple[int, int]] = None,
    ) -> Tuple[str, list]:
        """
        Build a reusable WHERE clause and params list from all filter options.

        Returns:
            Tuple of (where_clause_string, params_list)
        """
        conditions = ["timestamp >= ?", "timestamp <= ?"]
        params: list = [start_time.isoformat(), end_time.isoformat()]

        if event_type:
            conditions.append("event_type = ?")
            params.append(event_type)
        if classification:
            conditions.append("classification = ?")
            params.append(classification)
        if min_confidence is not None:
            conditions.append("avg_confidence >= ?")
            params.append(min_confidence)
        if min_duration is not None:
            conditions.append("duration_seconds >= ?")
            params.append(min_duration)
        if max_duration is not None:
            conditions.append("duration_seconds <= ?")
            params.append(max_duration)
        if entry_type:
            conditions.append("entry_type = ?")
            params.append(entry_type)
        if exit_direction:
            conditions.append("exit_direction = ?")
            params.append(exit_direction)
        if has_classification is not None:
            if has_classification:
                conditions.append("classification IS NOT NULL")
            else:
                conditions.append("classification IS NULL")
        if min_distance is not None:
            conditions.append("distance_pixels >= ?")
            params.append(min_distance)
        if max_distance is not None:
            conditions.append("distance_pixels <= ?")
            params.append(max_distance)
        if min_hits is not None:
            conditions.append("total_hits >= ?")
            params.append(min_hits)
        if max_hits is not None:
            conditions.append("total_hits <= ?")
            params.append(max_hits)
        if min_frames is not None:
            conditions.append("total_frames >= ?")
            params.append(min_frames)
        if has_ghost_recovery is not None:
            if has_ghost_recovery:
                conditions.append("ghost_recovery_count > 0")
            else:
                conditions.append("(ghost_recovery_count IS NULL OR ghost_recovery_count = 0)")
        if track_ids:
            placeholders = ','.join('?' * len(track_ids))
            conditions.append(f"track_id IN ({placeholders})")
            params.extend(track_ids)
        if track_id_range:
            conditions.append("track_id >= ? AND track_id <= ?")
            params.extend([track_id_range[0], track_id_range[1]])

        return " AND ".join(conditions), params

    def get_track_events_page(
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
        track_ids: Optional[List[int]] = None,
        track_id_range: Optional[Tuple[int, int]] = None,
        limit: int = 500,
        offset: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Get track events for a time range with advanced filtering.
        """
        where_clause, params = self._build_filter_clause(
            start_time, end_time,
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
            track_ids=track_ids,
            track_id_range=track_id_range,
        )

        # Get total count for pagination
        count_query = f"SELECT COUNT(*) as total FROM track_events WHERE {where_clause}"

        # Get paginated events
        data_query = f"""
            SELECT * FROM track_events
            WHERE {where_clause}
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
        """

        with self.db._cursor() as cursor:
            cursor.execute(count_query, params)
            total = cursor.fetchone()['total']

            cursor.execute(data_query, params + [limit, offset])
            events = [dict(row) for row in cursor.fetchall()]

        return events, total

    def get_track_event_stats(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """Get aggregated stats for time range with enhanced metrics."""
        return self.db.get_track_event_stats(
            start_date=start_time.isoformat(),
            end_date=end_time.isoformat()
        )

    def get_enhanced_stats(
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
        track_ids: Optional[List[int]] = None,
        track_id_range: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Any]:
        """
        Get enhanced statistics respecting all active filters.

        All stat queries apply the same filters as the events table so
        totals, breakdowns, and histograms match the filtered view.

        For the by-type breakdown, the event_type filter is intentionally
        excluded so the user always sees completed/lost/invalid counts.
        """
        # Full filter clause (all filters)
        where_clause, params = self._build_filter_clause(
            start_time, end_time,
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
            track_ids=track_ids,
            track_id_range=track_id_range,
        )

        # For by-type breakdown: exclude event_type filter so all 3 types are visible
        where_no_type, params_no_type = self._build_filter_clause(
            start_time, end_time,
            event_type=None,  # intentionally excluded
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
            track_ids=track_ids,
            track_id_range=track_id_range,
        )

        with self.db._cursor() as cursor:
            # Basic stats by event type (uses filters WITHOUT event_type)
            cursor.execute(f"""
                SELECT
                    event_type,
                    COUNT(*) as count,
                    AVG(duration_seconds) as avg_duration,
                    AVG(distance_pixels) as avg_distance,
                    AVG(avg_confidence) as avg_confidence,
                    MIN(duration_seconds) as min_duration,
                    MAX(duration_seconds) as max_duration
                FROM track_events
                WHERE {where_no_type}
                GROUP BY event_type
            """, params_no_type)
            by_type = {row['event_type']: dict(row) for row in cursor.fetchall()}

            # Classification breakdown (uses full filters)
            cursor.execute(f"""
                SELECT classification, COUNT(*) as count,
                       AVG(classification_confidence) as avg_conf
                FROM track_events
                WHERE {where_clause}
                  AND classification IS NOT NULL
                GROUP BY classification
                ORDER BY count DESC
            """, params)
            by_classification = [dict(row) for row in cursor.fetchall()]

            # Entry type distribution (uses full filters)
            cursor.execute(f"""
                SELECT entry_type, COUNT(*) as count
                FROM track_events
                WHERE {where_clause}
                GROUP BY entry_type
            """, params)
            by_entry_type = {row['entry_type'] or 'unknown': row['count'] for row in cursor.fetchall()}

            # Exit direction distribution (uses full filters)
            cursor.execute(f"""
                SELECT exit_direction, COUNT(*) as count
                FROM track_events
                WHERE {where_clause}
                GROUP BY exit_direction
            """, params)
            by_exit_direction = {row['exit_direction'] or 'unknown': row['count'] for row in cursor.fetchall()}

            # Ghost recovery stats (uses full filters)
            cursor.execute(f"""
                SELECT
                    SUM(CASE WHEN ghost_recovery_count > 0 THEN 1 ELSE 0 END) as recovered_tracks,
                    SUM(ghost_recovery_count) as total_recoveries,
                    SUM(shadow_count) as total_shadows
                FROM track_events
                WHERE {where_clause}
            """, params)
            recovery_row = cursor.fetchone()
            recovery_stats = dict(recovery_row) if recovery_row else {}

            # Duration histogram (uses full filters)
            cursor.execute(f"""
                SELECT
                    CASE
                        WHEN duration_seconds < 1 THEN '0-1s'
                        WHEN duration_seconds < 2 THEN '1-2s'
                        WHEN duration_seconds < 3 THEN '2-3s'
                        WHEN duration_seconds < 5 THEN '3-5s'
                        ELSE '5s+'
                    END as bucket,
                    COUNT(*) as count
                FROM track_events
                WHERE {where_clause}
                GROUP BY bucket
            """, params)
            duration_histogram = {row['bucket']: row['count'] for row in cursor.fetchall()}

            # Confidence histogram (uses full filters)
            cursor.execute(f"""
                SELECT
                    CASE
                        WHEN avg_confidence < 0.5 THEN '< 50%'
                        WHEN avg_confidence < 0.7 THEN '50-70%'
                        WHEN avg_confidence < 0.85 THEN '70-85%'
                        WHEN avg_confidence < 0.95 THEN '85-95%'
                        ELSE '95-100%'
                    END as bucket,
                    COUNT(*) as count
                FROM track_events
                WHERE {where_clause}
                  AND avg_confidence IS NOT NULL
                GROUP BY bucket
            """, params)
            confidence_histogram = {row['bucket']: row['count'] for row in cursor.fetchall()}

            total = sum(r['count'] for r in by_type.values())

        return {
            'total': total,
            'by_type': by_type,
            'by_classification': by_classification,
            'by_entry_type': by_entry_type,
            'by_exit_direction': by_exit_direction,
            'recovery_stats': recovery_stats,
            'duration_histogram': duration_histogram,
            'confidence_histogram': confidence_histogram
        }

    def get_track_lifecycle(self, track_id: int) -> Dict[str, Any]:
        """Get full lifecycle for a single track."""
        return self.db.get_track_lifecycle(track_id)

    def get_track_event_details_for_tracks(
        self,
        track_ids: List[int],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Get detail steps for a list of track IDs in a single batch query.

        IMPORTANT: track_id is NOT unique across sessions (resets on app restart).
        We must filter by timestamp to get only details within the current time window.

        Args:
            track_ids: List of track IDs to fetch details for
            start_time: Start of time window (required for correct filtering)
            end_time: End of time window (required for correct filtering)

        Returns:
            Dictionary mapping track_id -> list of detail steps
        """
        if not track_ids:
            return {}

        # Build query with timestamp filtering to avoid cross-session data
        placeholders = ','.join('?' * len(track_ids))
        params = list(track_ids)

        # Add timestamp filtering if provided (CRITICAL for session isolation)
        time_filter = ""
        if start_time and end_time:
            time_filter = " AND timestamp >= ? AND timestamp <= ?"
            params.extend([start_time.isoformat(), end_time.isoformat()])

        query = f"""
            SELECT * FROM track_event_details
            WHERE track_id IN ({placeholders}){time_filter}
            ORDER BY track_id, id ASC
        """

        with self.db._cursor() as cursor:
            cursor.execute(query, params)
            rows = [dict(row) for row in cursor.fetchall()]

        # Group by track_id
        result = {tid: [] for tid in track_ids}
        for row in rows:
            tid = row['track_id']
            if tid in result:
                result[tid].append(row)

        return result

    def get_track_animation_data(self, track_id: int) -> Dict[str, Any]:
        """
        Get animation data for a track's lifecycle visualization.

        Returns:
            Dictionary with position history, ROI collections, and classification events
        """
        lifecycle = self.db.get_track_lifecycle(track_id)

        if not lifecycle['summary']:
            return None

        summary = lifecycle['summary']
        details = lifecycle['details']

        # Parse position history
        position_history = []
        if summary.get('position_history'):
            import json
            try:
                position_history = json.loads(summary['position_history'])
            except (json.JSONDecodeError, TypeError):
                pass

        # Extract ROI collection events
        roi_events = []
        classification_events = []
        lifecycle_events = []

        for detail in details:
            step_type = detail['step_type']

            if step_type == 'roi_collected':
                roi_events.append({
                    'timestamp': detail['timestamp'],
                    'roi_index': detail['roi_index'],
                    'bbox': {
                        'x1': detail['bbox_x1'],
                        'y1': detail['bbox_y1'],
                        'x2': detail['bbox_x2'],
                        'y2': detail['bbox_y2']
                    },
                    'quality_score': detail['quality_score']
                })
            elif step_type == 'roi_classified':
                classification_events.append({
                    'timestamp': detail['timestamp'],
                    'roi_index': detail['roi_index'],
                    'class_name': detail['class_name'],
                    'confidence': detail['confidence'],
                    'is_rejected': detail['is_rejected']
                })
            elif step_type in ('track_created', 'track_completed', 'track_lost', 'track_invalid',
                               'voting_result', 'ghost_moved', 'ghost_recovered'):
                lifecycle_events.append({
                    'timestamp': detail['timestamp'],
                    'step_type': step_type,
                    'class_name': detail.get('class_name'),
                    'confidence': detail.get('confidence'),
                    'detail': detail.get('detail')
                })

        # Parse occlusion and merge events
        occlusion_events = []
        merge_events = []

        if summary.get('occlusion_events'):
            import json
            try:
                occlusion_events = json.loads(summary['occlusion_events'])
            except (json.JSONDecodeError, TypeError):
                pass

        if summary.get('merge_events'):
            import json
            try:
                merge_events = json.loads(summary['merge_events'])
            except (json.JSONDecodeError, TypeError):
                pass

        # If no position history, generate simulated path from entry to exit
        if not position_history:
            entry_x = summary.get('entry_x')
            entry_y = summary.get('entry_y')
            exit_x = summary.get('exit_x')
            exit_y = summary.get('exit_y')

            if entry_x is not None and entry_y is not None and exit_x is not None and exit_y is not None:
                # Generate straight line path with 50 points (matching actual conveyor movement)
                num_points = 50
                for i in range(num_points):
                    t = i / (num_points - 1)
                    # Straight line interpolation - no curve
                    x = entry_x + (exit_x - entry_x) * t
                    y = entry_y + (exit_y - entry_y) * t
                    position_history.append([int(x), int(y)])

        return {
            'track_id': track_id,
            'summary': {
                'event_type': summary['event_type'],
                'entry': {'x': summary['entry_x'], 'y': summary['entry_y']},
                'exit': {'x': summary['exit_x'], 'y': summary['exit_y']},
                'entry_type': summary.get('entry_type'),
                'exit_direction': summary.get('exit_direction'),
                'duration_seconds': summary['duration_seconds'],
                'distance_pixels': summary['distance_pixels'],
                'classification': summary.get('classification'),
                'classification_confidence': summary.get('classification_confidence'),
                'created_at': summary['created_at'],
                'ended_at': summary['timestamp'],
                'ghost_recovery_count': summary.get('ghost_recovery_count', 0),
                'shadow_of': summary.get('shadow_of'),
                'shadow_count': summary.get('shadow_count', 0)
            },
            'position_history': position_history,
            'roi_events': roi_events,
            'classification_events': classification_events,
            'lifecycle_events': lifecycle_events,
            'occlusion_events': occlusion_events,
            'merge_events': merge_events
        }

    def get_distinct_classifications(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> List[str]:
        """Get list of distinct classifications in the time range."""
        with self.db._cursor() as cursor:
            cursor.execute("""
                SELECT DISTINCT classification
                FROM track_events
                WHERE timestamp >= ? AND timestamp <= ?
                  AND classification IS NOT NULL
                ORDER BY classification
            """, (start_time.isoformat(), end_time.isoformat()))
            return [row['classification'] for row in cursor.fetchall()]

    def count_noise_tracks(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> int:
        """
        Count noise tracks in the time range.

        Noise tracks match ANY of these criteria:
        - ≤2 detection hits (single-frame false detections)
        - Travel distance < 30px (stationary edge artifacts)
        - Duration < 0.3s (ultra-short flashes)

        Returns:
            Count of noise tracks
        """
        with self.db._cursor() as cursor:
            cursor.execute("""
                SELECT COUNT(*) as count
                FROM track_events
                WHERE timestamp >= ? AND timestamp <= ?
                  AND (
                      (total_hits IS NULL OR total_hits <= 2)
                      OR (distance_pixels IS NOT NULL AND distance_pixels < 30)
                      OR (duration_seconds IS NOT NULL AND duration_seconds < 0.3)
                  )
            """, (start_time.isoformat(), end_time.isoformat()))
            row = cursor.fetchone()
            return row['count'] if row else 0
