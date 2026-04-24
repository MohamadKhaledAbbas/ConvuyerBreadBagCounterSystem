"""
Container Repository for database operations.

Provides data access methods for container tracking events and statistics.
Follows the repository pattern used by other modules in the project.

Usage:
    from src.endpoint.repositories.container_repository import ContainerRepository
    
    repo = ContainerRepository(db)
    events = repo.get_events(start_time, end_time)
    stats = repo.get_aggregated_stats(start_time, end_time)
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from src.logging.Database import DatabaseManager
from src.utils.AppLogging import logger


class ContainerRepository:
    """
    Repository for container tracking data access.
    
    Provides methods to query container_events and container_stats tables.
    """
    
    def __init__(self, db: DatabaseManager):
        """
        Initialize the container repository.
        
        Args:
            db: DatabaseManager instance
        """
        self.db = db
    
    def _safe_load_metadata(self, raw) -> Dict:
        """Safely parse the metadata JSON stored in the DB.

        Returns an empty dict when parsing fails or when the raw value
        is empty or not a dict.
        """
        if not raw:
            return {}
        try:
            if isinstance(raw, (bytes, bytearray)):
                raw = raw.decode('utf-8', errors='ignore')
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
            return {}
        except Exception:
            logger.warning("[ContainerRepository] failed to parse metadata JSON; returning empty dict")
            return {}
    
    def get_events(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        direction: Optional[str] = None,
        qr_code_value: Optional[int] = None,
        is_lost: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict]:
        """
        Get container events with optional filtering.
        
        Args:
            start_time: ISO 8601 start time filter
            end_time: ISO 8601 end time filter
            direction: Filter by direction ('positive', 'negative', 'unknown')
            qr_code_value: Filter by QR code value (1-5)
            is_lost: Filter by lost status (True/False)
            limit: Maximum events to return
            offset: Pagination offset
            
        Returns:
            List of event dictionaries
        """
        query = """
            SELECT 
                id, timestamp, qr_code_value, direction, track_id,
                entry_y, exit_y, duration_seconds, snapshot_path, metadata,
                COALESCE(is_lost, 0) as is_lost
            FROM container_events
            WHERE 1=1
        """
        params = []
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        
        if direction:
            query += " AND direction = ?"
            params.append(direction)
        
        if qr_code_value:
            query += " AND qr_code_value = ?"
            params.append(qr_code_value)
        
        if is_lost is not None:
            query += " AND COALESCE(is_lost, 0) = ?"
            params.append(1 if is_lost else 0)
        
        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        rows = self.db.fetchall(query, params)
        
        return [
            {
                'id': row[0],
                'timestamp': row[1],
                'qr_code_value': row[2],
                'direction': row[3],
                'track_id': row[4],
                'entry_x': row[5],
                'exit_x': row[6],
                'duration_seconds': row[7],
                'snapshot_path': row[8],
                'metadata': self._safe_load_metadata(row[9]),
                'is_lost': bool(row[10]) if len(row) > 10 else False,
            }
            for row in rows
        ]
    
    def get_event_count(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        direction: Optional[str] = None,
        qr_code_value: Optional[int] = None,
        is_lost: Optional[bool] = None,
    ) -> int:
        """Get total count of events matching filters."""
        query = "SELECT COUNT(*) FROM container_events WHERE 1=1"
        params = []
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        
        if direction:
            query += " AND direction = ?"
            params.append(direction)
        
        if qr_code_value:
            query += " AND qr_code_value = ?"
            params.append(qr_code_value)
        
        if is_lost is not None:
            query += " AND COALESCE(is_lost, 0) = ?"
            params.append(1 if is_lost else 0)
        
        result = self.db.fetchone(query, params)
        return result[0] if result else 0
    
    def get_aggregated_stats(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        direction: Optional[str] = None,
        qr_code_value: Optional[int] = None,
        is_lost: Optional[bool] = None,
    ) -> Dict:
        """
        Get aggregated statistics for container events.
        
        Args:
            start_time: ISO 8601 start time
            end_time: ISO 8601 end time
            
        Returns:
            Dictionary with aggregated statistics
        """
        # Base filter
        base_filter = ""
        params = []
        
        if start_time:
            base_filter += " AND timestamp >= ?"
            params.append(start_time)
        
        if end_time:
            base_filter += " AND timestamp <= ?"
            params.append(end_time)

        if direction:
            base_filter += " AND direction = ?"
            params.append(direction)

        if qr_code_value:
            base_filter += " AND qr_code_value = ?"
            params.append(qr_code_value)

        if is_lost is not None:
            base_filter += " AND COALESCE(is_lost, 0) = ?"
            params.append(1 if is_lost else 0)
        
        # Get direction counts
        direction_query = f"""
            SELECT direction, COUNT(*) as count
            FROM container_events
            WHERE 1=1 {base_filter}
            GROUP BY direction
        """
        direction_rows = self.db.fetchall(direction_query, params)
        
        direction_counts = {'positive': 0, 'negative': 0, 'unknown': 0}
        for row in direction_rows:
            direction_counts[row[0]] = row[1]
        
        # Get per-QR breakdown
        qr_query = f"""
            SELECT qr_code_value, direction, COUNT(*) as count
            FROM container_events
            WHERE 1=1 {base_filter}
            GROUP BY qr_code_value, direction
        """
        qr_rows = self.db.fetchall(qr_query, params)
        
        qr_breakdown = {i: {'positive': 0, 'negative': 0, 'unknown': 0} for i in range(1, 6)}
        for row in qr_rows:
            qr_value, direction, count = row
            if qr_value in qr_breakdown:
                qr_breakdown[qr_value][direction] = count
        
        # Get time boundaries
        time_query = f"""
            SELECT MIN(timestamp), MAX(timestamp)
            FROM container_events
            WHERE 1=1 {base_filter}
        """
        time_row = self.db.fetchone(time_query, params)
        first_event = time_row[0] if time_row else None
        last_event = time_row[1] if time_row else None
        
        # Calculate totals
        total_positive = direction_counts['positive']
        total_negative = direction_counts['negative']
        total_unknown = direction_counts['unknown']
        total = total_positive + total_negative + total_unknown
        mismatch = total_positive - total_negative
        
        # Get lost count
        lost_query = f"""
            SELECT COUNT(*) FROM container_events
            WHERE COALESCE(is_lost, 0) = 1 {base_filter}
        """
        lost_row = self.db.fetchone(lost_query, params)
        total_lost = lost_row[0] if lost_row else 0
        
        return {
            'total': total,
            'total_positive': total_positive,
            'total_negative': total_negative,
            'total_unknown': total_unknown,
            'total_lost': total_lost,
            'mismatch': mismatch,
            'mismatch_percent': (
                abs(mismatch) / max(total_positive, 1) * 100
                if total_positive > 0 else 0
            ),
            'qr_breakdown': qr_breakdown,
            'first_event': first_event,
            'last_event': last_event,
            'time_range': {
                'start': start_time,
                'end': end_time,
            },
        }
    
    def get_recent_events(self, limit: int = 10) -> List[Dict]:
        """Get most recent container events."""
        return self.get_events(limit=limit)
    
    def get_hourly_stats(
        self,
        start_time: str,
        end_time: str,
    ) -> List[Dict]:
        """
        Get hourly statistics for timeline charts.
        
        Args:
            start_time: ISO 8601 start time
            end_time: ISO 8601 end time
            
        Returns:
            List of hourly stat dictionaries
        """
        query = """
            SELECT 
                strftime('%Y-%m-%d %H:00:00', timestamp) as hour,
                direction,
                COUNT(*) as count
            FROM container_events
            WHERE timestamp >= ? AND timestamp <= ?
            GROUP BY hour, direction
            ORDER BY hour
        """
        rows = self.db.fetchall(query, (start_time, end_time))
        
        # Group by hour
        hourly_data = {}
        for row in rows:
            hour, direction, count = row
            if hour not in hourly_data:
                hourly_data[hour] = {'hour': hour, 'positive': 0, 'negative': 0, 'unknown': 0}
            hourly_data[hour][direction] = count
        
        return list(hourly_data.values())
    
    def get_mismatch_alerts(
        self,
        threshold: int = 5,
        window_hours: float = 1.0,
    ) -> List[Dict]:
        """
        Check for direction mismatches in recent time windows.
        
        A mismatch occurs when positive and negative counts differ
        significantly, indicating containers that haven't returned.
        
        Args:
            threshold: Minimum difference to trigger alert
            window_hours: Time window in hours to check
            
        Returns:
            List of mismatch alert dictionaries
        """
        # Get stats for recent window
        end_time = datetime.now().isoformat()
        start_time = (datetime.now() - timedelta(hours=window_hours)).isoformat()
        
        stats = self.get_aggregated_stats(start_time, end_time)
        
        alerts = []
        
        # Check overall mismatch
        if abs(stats['mismatch']) > threshold:
            alerts.append({
                'type': 'overall_mismatch',
                'severity': 'warning' if abs(stats['mismatch']) < threshold * 2 else 'error',
                'message': f"Container direction mismatch: {stats['mismatch']} difference",
                'positive': stats['total_positive'],
                'negative': stats['total_negative'],
                'delta': stats['mismatch'],
                'window_hours': window_hours,
                'timestamp': datetime.now().isoformat(),
            })
        
        # Check per-QR mismatches
        for qr_value, counts in stats['qr_breakdown'].items():
            qr_mismatch = counts['positive'] - counts['negative']
            if abs(qr_mismatch) > threshold // 2:  # Stricter per-QR threshold
                alerts.append({
                    'type': 'qr_mismatch',
                    'severity': 'warning',
                    'message': f"Container {qr_value} mismatch: {qr_mismatch}",
                    'qr_value': qr_value,
                    'positive': counts['positive'],
                    'negative': counts['negative'],
                    'delta': qr_mismatch,
                    'window_hours': window_hours,
                    'timestamp': datetime.now().isoformat(),
                })
        
        return alerts
    
    def insert_event(
        self,
        timestamp: str,
        qr_code_value: int,
        direction: str,
        track_id: Optional[int] = None,
        entry_y: Optional[int] = None,
        exit_y: Optional[int] = None,
        duration_seconds: Optional[float] = None,
        snapshot_path: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> int:
        """
        Insert a new container event.
        
        Returns:
            The ID of the inserted event
        """
        query = """
            INSERT INTO container_events (
                timestamp, qr_code_value, direction, track_id,
                entry_y, exit_y, duration_seconds, snapshot_path, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        metadata_json = None
        if metadata:
            try:
                metadata_json = json.dumps(metadata)
            except Exception:
                logger.warning("[ContainerRepository] failed to serialize metadata for insert; storing null")
                metadata_json = None
        
        cursor = self.db.execute(
            query,
            (
                timestamp, qr_code_value, direction, track_id,
                entry_y, exit_y, duration_seconds, snapshot_path, metadata_json
            )
        )
        
        return cursor.lastrowid
    
    def get_event_by_id(self, event_id: int) -> Optional[Dict]:
        """Get a single event by ID."""
        query = """
            SELECT 
                id, timestamp, qr_code_value, direction, track_id,
                entry_y, exit_y, duration_seconds, snapshot_path, metadata
            FROM container_events
            WHERE id = ?
        """
        row = self.db.fetchone(query, (event_id,))
        
        if not row:
            return None
        
        return {
            'id': row[0],
            'timestamp': row[1],
            'qr_code_value': row[2],
            'direction': row[3],
            'track_id': row[4],
            'entry_x': row[5],
            'exit_x': row[6],
            'duration_seconds': row[7],
            'snapshot_path': row[8],
            'metadata': self._safe_load_metadata(row[9]),
        }
