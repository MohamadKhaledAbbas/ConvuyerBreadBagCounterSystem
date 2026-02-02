"""
Analytics Service - Business Logic Layer.

Handles:
- Time parsing and validation
- Production run analysis
- Noise filtering
- Data aggregation
- Image path normalization

Separates business logic from HTTP handling for better testability and maintainability.
"""

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from fastapi import HTTPException

from src.logging.Database import DatabaseManager
from src.utils.AppLogging import logger


class AnalyticsService:
    """
    Service class for analytics business logic.

    Provides methods for data processing, run analysis, and aggregation
    without coupling to HTTP/FastAPI specifics.
    """

    # Timezone offset for production environment (hours)
    TIMEZONE_OFFSET = 3

    # Minimum bags in a run to be considered valid (not noise)
    NOISE_THRESHOLD = 10

    # Daily shift times
    SHIFT_START_HOUR = 16  # 4 PM
    SHIFT_END_HOUR = 14    # 2 PM next day

    def __init__(self, db: DatabaseManager):
        """
        Initialize analytics service.

        Args:
            db: Database manager instance
        """
        self.db = db

    @staticmethod
    def parse_datetime(val: Optional[str]) -> Optional[datetime]:
        """
        Parse datetime string in multiple formats.

        Supports:
        - ISO format: "2026-02-01T16:00:00"
        - Short format: "2026-02-01T16:00"
        - Space separator: "2026-02-01 16:00:00"

        Args:
            val: Datetime string to parse

        Returns:
            datetime object or None if val is None

        Raises:
            HTTPException: If format is invalid
        """
        if not val:
            return None

        # Normalize format (space to T)
        val = val.replace(' ', 'T').strip()

        # Pad if needed
        if len(val) == 16:  # YYYY-MM-DDTHH:MM
            val += ":00"
        elif len(val) == 19:  # YYYY-MM-DDTHH:MM:SS
            pass
        else:
            logger.error(f"[Analytics] Invalid time format length: {val}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid time format: {val}. Expected ISO format (YYYY-MM-DDTHH:MM:SS)"
            )

        try:
            return datetime.fromisoformat(val)
        except ValueError as e:
            logger.error(f"[Analytics] Failed to parse datetime '{val}': {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid datetime: {val}. Use ISO format (YYYY-MM-DDTHH:MM:SS)"
            )

    @staticmethod
    def calculate_daily_shift_times() -> tuple[datetime, datetime]:
        """
        Calculate daily shift times based on current time.

        Logic:
        - If current time is 16:00-23:59: today 16:00 to tomorrow 14:00
        - If current time is 00:00-15:59: yesterday 16:00 to today 14:00

        Applies timezone offset for production environment.

        Returns:
            tuple: (start_datetime, end_datetime)
        """
        # Apply timezone offset
        time_now = datetime.now() + timedelta(hours=AnalyticsService.TIMEZONE_OFFSET)

        # Determine shift boundaries
        if time_now.hour in range(AnalyticsService.SHIFT_START_HOUR, 24):
            # Current shift (today 16:00 to tomorrow 14:00)
            start_time = time_now
            end_time = time_now + timedelta(days=1)
        else:
            # Previous shift (yesterday 16:00 to today 14:00)
            start_time = time_now - timedelta(days=1)
            end_time = time_now

        # Set exact times
        start_time = start_time.replace(
            hour=AnalyticsService.SHIFT_START_HOUR,
            minute=0,
            second=0,
            microsecond=0
        )
        end_time = end_time.replace(
            hour=AnalyticsService.SHIFT_END_HOUR,
            minute=0,
            second=0,
            microsecond=0
        )

        logger.debug(f"[Analytics] Calculated shift times: {start_time} to {end_time}")
        return start_time, end_time

    def build_consecutive_runs(self, ordered_events: List[Dict]) -> List[Dict]:
        """
        Build consecutive production runs from ordered events.

        Groups consecutive events of same bag type into runs.
        Applies noise filtering and merges adjacent runs of same type.

        Args:
            ordered_events: List of events ordered by timestamp

        Returns:
            List of production runs with metadata
        """
        if not ordered_events:
            return []

        runs = []
        noise_count = 0
        noise_start = None
        noise_end = None
        current = None

        # Step 1: Group consecutive events by bag_type_id
        for ev in ordered_events:
            bag_type_id = ev.get("bag_type_id")

            if current is None or bag_type_id != current["bag_type_id"]:
                # Finalize previous run
                if current:
                    if current["count"] >= self.NOISE_THRESHOLD:
                        runs.append(current)
                    else:
                        # Accumulate noise
                        noise_count += current["count"]
                        if noise_start is None or current["start"] < noise_start:
                            noise_start = current["start"]
                        if noise_end is None or current["end"] > noise_end:
                            noise_end = current["end"]

                # Start new run
                current = {
                    "bag_type_id": bag_type_id,
                    "class_name": ev.get("class_name", "Unknown"),
                    "arabic_name": ev.get("arabic_name", "غير معروف"),
                    "thumb": ev.get("thumb", ""),
                    "weight": ev.get("weight") or 0,
                    "start": ev.get("timestamp"),
                    "end": ev.get("timestamp"),
                    "count": 1,
                }
            else:
                # Continue current run
                current["end"] = ev.get("timestamp")
                current["count"] += 1

        # Handle last run
        if current:
            if current["count"] >= self.NOISE_THRESHOLD:
                runs.append(current)
            else:
                noise_count += current["count"]
                if noise_start is None or current["start"] < noise_start:
                    noise_start = current["start"]
                if noise_end is None or current["end"] > noise_end:
                    noise_end = current["end"]

        # Step 2: Merge adjacent runs of same type
        merged = []
        last = None
        for run in runs:
            if last is None:
                last = run
            elif run["bag_type_id"] == last["bag_type_id"]:
                # Merge into last run
                last["end"] = run["end"]
                last["count"] += run["count"]
            else:
                merged.append(last)
                last = run

        if last:
            merged.append(last)

        # Step 3: Add noise run if exists
        if noise_count > 0:
            merged.append({
                "bag_type_id": "NOISE",
                "class_name": "Noise",
                "arabic_name": "تصنيفات مفلترة غير دقيقة",
                "thumb": "",
                "weight": 0,
                "start": noise_start,
                "end": noise_end,
                "count": noise_count,
            })

        logger.info(f"[Analytics] Built {len(merged)} runs ({noise_count} noise bags)")
        return merged

    @staticmethod
    def normalize_image_paths(data: Dict[str, Any]) -> None:
        """
        Normalize image paths for web serving.

        Replaces file system paths with URL paths:
        - data/classes/ -> known_classes/
        - data/unknown/ -> unknown_classes/

        Modifies data in-place.

        Args:
            data: Dictionary containing image paths (thumb fields)
        """
        def fix_path(thumb: Any) -> str:
            """Fix a single thumb path."""
            if not isinstance(thumb, str):
                return ""
            return (thumb.replace("data/classes/", "known_classes/")
                        .replace("data/unknown/", "unknown_classes/"))

        # Fix classifications
        for classification in data.get("data", {}).get("classifications", []):
            classification["thumb"] = fix_path(classification.get("thumb"))

        # Fix timeline events
        for event in data.get("timeline", {}).get("ordered_events", []):
            event["thumb"] = fix_path(event.get("thumb"))

        # Fix per-class windows
        for class_data in data.get("timeline", {}).get("per_class_windows", {}).values():
            class_data["thumb"] = fix_path(class_data.get("thumb"))

        # Fix runs
        for run in data.get("timeline", {}).get("runs", []):
            run["thumb"] = fix_path(run.get("thumb"))

    def get_analytics_data(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """
        Get complete analytics data for a time range.

        Aggregates:
        - Statistical summary
        - Timeline of events
        - Per-class time windows
        - Consecutive production runs

        Args:
            start_time: Start of analysis period
            end_time: End of analysis period

        Returns:
            Dictionary with analytics data matching V1 format
        """
        # Apply timezone offset for database query
        db_start = start_time - timedelta(hours=self.TIMEZONE_OFFSET)
        db_end = end_time - timedelta(hours=self.TIMEZONE_OFFSET)

        logger.info(f"[Analytics] Fetching data: {db_start.isoformat()} to {db_end.isoformat()}")

        # Get events from database (basic V2 method)
        events = self.db.get_events(
            start_date=db_start.isoformat(),
            end_date=db_end.isoformat(),
            limit=10000  # Reasonable limit
        )

        logger.info(f"[Analytics] Retrieved {len(events)} events")

        # Build aggregated statistics from events
        stats = self._build_aggregated_stats(events)

        # Sort events by timestamp for timeline
        ordered_events = sorted(events, key=lambda x: x.get('timestamp', ''))

        # Build per-class time windows
        per_class_windows = self._build_per_class_windows(events)

        # Build consecutive runs with proper structure for V1 template
        runs = self._build_runs_with_v1_structure(ordered_events)

        data = {
            "meta": {
                "start": start_time,  # Display times (with offset applied)
                "end": end_time,
                "request_time": datetime.now().strftime("%Y/%m/%d - %H:%M:%S")
            },
            "data": stats,
            "timeline": {
                "ordered_events": self._prepare_timeline_events(ordered_events),
                "per_class_windows": per_class_windows,
                "runs": runs
            }
        }

        # Normalize image paths for web serving
        self.normalize_image_paths(data)

        logger.info(
            f"[Analytics] Data ready: {stats.get('total', {}).get('count', 0)} bags, "
            f"{len(stats.get('classifications', []))} classes, {len(runs)} runs"
        )

        return data

    def _build_aggregated_stats(self, events: List[Dict]) -> Dict[str, Any]:
        """
        Build aggregated statistics from raw events.

        Matches V1 format expected by template.

        Args:
            events: List of event dictionaries

        Returns:
            Statistics dictionary with total and classifications
        """
        from collections import defaultdict

        total_count = len(events)
        high_count = sum(1 for e in events if e.get('confidence', 0) >= 0.8)
        low_count = total_count - high_count

        # Group by bag_type
        by_type = defaultdict(lambda: {'count': 0, 'high_count': 0, 'low_count': 0})

        for event in events:
            bag_type = event.get('bag_type', 'Unknown')
            by_type[bag_type]['count'] += 1
            if event.get('confidence', 0) >= 0.8:
                by_type[bag_type]['high_count'] += 1
            else:
                by_type[bag_type]['low_count'] += 1

        # Build classifications list (for template)
        classifications = []
        for bag_type, counts in sorted(by_type.items()):
            classifications.append({
                'id': bag_type,  # Use bag_type as ID
                'name': bag_type,
                'arabic_name': bag_type,  # Fallback to English name
                'count': counts['count'],
                'high_count': counts['high_count'],
                'low_count': counts['low_count'],
                'thumb': event.get('image_path', ''),  # Will be normalized later
                'weight': 0  # Default weight
            })

        return {
            'total': {
                'count': total_count,
                'high_count': high_count,
                'low_count': low_count,
                'weight': 0  # Calculate if weight data available
            },
            'classifications': classifications
        }

    def _build_per_class_windows(self, events: List[Dict]) -> Dict[str, Any]:
        """
        Build per-class time windows showing first/last seen.

        Args:
            events: List of event dictionaries

        Returns:
            Dictionary of {bag_type: {first_seen, last_seen, event_count}}
        """
        from collections import defaultdict

        windows = defaultdict(lambda: {
            'first_seen': None,
            'last_seen': None,
            'event_count': 0,
            'thumb': ''
        })

        for event in events:
            bag_type = event.get('bag_type', 'Unknown')
            timestamp = event.get('timestamp', '')

            if not windows[bag_type]['first_seen'] or timestamp < windows[bag_type]['first_seen']:
                windows[bag_type]['first_seen'] = timestamp

            if not windows[bag_type]['last_seen'] or timestamp > windows[bag_type]['last_seen']:
                windows[bag_type]['last_seen'] = timestamp
                windows[bag_type]['thumb'] = event.get('image_path', '')

            windows[bag_type]['event_count'] += 1

        return dict(windows)

    def _prepare_timeline_events(self, ordered_events: List[Dict]) -> List[Dict]:
        """
        Prepare timeline events with required fields for template.

        Args:
            ordered_events: Sorted list of events

        Returns:
            List of events with standardized fields
        """
        timeline_events = []
        for event in ordered_events:
            timeline_events.append({
                'bag_type_id': event.get('bag_type', 'Unknown'),
                'class_name': event.get('bag_type', 'Unknown'),
                'arabic_name': event.get('bag_type', 'Unknown'),
                'timestamp': event.get('timestamp', ''),
                'thumb': event.get('image_path', ''),
                'weight': 0,
                'confidence': event.get('confidence', 0)
            })
        return timeline_events

    def _build_runs_with_v1_structure(self, ordered_events: List[Dict]) -> List[Dict]:
        """
        Build runs with V1-compatible structure for template.

        Args:
            ordered_events: Sorted list of events

        Returns:
            List of runs with V1 fields (bag_type_id, start, end, etc.)
        """
        # Prepare events for run building
        timeline_events = self._prepare_timeline_events(ordered_events)

        # Build runs using existing logic
        runs = self.build_consecutive_runs(timeline_events)

        return runs
