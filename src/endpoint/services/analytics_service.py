"""Analytics Service - Refactored with Repository Pattern."""
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from fastapi import HTTPException

from src.config.config_manager import get_config
from src.endpoint.repositories.analytics_repository import AnalyticsRepository
from src.utils.AppLogging import logger


class AnalyticsService:
    def __init__(self, repository: AnalyticsRepository):
        self.repo = repository
        self.config = get_config()
    @staticmethod
    def parse_datetime(val: Optional[str]) -> Optional[datetime]:
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
    def calculate_daily_shift_times(self) -> tuple:
        """
        Calculate start and end times for the current daily shift.

        Note: Events are stored in LOCAL system time (not UTC).
        When system timezone matches production timezone, no offset is needed.
        The timezone_offset_hours config is for systems running in UTC.

        Handles shifts that span midnight (e.g., 16:00-14:00 next day).
        """
        # Use local system time directly (events are stored in local time)
        time_now = datetime.now()

        # Check if shift spans midnight (start hour > end hour)
        shift_spans_midnight = self.config.shift_start_hour > self.config.shift_end_hour

        if shift_spans_midnight:
            # For shifts spanning midnight (e.g., 16:00-14:00)
            if time_now.hour >= self.config.shift_start_hour:
                # Currently after shift start (e.g., 16:00-23:59)
                # Shift: Today 16:00 to Tomorrow 14:00
                start_time = time_now.replace(hour=self.config.shift_start_hour, minute=0, second=0, microsecond=0)
                end_time = (time_now + timedelta(days=1)).replace(hour=self.config.shift_end_hour, minute=0, second=0, microsecond=0)
            elif time_now.hour < self.config.shift_end_hour:
                # Currently before shift end (e.g., 00:00-13:59)
                # Still in yesterday's shift: Yesterday 16:00 to Today 14:00
                start_time = (time_now - timedelta(days=1)).replace(hour=self.config.shift_start_hour, minute=0, second=0, microsecond=0)
                end_time = time_now.replace(hour=self.config.shift_end_hour, minute=0, second=0, microsecond=0)
            else:
                # Currently between shift end and shift start (e.g., 14:00-15:59)
                # This is a "dead zone" - show the just-completed shift
                start_time = (time_now - timedelta(days=1)).replace(hour=self.config.shift_start_hour, minute=0, second=0, microsecond=0)
                end_time = time_now.replace(hour=self.config.shift_end_hour, minute=0, second=0, microsecond=0)
        else:
            # For shifts within same day (e.g., 08:00-16:00)
            if time_now.hour >= self.config.shift_start_hour:
                start_time = time_now.replace(hour=self.config.shift_start_hour, minute=0, second=0, microsecond=0)
                end_time = time_now.replace(hour=self.config.shift_end_hour, minute=0, second=0, microsecond=0)
            else:
                # Before shift start - show previous day's shift
                start_time = (time_now - timedelta(days=1)).replace(hour=self.config.shift_start_hour, minute=0, second=0, microsecond=0)
                end_time = (time_now - timedelta(days=1)).replace(hour=self.config.shift_end_hour, minute=0, second=0, microsecond=0)

        return start_time, end_time
    def get_analytics_data(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        # Events are stored in local system time, so use times directly without offset
        db_start = start_time
        db_end = end_time
        logger.info(f"[Analytics] Query range: {db_start.isoformat()} to {db_end.isoformat()}")
        repo_data = self.repo.get_time_range_analytics(db_start, db_end)
        stats = repo_data['stats']
        events = repo_data['events']
        per_class_windows = repo_data['per_class_windows']

        # Get ALL bag types from database (including those with zero events)
        all_bag_types = self.repo.get_all_bag_types()

        # Build classifications list including ALL bag types
        # Merge stats with all bag types - bag types with events get their counts,
        # those without events get zero counts
        classifications = []
        for bag_type in all_bag_types:
            bid = bag_type['id']
            if bid in stats['by_type']:
                # Has events - use stats data
                d = stats['by_type'][bid]
                classifications.append({
                    'id': bid,
                    'name': d['name'],
                    'arabic_name': d['arabic_name'],
                    'count': d['count'],
                    'high_count': d['high_count'],
                    'low_count': d['low_count'],
                    'thumb': d['thumb'],
                    'weight': d['weight']
                })
            else:
                # No events - create entry with zero counts
                classifications.append({
                    'id': bid,
                    'name': bag_type['name'],
                    'arabic_name': bag_type.get('arabic_name', bag_type['name']),
                    'count': 0,
                    'high_count': 0,
                    'low_count': 0,
                    'thumb': bag_type.get('thumb', ''),
                    'weight': bag_type.get('weight', 0)
                })

        # Sort by count descending for better UX (most common first)
        classifications.sort(key=lambda x: x['count'], reverse=True)
        
        ordered_events = sorted(events, key=lambda x: x['timestamp'])
        runs = self.build_consecutive_runs(ordered_events)
        
        # Pre-group runs by bag_type_id for efficient template rendering
        runs_by_type = self._group_runs_by_type(runs)
        
        data = {
            "meta": {
                "start": start_time, 
                "end": end_time, 
                "request_time": datetime.now().strftime("%Y/%m/%d - %H:%M:%S")
            }, 
            "data": {
                'total': stats['total'], 
                'classifications': classifications
            }, 
            "timeline": {
                "ordered_events": ordered_events, 
                "per_class_windows": per_class_windows, 
                "runs": runs,
                "runs_by_type": runs_by_type  # Pre-grouped for template efficiency
            }
        }
        
        self.normalize_image_paths(data)
        logger.info(f"[Analytics] Data ready: {stats['total']['count']} bags, {len(classifications)} classes, {len(runs)} runs")
        return data
    def build_consecutive_runs(self, ordered_events: List[Dict]) -> List[Dict]:
        if not ordered_events:
            return []
        runs = []
        noise_count = 0
        noise_start = None
        noise_end = None
        current = None
        for ev in ordered_events:
            bag_type_id = ev["bag_type_id"]
            if current is None or bag_type_id != current["bag_type_id"]:
                if current:
                    current_count = current.get("count", 0)
                    if current_count >= self.config.noise_threshold:
                        runs.append(current)
                    else:
                        noise_count += current_count
                        if noise_start is None or current["start"] < noise_start:
                            noise_start = current["start"]
                        if noise_end is None or current["end"] > noise_end:
                            noise_end = current["end"]
                current = {"bag_type_id": bag_type_id, "class_name": ev["bag_type"], "arabic_name": ev["arabic_name"], "thumb": ev["thumb"], "weight": ev["weight"], "start": ev["timestamp"], "end": ev["timestamp"], "count": 1}
            else:
                current["end"] = ev["timestamp"]
                current["count"] = current.get("count", 0) + 1
        if current:
            current_count = current.get("count", 0)
            if current_count >= self.config.noise_threshold:
                runs.append(current)
            else:
                noise_count += current_count
        merged = []
        last = None
        for run in runs:
            if last and run["bag_type_id"] == last["bag_type_id"]:
                last["end"] = run["end"]
                last["count"] += run["count"]
            else:
                if last:
                    merged.append(last)
                last = run
        if last:
            merged.append(last)
        if noise_count > 0:
            merged.append({"bag_type_id": "NOISE", "class_name": "Noise", "arabic_name": "تصنيفات مفلترة غير دقيقة", "thumb": "", "weight": 0, "start": noise_start, "end": noise_end, "count": noise_count})
        return merged
    def normalize_image_paths(self, data: Dict[str, Any]) -> None:
        """
        Normalize image paths for web serving.

        Converts filesystem paths (data/classes/...) to web paths (known_classes/...)
        Only needed for thumb images from bag_types table.
        """
        def fix_path(path: str) -> str:
            if not path:
                return ""
            # Convert filesystem paths to web-accessible paths
            return (path
                    .replace(self.config.known_classes_dir + "/", self.config.web_known_classes_path + "/")
                    .replace(self.config.unknown_classes_dir + "/", self.config.web_unknown_classes_path + "/"))

        # Fix thumb paths in classifications (from bag_types)
        for cls in data.get("data", {}).get("classifications", []):
            cls["thumb"] = fix_path(cls.get("thumb", ""))

        # Fix thumb paths in events (from bag_types via JOIN)
        for event in data.get("timeline", {}).get("ordered_events", []):
            event["thumb"] = fix_path(event.get("thumb", ""))

        # Fix thumb paths in per_class_windows (from bag_types)
        for window in data.get("timeline", {}).get("per_class_windows", {}).values():
            window["thumb"] = fix_path(window.get("thumb", ""))

        # Fix thumb paths in runs (from bag_types)
        for run in data.get("timeline", {}).get("runs", []):
            run["thumb"] = fix_path(run.get("thumb", ""))
        
        # Fix thumb paths in runs_by_type (from bag_types)
        for runs_list in data.get("timeline", {}).get("runs_by_type", {}).values():
            for run in runs_list:
                run["thumb"] = fix_path(run.get("thumb", ""))
    
    def _group_runs_by_type(self, runs: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Pre-group runs by bag_type_id for efficient template rendering.
        
        This avoids N+1 filtering in the template (selectattr for each classification).
        
        Args:
            runs: List of consecutive runs
            
        Returns:
            Dictionary mapping bag_type_id to list of runs
        """
        grouped = {}
        for run in runs:
            bag_type_id = run['bag_type_id']
            if bag_type_id not in grouped:
                grouped[bag_type_id] = []
            grouped[bag_type_id].append(run)
        return grouped
