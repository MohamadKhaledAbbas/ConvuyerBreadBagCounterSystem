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
        time_now = datetime.now() + timedelta(hours=self.config.timezone_offset_hours)
        if time_now.hour >= self.config.shift_start_hour:
            start_time = time_now
            end_time = time_now + timedelta(days=1)
        else:
            start_time = time_now - timedelta(days=1)
            end_time = time_now
        start_time = start_time.replace(hour=self.config.shift_start_hour, minute=0, second=0, microsecond=0)
        end_time = end_time.replace(hour=self.config.shift_end_hour, minute=0, second=0, microsecond=0)
        return start_time, end_time
    def get_analytics_data(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        db_start = start_time - timedelta(hours=self.config.timezone_offset_hours)
        db_end = end_time - timedelta(hours=self.config.timezone_offset_hours)
        logger.info(f"[Analytics] Query range: {db_start.isoformat()} to {db_end.isoformat()}")
        repo_data = self.repo.get_time_range_analytics(db_start, db_end)
        stats = repo_data['stats']
        events = repo_data['events']
        per_class_windows = repo_data['per_class_windows']
        classifications = [{'id': bid, 'name': d['name'], 'arabic_name': d['arabic_name'], 'count': d['count'], 'high_count': d['high_count'], 'low_count': d['low_count'], 'thumb': d['thumb'], 'weight': d['weight']} for bid, d in stats['by_type'].items()]
        ordered_events = sorted(events, key=lambda x: x['timestamp'])
        runs = self.build_consecutive_runs(ordered_events)
        data = {"meta": {"start": start_time, "end": end_time, "request_time": datetime.now().strftime("%Y/%m/%d - %H:%M:%S")}, "data": {'total': stats['total'], 'classifications': classifications}, "timeline": {"ordered_events": ordered_events, "per_class_windows": per_class_windows, "runs": runs}}
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
