"""Analytics Service - Refactored with Repository Pattern."""
import json
from collections import defaultdict
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
        
        # Build per-class confidence insights (smoothed counts, confusion pairs)
        # Build a name→arabic_name lookup so confusion detail shows Arabic labels.
        # If arabic_name in DB still equals the English name (was never translated),
        # fall back to the canonical Arabic translations from the seed data.
        _CANONICAL_ARABIC = {
            'Rejected':      'غير واضحة',
            'Unknown':       'غير معروف',
            'Brown_Orange':  'عربي',
            'Red_Yellow':    '11 رغيف',
            'Blue_Yellow':   '12 رغيف',
            'Green_Yellow':  '14 رغيف',
            'Bran':          'نخالة',
            'Black_Orange':  'عشرات',
            'Purple_Yellow': 'شاورما',
            'Wheatberry':    'قمحة',
        }
        name_to_arabic = {}
        for bt in all_bag_types:
            eng = bt['name']
            arabic = bt.get('arabic_name') or ''
            # Use DB value only if it's a real translation (differs from English name)
            if arabic and arabic != eng:
                name_to_arabic[eng] = arabic
            else:
                # Fall back to canonical map, then English name as last resort
                name_to_arabic[eng] = _CANONICAL_ARABIC.get(eng, eng)
        confidence_insights = self._build_confidence_insights(ordered_events, name_to_arabic)
        for cls in classifications:
            cid = cls['id']
            insight = confidence_insights.get(cid, {})
            cls['smoothed_count'] = insight.get('smoothed_count', 0)
            cls['confusion_pairs'] = insight.get('confusion_pairs', [])
            cls['low_conf_details'] = insight.get('low_conf_details', [])

        # Build batch anomaly analysis
        batch_anomalies = self._build_batch_anomalies(runs, ordered_events)

        data = {
            "meta": {
                "start": start_time, 
                "end": end_time, 
                "request_time": datetime.now().strftime("%Y/%m/%d - %H:%M:%S")
            }, 
            "data": {
                'total': stats['total'], 
                'classifications': classifications,
                'batch_anomalies': batch_anomalies
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
    def _build_confidence_insights(
        self,
        ordered_events: List[Dict],
        name_to_arabic: Dict[str, str] = None,
    ) -> Dict[int, Dict]:
        """
        Parse event metadata to build per-class confidence insights.

        For each bag type, extracts:
        - smoothed_count: How many items were corrected by the smoother
        - confusion_pairs: List of dicts showing what the model originally
          predicted before smoothing overrode it (with arabic_name resolved)
        - low_conf_details: List of dicts for low-confidence items (conf < 0.8)
          that were NOT smoothed — shows natural confusion (with arabic_name)

        Args:
            ordered_events: Time-sorted list of event dicts with 'metadata' JSON
            name_to_arabic: Optional mapping of English class name → Arabic name

        Returns:
            Dict keyed by bag_type_id with insight dicts
        """
        if name_to_arabic is None:
            name_to_arabic = {}

        insights: Dict[int, Dict] = {}

        for ev in ordered_events:
            bid = ev['bag_type_id']
            if bid not in insights:
                insights[bid] = {
                    'smoothed_count': 0,
                    '_confusion_map': defaultdict(int),
                    '_low_conf_map': defaultdict(int),
                }

            meta_raw = ev.get('metadata')
            if not meta_raw:
                continue

            try:
                meta = json.loads(meta_raw) if isinstance(meta_raw, str) else meta_raw
            except (json.JSONDecodeError, TypeError):
                continue

            is_smoothed = meta.get('smoothed', False)
            original_class = meta.get('original_class')
            confidence = ev.get('confidence', 1.0)

            if is_smoothed and original_class:
                insights[bid]['smoothed_count'] += 1
                insights[bid]['_confusion_map'][original_class] += 1
            elif confidence < 0.8 and original_class and original_class != ev.get('bag_type', ''):
                # Low confidence but not smoothed — model was uncertain
                insights[bid]['_low_conf_map'][original_class] += 1

        # Convert internal maps to sorted lists for template rendering
        for bid, data in insights.items():
            confusion_sorted = sorted(
                data['_confusion_map'].items(), key=lambda x: x[1], reverse=True
            )
            data['confusion_pairs'] = [
                {
                    'class_name': cls,
                    'arabic_name': name_to_arabic.get(cls, cls),
                    'count': cnt,
                }
                for cls, cnt in confusion_sorted
            ]

            low_conf_sorted = sorted(
                data['_low_conf_map'].items(), key=lambda x: x[1], reverse=True
            )
            data['low_conf_details'] = [
                {
                    'class_name': cls,
                    'arabic_name': name_to_arabic.get(cls, cls),
                    'count': cnt,
                }
                for cls, cnt in low_conf_sorted
            ]

            del data['_confusion_map']
            del data['_low_conf_map']

        return insights

    def _build_batch_anomalies(
        self,
        runs: List[Dict],
        ordered_events: List[Dict]
    ) -> Dict[str, Any]:
        """
        Analyze batch discipline from production runs.

        Detects anomalies that indicate workers are not following batch
        processing guidelines:
        - short_batches: Runs with very few items (< noise_threshold)
        - transition_count: Total number of batch changes
        - rapid_switches: Adjacent runs shorter than expected
        - total_smoothed: Total items corrected across all classes
        - smoothing_rate: Percentage of all items that were corrected

        Args:
            runs: Consecutive runs from build_consecutive_runs (noise already filtered)
            ordered_events: All events for smoothing stats

        Returns:
            Dict with anomaly summary
        """
        # Filter out the NOISE pseudo-run
        real_runs = [r for r in runs if r.get('bag_type_id') != 'NOISE']

        # Count transitions (number of batch changes)
        transition_count = max(0, len(real_runs) - 1)

        # Detect short batches (batches with count < 2× noise_threshold)
        short_threshold = self.config.noise_threshold * 2
        short_batches = []
        for i, run in enumerate(real_runs):
            if run['count'] < short_threshold:
                short_batches.append({
                    'class_name': run.get('arabic_name') or run.get('class_name', '?'),
                    'count': run['count'],
                    'time': run.get('start', '')[:16] if run.get('start') else '',
                    'position': i + 1,
                })

        # Detect rapid switches: consecutive short runs (both < short_threshold)
        rapid_switches = 0
        for i in range(len(real_runs) - 1):
            if (real_runs[i]['count'] < short_threshold and
                    real_runs[i + 1]['count'] < short_threshold):
                rapid_switches += 1

        # Total smoothed items from metadata
        total_smoothed = 0
        total_events = len(ordered_events)
        for ev in ordered_events:
            meta_raw = ev.get('metadata')
            if not meta_raw:
                continue
            try:
                meta = json.loads(meta_raw) if isinstance(meta_raw, str) else meta_raw
            except (json.JSONDecodeError, TypeError):
                continue
            if meta.get('smoothed', False):
                total_smoothed += 1

        smoothing_rate = (
            round(total_smoothed / total_events * 100, 1)
            if total_events > 0 else 0.0
        )

        # Severity assessment
        # Green: 0-2 transitions & no rapid switches
        # Yellow: 3-5 transitions or some short batches
        # Red: >5 transitions or rapid switches or smoothing_rate > 15%
        if rapid_switches > 0 or transition_count > 5 or smoothing_rate > 15:
            severity = 'high'
        elif transition_count > 2 or len(short_batches) > 2:
            severity = 'medium'
        else:
            severity = 'low'

        return {
            'transition_count': transition_count,
            'batch_count': len(real_runs),
            'short_batches': short_batches,
            'short_batch_count': len(short_batches),
            'rapid_switches': rapid_switches,
            'total_smoothed': total_smoothed,
            'smoothing_rate': smoothing_rate,
            'severity': severity,
        }

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
