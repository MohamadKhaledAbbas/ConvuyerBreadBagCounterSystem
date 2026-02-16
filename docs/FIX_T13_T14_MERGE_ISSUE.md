# Fix: T13 and T14 Merge Issue - Adaptive Travel Duration Validation

## Problem Summary

**Issue**: Two separate bags (T13 and T14) were detected and tracked correctly, but T13 was incorrectly invalidated and not counted, resulting in only 1 count instead of 2.

### Log Analysis

```
15:03:30 | TRACK_LIFECYCLE | T13 CREATED | bbox=(645, 556, 825, 720) center=(735, 638) conf=0.83
15:03:31 | TRACK_LIFECYCLE | T14 CREATED | bbox=(576, 377, 760, 562) center=(668, 469) conf=0.87
...
15:03:32 | TRACK_LIFECYCLE | T13 COMPLETED | type=track_invalid exit=top hits=6 missed=13 duration=1.93s
15:03:32 | PIPELINE | T13 INVALID_TRAVEL | exit=top frames=19 reason=did_not_follow_bottom_to_top_path
...
15:03:35 | TRACK_LIFECYCLE | T14 COMPLETED | type=track_completed exit=top hits=57 missed=3 duration=3.72s
15:03:35 | CLASSIFICATION | T14 COMPLETE | final=Brown_Orange conf=1.000
```

### Root Cause

1. **T13** appeared at Y=638 (middle of frame, not bottom)
2. **T13** only lived for **1.93 seconds** before exiting at the top
3. The travel validation logic required **2.0 seconds minimum duration**
4. **T13 was rejected** even though it was a valid bag moving in the correct direction

**Why T13 appeared mid-frame:**
- Late detection (detector missed it initially at the bottom)
- Lower confidence at entry point
- Partial occlusion or poor lighting at bottom
- Detector warm-up delay

**Why this is a problem:**
- Valid bags that are detected late get unfairly rejected
- The fixed 2.0s duration assumes all bags appear at the bottom of frame
- Bags appearing mid-frame have less distance to travel, so shorter durations are valid

## Solution: Adaptive Travel Duration Validation

### Implementation

Modified `ConveyorTracker._has_valid_travel_path()` to use **adaptive duration scaling**:

```python
# Calculate where track started as a ratio (0=top, 1=bottom)
entry_ratio = track.entry_center_y / self.frame_height

# Scale the minimum duration requirement proportionally:
# - Bottom entry (ratio=1.0): 100% of min duration (2.0s)
# - Mid entry (ratio=0.5): 50% of min duration (1.0s)
# - Top entry (ratio=0.3): 30% of min duration (0.6s)
duration_scale = max(0.3, entry_ratio)
min_travel_seconds = min_travel_seconds_base * duration_scale
```

### Example Calculations

| Entry Y | Frame Height | Entry Ratio | Duration Scale | Required Duration (base=2.0s) |
|---------|-------------|-------------|----------------|------------------------------|
| 700     | 720         | 0.97        | 0.97           | 1.94s                        |
| 638     | 720         | 0.89        | 0.89           | **1.78s** ✓ (T13 would pass) |
| 469     | 720         | 0.65        | 0.65           | 1.30s                        |
| 360     | 720         | 0.50        | 0.50           | 1.00s                        |
| 200     | 720         | 0.28        | 0.30*          | 0.60s                        |

*Minimum 30% floor to prevent ultra-short tracks

### Benefits

1. **Prevents false rejections**: T13 (1.93s duration, entry at Y=638) would now be valid
   - Required duration: 1.78s (89% of 2.0s)
   - Actual duration: 1.93s ✓ PASS

2. **Maintains noise filtering**: Ultra-short tracks still rejected
   - Minimum 30% floor ensures at least 0.6s duration required

3. **Adapts to real-world conditions**:
   - Late detections from occlusion
   - Detector confidence variations
   - Variable conveyor speeds
   - Lighting changes across frame

4. **Still validates direction**: Tracks must exit from top zone

## Configuration

The base duration is still configurable:

```python
# Environment variable
MIN_TRAVEL_DURATION_SECONDS=2.0  # Default

# Or in tracking_config.py
min_travel_duration_seconds: float = 2.0
```

The adaptive scaling is automatic based on entry position.

## Testing Recommendations

1. **Test with same video**: T13 should now be counted (expect 2 counts instead of 1)
2. **Monitor invalidation rate**: Should decrease significantly
3. **Check for false positives**: Ensure noise is still filtered (< 0.6s duration)
4. **Validate multi-bag scenarios**: Ensure adjacent bags (T13+T14) are both counted

## Verification

Run the system with the same video feed and verify:
- T13 is marked as `track_completed` instead of `track_invalid`
- T13 goes through classification and gets counted
- Final count shows 2 bags instead of 1

Expected log output:
```
T13 VALID: duration=1.93s entry_ratio=0.89 required_duration=1.78s exiting from top
T13 SUBMIT_CLASSIFY | total_rois=6 using=5
T13 COMPLETE | final=Brown_Orange conf=X.XXX
```

## Related Files Modified

1. `src/tracking/ConveyorTracker.py`
   - Modified `_has_valid_travel_path()` method
   - Added adaptive duration calculation logic

2. `src/config/tracking_config.py`
   - Updated `min_travel_duration_seconds` documentation
   - Explained adaptive scaling behavior

## Additional Notes

- This fix does NOT change the tracking/matching logic
- T13 and T14 were never merged - they were always separate tracks
- The issue was purely in the travel path validation rejecting T13
- No changes needed to IoU thresholds, distance thresholds, or matching logic
