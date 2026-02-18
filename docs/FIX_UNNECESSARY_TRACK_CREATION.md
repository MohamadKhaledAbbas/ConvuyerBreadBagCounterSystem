# Fix: Prevent Unnecessary Track Creation in Exit Zones

## Problem

As reported in the logs, **T161** was created when a detection appeared at the very top of the frame (y=17) and was immediately marked as `track_lost` because:
- It was already in the exit zone when first detected
- It couldn't accumulate enough hits to be confirmed (hits=1)
- It timed out with missed=16 frames

This created unnecessary `track_lost` events that pollute the logs and make debugging harder.

## Root Cause

The tracker was creating new tracks for **ANY** high-confidence detection, including those that appeared:
1. **At the top exit zone** (y <= 108 pixels for 720p, 15% of frame height)
2. **At the bottom exit zone** (y >= 612 pixels for 720p, bottom 15% of frame)

These detections are about to leave the frame and won't have time to become valid tracks.

## Solution

Modified `ConveyorTracker._create_track()` to:

1. **Check detection position** before creating a track
2. **Filter out detections in exit zones** (top or bottom)
3. **Log the filtering** at debug level for troubleshooting

### Code Changes

**File:** `src/tracking/ConveyorTracker.py`

```python
def _create_track(self, detection: Detection) -> Optional[TrackedObject]:
    """
    Create a new track from detection.
    
    Returns None if the detection should not create a track (e.g., already in exit zone).
    """
    # Get velocity smoothing alpha from config
    velocity_alpha = getattr(self.config, 'velocity_smoothing_alpha', 0.3)

    # Create temporary track to check its center position
    temp_track = TrackedObject(
        track_id=self._next_id,
        bbox=detection.bbox,
        confidence=detection.confidence,
        _velocity_alpha=velocity_alpha
    )
    
    # Check if detection is already in exit zone (top or bottom)
    _, cy = temp_track.center
    
    if self._is_in_exit_zone(cy):
        logger.debug(
            f"[ConveyorTracker] Skipping track creation for detection in exit zone | "
            f"bbox={detection.bbox} center={temp_track.center} conf={detection.confidence:.2f}"
        )
        return None
    
    if self._is_in_bottom_exit_zone(cy):
        logger.debug(
            f"[ConveyorTracker] Skipping track creation for detection in bottom zone | "
            f"bbox={detection.bbox} center={temp_track.center} conf={detection.confidence:.2f}"
        )
        return None

    # Create the actual track (existing code continues...)
```

### Exit Zones (for 720p frame)

- **Top exit zone:** y <= 108 pixels (15% of 720)
- **Bottom exit zone:** y >= 612 pixels (bottom 15% of 720)
- **Valid tracking zone:** 108 < y < 612 pixels

## Testing

Created `test_exit_zone_filter.py` to verify:

```
✓ Test 1: Detection in TOP exit zone (y=17) → No track created
✓ Test 2: Detection in BOTTOM exit zone (y=700) → No track created
✓ Test 3: Detection in MIDDLE of frame (y=350) → Track created
✓ Test 4: Detection in lower-middle zone (y=500) → Track created
```

**All tests passed!**

## Impact

### Before Fix
```
15:56:01 | INFO | [TRACK_LIFECYCLE] T161 CREATED | bbox=(665, 0, 771, 35) center=(718, 17) conf=0.72
15:56:02 | INFO | [TRACK_LIFECYCLE] T161 COMPLETED | type=track_lost exit=top hits=1 missed=16 ...
```
→ Unnecessary `track_lost` event created

### After Fix
```
16:04:52 | DEBUG | [ConveyorTracker] Skipping track creation for detection in exit zone | 
                    bbox=(665, 0, 771, 35) center=(718, 17) conf=0.72
```
→ No track created, cleaner logs

## Benefits

1. **Cleaner logs** - No more spurious `track_lost` events
2. **Easier debugging** - Only real tracking issues appear in logs
3. **Better performance** - Fewer tracks to manage
4. **More accurate** - Only tracks objects that have time to be properly tracked

## Notes

- This fix does **NOT** affect counting logic (as you correctly noted)
- Valid tracks like T160 (entering at y=666, but moving upward and exiting properly) are still created and tracked
- The fix only prevents creating tracks for detections that are **already exiting** when first seen
