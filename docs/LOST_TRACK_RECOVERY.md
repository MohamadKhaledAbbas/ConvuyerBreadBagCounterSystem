# Lost Track Recovery - Valid Journey Detection

## Overview

This feature rescues tracks that get "lost" before reaching the strict exit zone but have completed a valid journey from bottom to near-top of the frame.

## Problem Statement

In conveyor belt tracking, bags don't just vanish. If a bag travels from the entry point (bottom) up 70% of the screen and then "disappears" (tracker loses it), it almost certainly passed through. This can happen due to:

1. **Occlusion/Merging**: Objects merge as they get smaller/further away
2. **Detector Confidence Issues**: Detection gets spotty at the far end
3. **Frame Boundaries**: Objects move slightly out of frame or camera cuts off top edge
4. **Lighting**: Poor lighting conditions near the exit

Previously, these tracks were marked as `track_lost` and not counted, resulting in undercounting.

## Solution: Valid Journey Detection

Instead of asking "Did it exit the top pixel?", we ask "Did it travel a valid journey?"

### Validation Criteria

A lost track is rescued as `track_completed` if ALL of the following are true:

1. **Entry Zone**: Started in bottom 60% of frame (y >= 40% of frame_height)
2. **Exit Zone**: Ended in top 40% of frame (y <= 40% of frame_height)  
3. **Travel Distance**: Traveled at least 30% of frame height vertically
4. **Detection Quality**: Hit rate >= 50% (hits / (hits + age))

### Configuration Parameters

```python
# In tracking_config.py

lost_track_entry_zone_ratio: float = 0.6
"""
Bottom fraction where valid tracks should start (0.6 = bottom 60%).
Ensures we only count bags that entered from the expected direction.
"""

lost_track_exit_zone_ratio: float = 0.4
"""
Top fraction where lost tracks must reach to be rescued (0.4 = top 40%).
More relaxed than strict exit_zone_ratio (15%) to handle occlusion/merging.
"""

lost_track_min_travel_ratio: float = 0.3
"""
Minimum vertical distance as fraction of frame height (0.3 = 30%).
Prevents counting noise/short-lived tracks.
"""

lost_track_min_hit_rate: float = 0.5
"""
Minimum detection hit rate for recovery (0.5 = 50%).
Ensures we only rescue tracks that were reliably detected.
"""
```

### Example Scenarios (for 720p frame = 720px height)

#### ✅ Scenario 1: Valid Journey - RESCUED
- Start: y=650 (bottom 10%, in entry zone)
- End: y=250 (top 35%, in rescue zone)
- Travel: 400px (56% of frame)
- Hit rate: 12/15 = 80%
- **Result**: `track_completed` ✓

#### ✅ Scenario 2: Occlusion Near Top - RESCUED  
- Start: y=600 (bottom 17%, in entry zone)
- End: y=280 (top 39%, just made it to rescue zone)
- Travel: 320px (44% of frame)
- Hit rate: 8/10 = 80%
- **Result**: `track_completed` ✓

#### ❌ Scenario 3: Started Too High - NOT RESCUED
- Start: y=250 (top 35%, NOT in entry zone)
- End: y=150 (top 21%)
- Travel: 100px (14% of frame)
- **Result**: `track_lost` ✗ (didn't enter from bottom)

#### ❌ Scenario 4: Insufficient Travel - NOT RESCUED
- Start: y=650 (in entry zone)
- End: y=450 (mid-frame, NOT in rescue zone)
- Travel: 200px (28% of frame, < 30% required)
- **Result**: `track_lost` ✗ (didn't reach near-top)

#### ❌ Scenario 5: Poor Detection Quality - NOT RESCUED
- Start: y=650 (in entry zone)
- End: y=250 (in rescue zone)
- Travel: 400px (sufficient)
- Hit rate: 3/10 = 30% (< 50% required)
- **Result**: `track_lost` ✗ (unreliable tracking)

## Implementation

### Code Location

- **Tracker**: `src/tracking/ConveyorTracker.py`
  - `_validate_lost_track_as_completed()` - validation logic
  - `_check_completed_tracks()` - calls validation when track times out

- **Config**: `src/config/tracking_config.py`
  - Configuration parameters with defaults

### Logic Flow

```
Track timeout detected (missed > max_frames_without_detection)
    ↓
_validate_lost_track_as_completed() called
    ↓
Check 1: Started in entry zone? (bottom 60%)
    ↓ YES
Check 2: Ended in rescue zone? (top 40%)
    ↓ YES
Check 3: Traveled enough? (>= 30% of height)
    ↓ YES
Check 4: Good hit rate? (>= 50%)
    ↓ YES
RESCUE: Mark as track_completed ✓
    ↓
Track gets counted!
```

### Log Messages

#### Rescued Track
```
[TRACK_LIFECYCLE] T42 RESCUED | Lost track validated as completed (valid journey from bottom to top)
[TRACK_LIFECYCLE] T42 COMPLETED | type=track_completed exit=timeout hits=12 missed=16 duration=2.34s distance=456px ...
```

#### Rejected Track (with debug logging)
```
[VALIDATE_LOST] T43 REJECT: didn't reach near-top (y=450 > 288.0)
[TRACK_LIFECYCLE] T43 COMPLETED | type=track_lost ...
```

## Testing

### Unit Test
```bash
python test_lost_track_validation.py
```

Tests the `_validate_lost_track_as_completed()` method directly with various scenarios.

### Integration Test
```bash
python test_lost_track_recovery.py
```

Tests full tracking pipeline with simulated detection sequences.

## Benefits

1. **Accurate Counting**: Recovers counts from bags that were tracked successfully but lost near the exit
2. **Handles Occlusion**: Counts bags that merge with others near the top
3. **Robust to Detector Issues**: Tolerates detection failures in the far field
4. **Configurable**: Easy to tune thresholds based on your specific setup
5. **Transparent**: Clear logging shows which tracks are rescued and why

## Tuning Guide

### If you're OVER-counting (rescuing too many invalid tracks):

- **Increase** `lost_track_min_travel_ratio` (e.g., 0.35 or 0.4)
- **Increase** `lost_track_min_hit_rate` (e.g., 0.6 or 0.7)
- **Decrease** `lost_track_exit_zone_ratio` (e.g., 0.3 to require closer to actual exit)

### If you're UNDER-counting (not rescuing valid tracks):

- **Decrease** `lost_track_min_travel_ratio` (e.g., 0.25 or 0.2)
- **Decrease** `lost_track_min_hit_rate` (e.g., 0.4)
- **Increase** `lost_track_exit_zone_ratio` (e.g., 0.5 to accept tracks that didn't go as far)

### Environment Variables

You can override defaults via environment variables:

```bash
export LOST_TRACK_ENTRY_ZONE_RATIO=0.7
export LOST_TRACK_EXIT_ZONE_RATIO=0.35
export LOST_TRACK_MIN_TRAVEL_RATIO=0.25
export LOST_TRACK_MIN_HIT_RATE=0.6
```

## Monitoring

### Key Metrics to Watch

1. **Rescue Rate**: `track_completed (rescued) / total track_lost`
   - Typical: 5-15% of lost tracks should be rescued
   - If > 30%: Detector or tracking issues, tighten criteria
   - If < 5%: May be too strict, relax criteria

2. **False Positive Rate**: Manual spot-check rescued tracks
   - Review saved ROIs from rescued tracks
   - Should show clear valid journeys

3. **Count Accuracy**: Compare to manual counts
   - Should improve accuracy by 2-5%

## Related Features

- **Exit Zone Filtering**: Prevents creating tracks in exit zones ([FIX_UNNECESSARY_TRACK_CREATION.md](FIX_UNNECESSARY_TRACK_CREATION.md))
- **Adaptive Duration**: Time-based validation adapts to entry position
- **Bidirectional Smoothing**: Corrects misclassifications post-counting

## Version History

- **v1.0** (2025-02-18): Initial implementation
  - 4 validation criteria
  - Configurable thresholds
  - Debug logging

---

**Note**: This feature works in conjunction with the existing strict exit zone validation. Tracks that cleanly exit through the top zone are still marked as `track_completed` via the normal path. This rescue logic only applies to tracks that timeout before reaching the strict exit zone.
