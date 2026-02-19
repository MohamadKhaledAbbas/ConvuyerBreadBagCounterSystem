# Track Lifecycle Architecture

## Overview

The ConveyorTracker implements a production-grade counting system based on one clean rule:
**A bread bag is counted if and only if its track exits from the top of the frame.**

Lost tracks are never counted — they are logged for debugging only.

## Core Counting Rule

```
A bread bag is COUNTED if and only if:
  1. Track exits from the TOP of the frame (exit_zone_ratio check)
  2. Track passes _has_valid_travel_path() (adaptive time + direction checks)

Lost tracks are NEVER counted. They are logged to the database for analytics.
```

## Three Reliability Layers

### Layer 1: Ghost Track Recovery (Occlusion Handling)

**Problem**: A bag gets temporarily occluded → track lost after `max_frames_without_detection` frames → bag reappears → new track ID → original track never exits top → under-count.

**Solution**: When a track is lost, hold it in a `ghost_tracks` buffer for up to `ghost_track_max_age_seconds` (default: 4s). Each frame:

1. Predict each ghost's position using its last known velocity
2. Before creating new tracks, check if any unmatched detection matches a ghost:
   - X-axis within `ghost_track_x_tolerance_pixels` (50px) — bags don't move sideways
   - Y-axis within `ghost_track_max_y_gap_ratio` × frame_height (20%) of predicted position
3. If matched → re-associate as same track_id with incremented `ghost_recovery_count`
4. If ghost expires → finalize as `track_lost`

### Layer 2: Shadow Tracks (Merge Detection)

**Problem**: Two bags traveling close together merge into one detection near the top of frame → one track lost → under-count.

**Solution**: When a track goes missing, check if a nearby active track absorbed it:

1. Surviving track's bbox width grew ≥ `merge_bbox_growth_threshold` (1.4 = 40%)
2. Tracks were spatially adjacent (X gap < 50px, Y diff < 30px)
3. Both moving in same direction (toward top)

If all pass → lost track becomes a "shadow" of the survivor. When survivor exits top, count includes all shadows.

### Layer 3: Entry Type Classification (Diagnostics Only)

Classifies each track's origin for operator visibility. **Does NOT affect counting.**

| Entry Type | Rule | Flag |
|---|---|---|
| `bottom_entry` | Created in bottom 40% of frame | None |
| `thrown_entry` | Created mid-frame + high initial velocity (≥ 15 px/frame) | None |
| `midway_entry` | Created mid-frame + normal velocity | `suspected_duplicate` |

## Configuration Parameters

```python
# Ghost Track Recovery
ghost_track_max_age_seconds = 4.0        # Max time to hold ghost
ghost_track_x_tolerance_pixels = 50.0    # X-axis tolerance for re-association
ghost_track_max_y_gap_ratio = 0.2        # Max Y gap as fraction of frame height

# Shadow / Merge Detection
merge_bbox_growth_threshold = 1.4        # 40% bbox width growth triggers merge check
merge_spatial_tolerance_pixels = 50.0    # Max X gap between tracks
merge_y_tolerance_pixels = 30.0          # Max Y difference between tracks

# Entry Type Classification
bottom_entry_zone_ratio = 0.4            # Bottom 40% = bottom_entry
thrown_entry_min_velocity = 15.0          # px/frame threshold for thrown_entry
thrown_entry_detection_frames = 5         # Frames to measure initial velocity
```

## Database Schema

The `track_events` table includes enriched lifecycle fields:

- `entry_type`: bottom_entry | midway_entry | thrown_entry
- `suspected_duplicate`: 1 if midway_entry
- `ghost_recovery_count`: times re-associated after occlusion
- `shadow_of`: track_id this was a shadow of
- `shadow_count`: shadows riding on this track at exit
- `occlusion_events`: JSON array of recovery events
- `merge_events`: JSON array of merge/unmerge events

## Scenario Outcomes

| # | Scenario | Count | Correct? |
|---|---|---|---|
| 1 | Normal: bottom → exits top | 1 | ✅ |
| 2 | Falls at 50%, worker puts back | 1 | ✅ |
| 3 | Occluded → reappears after 2s | 1 | ✅ |
| 4 | Occluded → never reappears | 0 | ✅ |
| 5 | Two bags merge near top | 2 | ✅ |
| 6 | Falls off, never put back | 0 | ✅ |

## What Was Removed

- `_validate_lost_track_as_completed()` method — deleted entirely
- `lost_track_entry_zone_ratio`, `lost_track_exit_zone_ratio`, `lost_track_min_travel_ratio`, `lost_track_min_hit_rate` config params — removed
- The "rescue" logic that changed `track_lost` to `track_completed` — removed
