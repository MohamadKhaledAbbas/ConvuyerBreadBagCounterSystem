# Quick Reference: Lost Track Recovery

## What It Does
Rescues tracks that traveled from bottom to near-top but got lost before strictly exiting, preventing undercounting.

## When It Activates
When a track times out (`missed > max_frames_without_detection`)

## Validation Checks (ALL must pass)

| Check | Threshold | Purpose |
|-------|-----------|---------|
| **Entry Zone** | Started in bottom 60% | Ensure valid entry |
| **Exit Zone** | Ended in top 40% | Ensure reached near-exit |
| **Travel Distance** | >= 30% of frame height | Ensure significant journey |
| **Hit Rate** | >= 50% | Ensure reliable tracking |

## Configuration Quick Edit

```python
# src/config/tracking_config.py

lost_track_entry_zone_ratio = 0.6   # Default: bottom 60%
lost_track_exit_zone_ratio = 0.4    # Default: top 40%
lost_track_min_travel_ratio = 0.3   # Default: 30% of height
lost_track_min_hit_rate = 0.5       # Default: 50% detection rate
```

## Environment Variables

```bash
export LOST_TRACK_ENTRY_ZONE_RATIO=0.6
export LOST_TRACK_EXIT_ZONE_RATIO=0.4
export LOST_TRACK_MIN_TRAVEL_RATIO=0.3
export LOST_TRACK_MIN_HIT_RATE=0.5
```

## Quick Diagnostics

```bash
# Count rescues in last hour
grep -E "RESCUED" data/logs/app.log | tail -100 | wc -l

# View rescue details
grep "RESCUED" data/logs/app.log | tail -20

# Compare lost vs rescued
echo "Lost tracks:" $(grep "type=track_lost" data/logs/app.log | wc -l)
echo "Rescued tracks:" $(grep "RESCUED" data/logs/app.log | wc -l)
```

## Expected Behavior

### ✅ Good Signs
- Rescue rate: 5-15% of lost tracks
- Rescued tracks have clear bottom→top trajectory
- Total counts increase by 2-5%

### ⚠️ Warning Signs  
- Rescue rate > 30%: Too permissive, tighten thresholds
- Rescue rate < 2%: Too strict, relax thresholds
- Unexpected count jumps: Verify with manual inspection

## Quick Tuning

### Too Many Rescues?
```python
lost_track_min_travel_ratio = 0.35  # ↑ Require more travel
lost_track_min_hit_rate = 0.6       # ↑ Require better quality
lost_track_exit_zone_ratio = 0.3    # ↓ Must reach closer to top
```

### Too Few Rescues?
```python
lost_track_min_travel_ratio = 0.25  # ↓ Accept less travel
lost_track_min_hit_rate = 0.4       # ↓ Accept lower quality
lost_track_exit_zone_ratio = 0.5    # ↑ Accept further from top
```

## Test Commands

```bash
# Unit test (validation logic)
python test_lost_track_validation.py

# Exit zone filter test
python test_exit_zone_filter.py
```

## Log Messages to Watch

### Success
```
[TRACK_LIFECYCLE] T42 RESCUED | Lost track validated as completed (valid journey from bottom to top)
```

### Rejection (debug level)
```
[VALIDATE_LOST] T43 REJECT: didn't reach near-top (y=450 > 288.0)
[VALIDATE_LOST] T44 REJECT: insufficient travel (traveled=180px < 216px required)
[VALIDATE_LOST] T45 REJECT: low hit rate (0.30 < 0.50)
```

## Related Features

| Feature | Doc |
|---------|-----|
| Exit zone filtering | `docs/FIX_UNNECESSARY_TRACK_CREATION.md` |
| Full details | `docs/LOST_TRACK_RECOVERY.md` |
| Implementation summary | `docs/IMPLEMENTATION_SUMMARY_TRACK_RECOVERY.md` |

---

**Quick Check**: Is it working?
```bash
# Should show RESCUED messages if feature is active
tail -f data/logs/app.log | grep RESCUED
```
