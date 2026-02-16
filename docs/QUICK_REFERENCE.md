# Quick Reference: What Changed

## Problem 1: T13 Invalid Rejection (Appeared as "Merge")
**Symptom**: Two bags detected but only one counted  
**Fix**: Adaptive travel duration validation  
**File**: `src/tracking/ConveyorTracker.py`

## Problem 2: Counts Page Shows Old Data
**Symptom**: `/counts` shows yesterday's data  
**Fix**: Reset pipeline state on startup  
**File**: `src/app/ConveyorCounterApp.py`

---

## How to Verify Fixes

### After Application Restart:

1. **Check startup logs** for:
   ```
   [ConveyorCounterApp] Pipeline state reset - counts page will show today's data only
   ```

2. **Open** http://localhost:8000/counts
   - Should show **0 counts** initially
   - Should increment as bags are counted
   - Should NOT show old data

3. **Look for T13-like tracks** in logs:
   - Should see "VALID" instead of "INVALID" for mid-frame entries
   - Should see "SUBMIT_CLASSIFY" for tracks that were previously rejected

---

## Test Commands

```bash
# Test adaptive duration fix
python test_adaptive_duration.py

# Test state reset fix
python test_state_reset.py
```

---

## If Issues Persist

### T13 Still Being Rejected:
- Check log: Is entry_y and required_duration being calculated?
- Look for: `entry_ratio=0.89 required_duration=1.78s`
- Actual duration must be >= required duration

### Counts Page Still Shows Old Data:
- Check if application actually restarted (not just reconnected)
- Check log for "Pipeline state reset" message
- Manually delete `data/pipeline_state.json` and restart

---

## Rollback (If Needed)

### To Disable Adaptive Duration:
In `src/tracking/ConveyorTracker.py`, line ~670, change:
```python
# From adaptive:
duration_scale = max(0.3, entry_ratio)
min_travel_seconds = min_travel_seconds_base * duration_scale

# To fixed:
min_travel_seconds = min_travel_seconds_base
```

### To Disable State Reset:
In `src/app/ConveyorCounterApp.py`, line ~848, comment out:
```python
# write_pipeline_state(initial_state)
# logger.info("[ConveyorCounterApp] Pipeline state reset...")
```

---

## Documentation

- `docs/FIX_T13_T14_MERGE_ISSUE.md` - Detailed T13/T14 analysis
- `docs/FIX_COUNTS_PAGE_OLD_DATA.md` - Detailed state reset explanation
- `docs/SUMMARY_TWO_FIXES.md` - Complete summary of both fixes
