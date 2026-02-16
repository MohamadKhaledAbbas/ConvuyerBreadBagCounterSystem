# Fix: Counts Page Showing Old Data

## Problem

The counts page (`/counts`) was displaying data from previous days, including yesterday's counts. The data persisted across application restarts because it was stored in a JSON file (`data/pipeline_state.json`) that was never reset.

### Root Cause

1. **Persistent State File**: The application writes count data to `data/pipeline_state.json`
2. **No Reset on Startup**: When the application restarted, it created a fresh in-memory `CounterState()`, but the old JSON file remained unchanged
3. **Web UI Reads Old Data**: The counts page reads directly from the persistent JSON file, not from the running application's memory
4. **Accumulation Over Days**: Data accumulated indefinitely, showing counts from yesterday, the day before, etc.

### Data Flow

```
Main App (ConveyorCounterApp)
  └─> Writes to pipeline_state.json every time a bag is classified/confirmed
  
FastAPI Server (/counts page)
  └─> Reads from pipeline_state.json (independent of main app's memory state)
```

## Solution

### Implementation

Added automatic state reset when the application starts in `ConveyorCounterApp.run()`:

```python
# Reset pipeline state file to clear old data
initial_state = {
    "confirmed": {},
    "pending": {},
    "just_classified": {},
    "confirmed_total": 0,
    "pending_total": 0,
    "just_classified_total": 0,
    "smoothing_rate": 0.0,
    "window_status": {
        "size": self._smoother.window_size,
        "current_items": 0,
        "next_confirmation_in": self._smoother.window_size
    },
    "recent_events": [],
    "current_batch_type": None,
    "previous_batch_type": None,
    "last_classified_type": None
}
write_pipeline_state(initial_state)
logger.info("[ConveyorCounterApp] Pipeline state reset - counts page will show today's data only")
```

### Behavior After Fix

1. **Application Startup**: `pipeline_state.json` is cleared to empty state
2. **During Operation**: File is updated with current session's counts
3. **Web UI**: Always shows current session data only
4. **Next Day**: When app restarts, state resets again automatically

### What Gets Reset

- **Confirmed counts**: All class counts reset to 0
- **Pending counts**: Smoothing buffer cleared
- **Just classified**: Tentative counts cleared
- **Recent events**: Event log cleared
- **Batch tracking**: Current/previous batch types cleared
- **Statistics**: Smoothing rate reset to 0.0

### What Persists (in Database)

The **database** (`data/counting.db`) still contains the full historical record:
- All classified bags with timestamps
- Track lifecycle events
- Full analytics history

The database is NOT reset, so you can still:
- View analytics for any date range
- Query historical data
- Generate reports for previous days/shifts

## File Modified

- `src/app/ConveyorCounterApp.py`
  - Modified `run()` method to reset pipeline state at startup

## Testing

1. **Before Fix**: 
   - Stop application
   - Check `/counts` page → shows old data
   - Start application
   - Check `/counts` page → still shows old data

2. **After Fix**:
   - Stop application
   - Check `/counts` page → shows old data (expected)
   - Start application
   - Check `/counts` page → shows 0 counts (reset successful)
   - Wait for bags to be counted
   - Check `/counts` page → shows only new counts

## Design Considerations

### Why Reset on Startup vs. Midnight Reset?

**Option 1: Reset at Midnight** (Not chosen)
- Requires background timer thread
- Timezone complications
- What if app isn't running at midnight?
- More complex code

**Option 2: Reset on Startup** (Chosen ✓)
- Simple and reliable
- No background threads needed
- Works regardless of timezone
- Clear behavior: "counts start from 0 when app starts"
- Aligns with typical usage pattern (daily restarts)

### Why Not Clear the Database?

The database contains valuable historical data:
- Long-term trend analysis
- Shift comparisons
- Quality control metrics
- Audit trail

Clearing it would lose important business intelligence. The separation between:
- **Real-time state** (pipeline_state.json) - ephemeral, session-based
- **Historical data** (counting.db) - persistent, permanent

...provides the best of both worlds.

## Related Endpoints

- `GET /counts` - HTML dashboard (reads from pipeline_state.json)
- `GET /api/counts` - JSON endpoint (reads from pipeline_state.json)
- `GET /api/counts/stream` - SSE stream (reads from pipeline_state.json)
- `GET /analytics` - Historical analytics (reads from database)

Only the `/counts` endpoints are affected by the state reset.

## Usage Notes

- **Daily Operation**: Start the application once per day/shift, and it will show counts from 0
- **Continuous Operation**: If running 24/7, counts accumulate until the app restarts
- **Historical Data**: Use `/analytics` endpoint to view previous day/shift data
- **State File Location**: `data/pipeline_state.json` (configurable via `PIPELINE_STATE_FILE` env var)

## Verification

Check the logs on startup:
```
[ConveyorCounterApp] Starting with modular pipeline...
[ConveyorCounterApp] Components initialized with modular architecture
[ConveyorCounterApp] Pipeline state reset - counts page will show today's data only
```

If you see the "Pipeline state reset" message, the fix is working correctly.
