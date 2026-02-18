# UI Enhancements: Version, Total Count, and Lost Tracks

**Date**: February 18, 2026  
**Version**: v2.4.2

## Overview

Enhanced the UI frame display to show additional important information:
1. **APP_VERSION** - Application version number
2. **Total Count** - Sum of pending + persisted counts
3. **Lost Track Count** - Number of lost track events

## Changes Made

### 1. CounterState (ConveyorCounterApp.py)

**Added Field:**
```python
lost_track_count: int = 0  # Total lost track events
```

**Purpose:** Track the cumulative number of tracks that were lost before reaching the exit zone.

### 2. Track Lost Event Counting (ConveyorCounterApp.py)

**Modified `_on_track_event` method:**
```python
def _on_track_event(self, event: str):
    """Callback for track-related events (for UI debugging)."""
    self.state.add_event(event)
    
    # Count lost track events
    if "lost before exit" in event:
        with self.state._lock:
            self.state.lost_track_count += 1
```

**Purpose:** Automatically increment the lost track counter when a track is lost before exit.

### 3. Debug Info (ConveyorCounterApp.py)

**Added to debug_info dict:**
```python
debug_info = {
    # ...existing fields...
    'lost_track_count': self.state.lost_track_count
}
```

**Purpose:** Pass lost track count to the visualizer for display.

### 4. Pipeline Visualizer (pipeline_visualizer.py)

#### Updated `annotate_frame` method:
```python
lost_track_count = debug_info.get('lost_track_count', 0) if debug_info else 0
self._draw_status(
    annotated, fps, active_tracks, total_counted, counts_by_class,
    tentative_total, tentative_counts, lost_track_count
)
```

#### Enhanced `_draw_status` method:

**Added to display:**

1. **App Version** (top of panel)
   ```python
   from src.config.settings import config
   version_text = f"v{config.APP_VERSION}"
   cv2.putText(frame, version_text, ...)
   ```

2. **Lost Tracks Counter** (below Active Tracks)
   ```python
   cv2.putText(
       frame, f"Lost Tracks: {lost_track_count}", (x, y + 12),
       self.FONT, self.FONT_SCALE_MEDIUM,
       self.COLORS['text_error'], self.FONT_WEIGHT_NORMAL
   )
   ```

3. **Total Count (All)** - Pending + Persisted (before individual counts)
   ```python
   total_all = total_counted + tentative_total
   cv2.putText(frame, "Total (All)", ...)
   # Display total_all in a badge with gold/yellow color
   ```

**Panel Height Increased:**
```python
panel_h = 360 + (num_classes * self.LINE_HEIGHT_SMALL)  # Was 280
```

## Visual Layout

The updated status panel now displays (top to bottom):

```
┌─────────────────────────────┐
│     COUNTER STATUS          │
├─────────────────────────────┤
│ v2.4.2                      │  ← NEW: App Version
│ ● FPS: 25.3                 │
│ Active Tracks: 3            │
│ Lost Tracks: 12             │  ← NEW: Lost Track Count
├─────────────────────────────┤
│ Total (All)            [45] │  ← NEW: Pending + Persisted
│ Pending                 [5] │
│ Confirmed                   │
│           40                │  ← Large number
├─────────────────────────────┤
│ By Class                    │
│   Black_Orange          15  │
│   Blue_Yellow           10  │
│   ...                       │
└─────────────────────────────┘
```

## Color Scheme

- **App Version**: Gray text (text_secondary)
- **Lost Tracks**: Red text (text_error)
- **Total Count**: Gold badge (text_info)
- **Pending**: Orange badge (text_warning)
- **Confirmed**: Green text (text_success)

## Benefits

1. **Version Visibility**: Easy to verify which version is running
2. **Complete Count**: Shows total items detected (pending + confirmed)
3. **Quality Metrics**: Lost track count helps assess tracking performance
4. **Debugging**: Helps identify if tracking quality degrades over time

## Testing

To verify the changes:

1. Run the application
2. Check the top-left status panel
3. Verify:
   - Version number appears at top
   - Lost Tracks counter increments when tracks are lost
   - Total (All) = Pending + Confirmed counts
   - All numbers update in real-time

## Snapshot Feature

The snapshot endpoint (`/snapshot?overlay=true`) automatically includes these enhancements because it uses the same visualizer with the same `debug_info` data.

## Notes

- Lost track count is cumulative (never resets during runtime)
- Total count updates in real-time as items are detected and classified
- Thread-safe implementation ensures accurate counting in multi-threaded environment
- Changes are backward compatible - old code continues to work

## Related Files

- `src/app/ConveyorCounterApp.py` - State management and event tracking
- `src/app/pipeline_visualizer.py` - UI rendering
- `src/config/settings.py` - APP_VERSION constant

## Version History

- **v2.4.2** (Feb 18, 2026): Added version, total count, and lost track display
