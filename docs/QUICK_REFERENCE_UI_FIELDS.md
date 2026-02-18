# Quick Reference: UI Display Fields

## Status Panel (Top-Left)

### New Fields Added (Feb 18, 2026)

1. **App Version**
   - Location: Top of status panel
   - Format: `v2.4.2`
   - Color: Gray (secondary text)
   - Source: `config.APP_VERSION` from `src/config/settings.py`

2. **Lost Tracks**
   - Location: Below "Active Tracks"
   - Format: `Lost Tracks: 12`
   - Color: Red (error text)
   - Source: `state.lost_track_count` (cumulative counter)
   - Increments: Every time a track is lost before reaching exit zone

3. **Total (All)**
   - Location: Above "Pending" count
   - Format: Large badge with number
   - Color: Gold/Yellow badge
   - Calculation: `total_counted + tentative_total`
   - Shows: All detected items (confirmed + pending)

## Field Hierarchy

```
┌─ COUNTER STATUS ─────────────┐
│                               │
│ v2.4.2                    ← Version
│ ● FPS: 25.3               ← FPS
│ Active Tracks: 3          ← Active
│ Lost Tracks: 12           ← Lost (NEW)
├───────────────────────────────┤
│ Total (All)          [45] ← Total (NEW)
│ Pending               [5] ← Pending
│ Confirmed                 ← Confirmed
│           40              
├───────────────────────────────┤
│ By Class                  ← Classes
│   Black_Orange        15  
│   Blue_Yellow         10  
└───────────────────────────────┘
```

## Data Flow

```
ConveyorCounterApp
    ↓
CounterState.lost_track_count
    ↓
debug_info['lost_track_count']
    ↓
PipelineVisualizer._draw_status()
    ↓
UI Display
```

## State Variables

| Variable | Type | Location | Purpose |
|----------|------|----------|---------|
| `state.lost_track_count` | int | ConveyorCounterApp | Tracks lost count |
| `config.APP_VERSION` | str | settings.py | App version string |
| `total_counted` | int | CounterState | Confirmed count |
| `tentative_total` | int | CounterState | Pending count |

## Event Detection

Lost tracks are detected in `_on_track_event()`:
```python
if "lost before exit" in event:
    with self.state._lock:
        self.state.lost_track_count += 1
```

Triggered by `pipeline_core.py` when:
- Track exceeds max age without detection
- Track doesn't reach exit zone
- Not rescued by lost track recovery

## API/Endpoints

**Snapshot Endpoint**: `/snapshot?overlay=true`
- Automatically includes all new fields
- Uses same visualizer pipeline
- No changes needed

## Configuration

No configuration needed. All fields are automatically displayed when:
- `enable_display = True` in database config
- OR snapshot is requested via `/snapshot?overlay=true`

## Troubleshooting

**Version not showing:**
- Check: `from src.config.settings import config` in pipeline_visualizer.py
- Verify: `config.APP_VERSION` is defined in settings.py

**Lost track count not incrementing:**
- Check: Track events contain "lost before exit" string
- Verify: `_on_track_event()` callback is registered
- Debug: Add logging in `_on_track_event()`

**Total count incorrect:**
- Verify: `total_all = total_counted + tentative_total`
- Check: Both counts are being updated correctly
- Debug: Log both values before calculation

## Color Codes (BGR)

```python
'text_error': (120, 120, 255),     # Red for Lost Tracks
'text_info': (255, 200, 100),       # Gold for Total
'text_secondary': (180, 180, 190),  # Gray for Version
'text_warning': (100, 180, 255),    # Orange for Pending
'text_success': (100, 230, 120),    # Green for Confirmed
```

## Panel Sizing

- Width: 280px (unchanged)
- Height: `360 + (num_classes * 22)`
  - Was: `280 + (num_classes * 22)`
  - Increase: +80px for new fields
