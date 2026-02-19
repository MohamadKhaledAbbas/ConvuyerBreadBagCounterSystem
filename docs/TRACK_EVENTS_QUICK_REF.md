# Track Events - Quick Reference Guide

## 🎯 Quick Access

| Feature | URL | Purpose |
|---------|-----|---------|
| **Main Page** | `/track-events` | View all tracks with filtering |
| **Track Details** | `/track-events/{id}` | JSON: Full lifecycle details |
| **Animation Data** | `/track-events/{id}/animation` | JSON: Animation keyframes |
| **Animation View** | `/track-events/{id}/visualize` | Interactive animated trajectory |
| **Events API** | `/api/track-events` | JSON: Paginated events |
| **Stats API** | `/api/track-events/stats` | JSON: Analytics & histograms |

## 🔍 Common Filters

### By Status
```
/track-events?event_type=track_completed    # ✅ Successful tracks
/track-events?event_type=track_lost         # 👻 Lost tracks
/track-events?event_type=track_invalid      # ❌ Invalid paths
```

### By Classification
```
/track-events?classification=Wheatberry                    # Wheatberry bags
/track-events?classification=Brown_Orange&event_type=track_lost   # Lost brown/orange
```

### By Quality
```
/track-events?min_confidence=0.85                # High confidence only
/track-events?min_duration=0.5&max_duration=2   # 0.5-2 second tracks
```

### By Entry Type
```
/track-events?entry_type=bottom_entry    # Normal bottom entry
/track-events?entry_type=thrown_entry    # Items thrown mid-frame
/track-events?entry_type=midway_entry    # Partial trajectory (suspicious)
```

### By Exit
```
/track-events?exit_direction=bottom   # Exited normally
/track-events?exit_direction=top      # Exited from top (anomaly)
/track-events?exit_direction=timeout  # Timed out (lost)
```

### Combined Filters
```
# High-quality successful Wheatberry
?event_type=track_completed&classification=Wheatberry&min_confidence=0.9

# Lost midway entries (data quality issue)
?entry_type=midway_entry&event_type=track_lost

# Recovery analysis
?classification=Wheatberry&start_time=2026-02-19T00:00:00&end_time=2026-02-19T23:59:59
```

## 📊 Statistics Dashboard

The main page shows 5 key metrics:

```
┌─────────────────┬─────────────┬──────────┬─────────────┬──────────────┐
│ Total Events    │ Completed   │ Lost     │ Invalid     │ Success Rate │
│ 1000 tracks     │ 800 ✅      │ 150 👻   │ 50 ❌       │ 80%          │
└─────────────────┴─────────────┴──────────┴─────────────┴──────────────┘
```

Plus 3 distribution charts:

```
Top Classifications       Duration Distribution      Confidence Distribution
─────────────────        ─────────────────          ────────────────────
Wheatberry     400       0-1s      ████████ 200     95-100% ██████████ 460
Brown_Orange   300       1-2s      ███████████ 400  85-95%  ████████ 400
Bran           200       2-3s      █████████ 300    70-85%  ██ 100
...                      3-5s      ██ 80            50-70%  █ 30
```

## 🎬 Track Animation

### View Animation
1. Click track ID (e.g., "T123") in table - opens in new tab
2. Click "Visualize" link next to track ID
3. Direct URL: `/track-events/123/visualize`

### Animation Controls
```
┌─────────────────────────────────────────┐
│ ▶ Play  🔄 Reset  Speed: [1x ↕ 3x]      │
├─────────────────────────────────────────┤
│ Timeline: 0:00 ━━━●━━━━━ / 2:30        │
├─────────────────────────────────────────┤
│ Canvas: Track path from entry to exit   │
│ 🟡 Entry    ────────→ ❌ Exit           │
│ 🟪 ROI boxes  🟢 Classification events  │
└─────────────────────────────────────────┘
```

### Animation Features
- **Play/Pause**: Control playback
- **Speed**: 0.5x to 3x (default 1x)
- **Scrubber**: Click timeline to jump to frame
- **Fullscreen**: Expand to full screen
- **Sidebar**: Track stats, events, legend

## 📑 Pagination

Pages show 50 events by default (adjustable 10-200 per page):

```
Showing page 1 of 20 (1000 total events)
[◀ First] [◀ Prev] [Next ▶] [Last ▶]
```

**Note**: Add `&page_size=100` to URL to change page size

## 🔗 API Examples

### Get events as JSON
```bash
curl '/api/track-events?event_type=track_completed&page=1&page_size=100'
```

### Get statistics
```bash
curl '/api/track-events/stats?start_time=2026-02-19T00:00:00&end_time=2026-02-19T23:59:59'
```

### Get animation keyframes
```bash
curl '/track-events/123/animation' | jq '.animation.keyframes | length'
# Output: 50 (frames in animation)
```

### Get full lifecycle
```bash
curl '/track-events/123' | jq '.details | length'
# Output: 15 (lifecycle steps)
```

## 💡 Tips & Tricks

### Find Problem Tracks
```
1. Filter: entry_type=midway_entry&event_type=track_lost
2. Review: Which bags cause mid-frame entry?
3. Action: Adjust entry detection parameters
```

### Analyze Classification Accuracy
```
1. Click Stats API to get distribution
2. Check: Ghost recovery count vs total tracks
3. Compare: by_classification vs by_type
```

### Export Track Data
```
1. Use /api/track-events with filters
2. Save JSON response
3. Process with your tools (pandas, excel, etc.)
```

### Debug a Failed Track
```
1. Note Track ID (e.g., T456)
2. Go to: /track-events/456/visualize
3. Watch animation with ROI overlays
4. Check lifecycle steps in Details tab
```

### Monitor Ghost Recovery
```
1. View stats API response
2. Check: recovery_stats.recovered_tracks
3. Track: Occlusion frequency over time
4. Adjust: Ghost buffer timeout if needed
```

## ⚙️ Performance Tips

### Speed Up Queries
- ✅ Use specific time ranges (1-7 days)
- ✅ Add classification filter
- ❌ Avoid very large date ranges (month+)
- ❌ Avoid combining many filters

### Pagination Strategy
- Small page size (10-20) = Fast loads, more clicking
- Medium page size (50-100) = Good balance
- Large page size (200) = Slow loads, less clicking

### Animation Performance
- 📱 Use fullscreen mode for smooth playback
- 🖥️ Desktop faster than mobile
- ⚡ Reduce speed to 0.5x if stuttering
- 🔄 Close other browser tabs

## 🐛 Common Issues

### "No events found"
- ✅ Check time range (defaults to last 24h)
- ✅ Adjust filters (too restrictive?)
- ✅ Verify data exists: check `/analytics` page

### Animation doesn't play
- ✅ Try fullscreen mode
- ✅ Reduce speed to 0.5x
- ✅ Check browser console (F12)
- ✅ Try different browser

### Slow page load
- ✅ Reduce date range
- ✅ Add classification filter
- ✅ Check internet speed
- ✅ Try smaller page size

### Wrong count in pagination
- ✅ Filters may have changed
- ✅ Refresh page
- ✅ Clear browser cache
- ✅ Check API endpoint directly

## 📞 Support Resources

| Resource | Location |
|----------|----------|
| Full Docs | `/docs/TRACK_EVENTS_ENHANCEMENTS.md` |
| Database Schema | `/src/logging/schema.sql` |
| API Code | `/src/endpoint/routes/track_lifecycle.py` |
| Templates | `/src/endpoint/templates/` |
| Logs | `/data/logs/app.log` |

## 🎓 Learning Path

1. **Start**: Use basic `/track-events` page
2. **Explore**: Try different filters
3. **Analyze**: Check stats and distributions
4. **Visualize**: Click on track to see animation
5. **Deep Dive**: Use `/api/track-events` for custom analysis
6. **Automate**: Build tools using JSON endpoints

---

**Last Updated**: Feb 19, 2026 | **Version**: 2.0
