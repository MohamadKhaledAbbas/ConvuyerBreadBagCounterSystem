# Track Events - Testing & Validation Guide

## 🧪 Quick Test Checklist

Run through these tests to verify all features work correctly.

## Part 1: Main Page & Filtering

### Test 1.1: Basic Page Load
```
1. Navigate to: http://localhost:8000/track-events
2. Expected: Page loads with stats dashboard
3. Verify:
   - ✓ Total Events counter appears
   - ✓ Completed/Lost/Invalid cards show counts
   - ✓ Success Rate percentage displays
   - ✓ Events table shows tracks (if data exists)
```

### Test 1.2: Event Type Filter
```
1. Open: /track-events
2. Change "Event Type" dropdown to "Completed"
3. Click "Filter"
4. Expected:
   - ✓ Only track_completed events shown
   - ✓ URL changes to ?event_type=track_completed
   - ✓ Stats update to show only completed stats
```

### Test 1.3: Classification Filter
```
1. Open: /track-events
2. Select a classification (e.g., "Wheatberry")
3. Click "Filter"
4. Expected:
   - ✓ Only tracks with that classification shown
   - ✓ Classification filter value preserved in dropdown
   - ✓ Stats recalculate for filtered subset
```

### Test 1.4: Confidence Filter
```
1. Open: /track-events
2. Enter "0.85" in "Min Confidence"
3. Click "Filter"
4. Expected:
   - ✓ Only tracks with avg_confidence >= 0.85 shown
   - ✓ Confidence bars in table show high values
```

### Test 1.5: Duration Filters
```
1. Open: /track-events
2. Enter min_duration=0.5, max_duration=2
3. Click "Filter"
4. Expected:
   - ✓ Only tracks between 0.5-2 seconds shown
   - ✓ Duration column shows values in that range
```

### Test 1.6: Entry Type Filter
```
1. Open: /track-events
2. Select "entry_type=midway_entry"
3. Click "Filter"
4. Expected:
   - ✓ Only midway entries shown
   - ✓ Useful for finding suspicious partial tracks
```

### Test 1.7: Exit Direction Filter
```
1. Open: /track-events
2. Select "exit_direction=top"
3. Click "Filter"
4. Expected:
   - ✓ Only tracks that exited from top shown
   - ✓ Useful for anomaly detection
```

### Test 1.8: Reset Filters
```
1. Apply multiple filters
2. Click "Reset" link
3. Expected:
   - ✓ All filters cleared
   - ✓ Returns to default 24-hour time range
   - ✓ Shows all event types
```

### Test 1.9: Combined Filters
```
1. Filter by: event_type=track_lost AND entry_type=midway_entry
2. Expected:
   - ✓ Shows only lost tracks that started midway
   - ✓ Helps identify specific problem cases
```

## Part 2: Statistics Dashboard

### Test 2.1: Basic Stats Cards
```
1. Open: /track-events (no filters)
2. Verify stats cards:
   - ✓ Total Events = sum of all type counts
   - ✓ Completed + Lost + Invalid ≤ Total
   - ✓ Success Rate = Completed / Total * 100%
```

### Test 2.2: Top Classifications Chart
```
1. Open: /track-events
2. Look for "Top Classifications" chart
3. Expected:
   - ✓ Shows most common classifications
   - ✓ Counts add up to total tracks
   - ✓ Bars sorted descending by count
```

### Test 2.3: Duration Distribution Chart
```
1. Open: /track-events
2. Look for "Duration Distribution" chart
3. Expected:
   - ✓ Shows buckets: 0-1s, 1-2s, 2-3s, 3-5s, 5s+
   - ✓ Counts add up to total
   - ✓ Visual bar length matches value
```

### Test 2.4: Confidence Distribution Chart
```
1. Open: /track-events
2. Look for "Confidence Distribution" chart
3. Expected:
   - ✓ Shows buckets: <50%, 50-70%, 70-85%, 85-95%, 95-100%
   - ✓ Most tracks in high confidence buckets
   - ✓ Distribution shows system quality
```

## Part 3: Pagination

### Test 3.1: First Page Load
```
1. Open: /track-events
2. Expected:
   - ✓ Shows "Page 1 of N"
   - ✓ Only 50 events shown (or configured page_size)
   - ✓ "Previous" and "First" buttons disabled
```

### Test 3.2: Next Page Navigation
```
1. On page 1, click "Next" button
2. Expected:
   - ✓ URL changes to ?page=2
   - ✓ New events displayed
   - ✓ Shows "Page 2 of N"
   - ✓ First button now enabled
```

### Test 3.3: Last Page Navigation
```
1. Click "Last" button
2. Expected:
   - ✓ Goes to last page
   - ✓ "Next" button disabled
   - ✓ Fewer items if not exact multiple
```

### Test 3.4: Page Size Query Parameter
```
1. Open: /track-events?page_size=100
2. Expected:
   - ✓ Shows 100 events instead of default 50
   - ✓ Fewer total pages
   - ✓ Page size honored with page navigation
```

## Part 4: Track Visualization Animation

### Test 4.1: Animation Link
```
1. Open: /track-events
2. Click on a Track ID link (e.g., "T123")
3. Expected:
   - ✓ Opens new tab/window
   - ✓ Shows animation visualization page
   - ✓ Title shows "Track #123"
```

### Test 4.2: Animation Canvas Loads
```
1. On animation page
2. Expected:
   - ✓ Canvas shows track trajectory
   - ✓ Yellow circle marks entry point
   - ✓ Red X marks exit point
   - ✓ Path line shows route traveled
```

### Test 4.3: Play/Pause Control
```
1. Click "Play" button
2. Expected:
   - ✓ Animation starts moving
   - ✓ Blue dot moves along path
   - ✓ Button changes to "Pause"
   - ✓ Timeline progresses
3. Click "Pause"
4. Expected:
   - ✓ Animation stops
   - ✓ Button changes back to "Play"
```

### Test 4.4: Speed Control
```
1. Set speed slider to 0.5x
2. Click "Play"
3. Expected:
   - ✓ Animation plays slower
   - ✓ Takes longer to complete
4. Set speed to 3x
5. Expected:
   - ✓ Animation plays faster
   - ✓ Completes quickly
```

### Test 4.5: Timeline Scrubber
```
1. Click halfway along the timeline
2. Expected:
   - ✓ Blue dot jumps to middle of path
   - ✓ Time label updates
   - ✓ Progress bar shows current position
3. Click near end of timeline
4. Expected:
   - ✓ Dot jumps to near end
   - ✓ Almost at exit point
```

### Test 4.6: Reset Button
```
1. Play animation partway
2. Click "Reset"
3. Expected:
   - ✓ Resets to start
   - ✓ Stops playing
   - ✓ Time shows 0:00
   - ✓ Dot back at entry
```

### Test 4.7: Sidebar Information
```
1. On animation page, check sidebar
2. Expected:
   - ✓ Status badge (Completed/Lost/Invalid)
   - ✓ Duration display
   - ✓ Distance traveled
   - ✓ Entry/Exit coordinates
   - ✓ Classification if available
3. Scroll down
4. Expected:
   - ✓ Recovery stats show ghost recovery count
   - ✓ Events list shows lifecycle steps
   - ✓ Legend explains visual symbols
```

### Test 4.8: Fullscreen Mode
```
1. Click "Fullscreen" button
2. Expected:
   - ✓ Canvas expands to full screen
   - ✓ Controls still visible
   - ✓ Better viewing experience
3. Press ESC
4. Expected:
   - ✓ Returns to normal view
```

## Part 5: JSON APIs

### Test 5.1: Events API
```bash
curl 'http://localhost:8000/api/track-events?event_type=track_completed&page=1&page_size=10'
```
Expected:
```json
{
  "events": [
    {
      "track_id": 123,
      "event_type": "track_completed",
      "classification": "Wheatberry",
      ...
    }
  ],
  "pagination": {
    "page": 1,
    "page_size": 10,
    "total_count": 500,
    "total_pages": 50
  }
}
```

### Test 5.2: Stats API
```bash
curl 'http://localhost:8000/api/track-events/stats?start_time=2026-02-19T00:00:00&end_time=2026-02-19T23:59:59'
```
Expected:
```json
{
  "time_range": {...},
  "stats": {
    "total": 1000,
    "by_type": {...},
    "by_classification": [...],
    "duration_histogram": {...},
    "confidence_histogram": {...},
    "recovery_stats": {...}
  }
}
```

### Test 5.3: Animation Data API
```bash
curl 'http://localhost:8000/track-events/123/animation' | jq '.animation.suggested_duration_ms'
```
Expected: A number in milliseconds (e.g., 2500)

### Test 5.4: Track Lifecycle API
```bash
curl 'http://localhost:8000/track-events/123' | jq '.details | length'
```
Expected: A number representing lifecycle steps (e.g., 15)

## Part 6: Edge Cases

### Test 6.1: Invalid Track ID
```
1. Open: /track-events/999999/visualize
2. Expected: 404 error - Track not found
```

### Test 6.2: Invalid Date Range
```
1. Set start_time after end_time
2. Click Filter
3. Expected: 422 error - Start time must be before end time
```

### Test 6.3: Empty Time Range
```
1. Set time range with no data
2. Click Filter
3. Expected:
   - ✓ Empty state message shown
   - ✓ "No track events found"
   - ✓ Stats show zeros
```

### Test 6.4: Large Page Size
```
1. Open: /track-events?page_size=1000
2. Expected: Capped to 200 maximum
```

### Test 6.5: Invalid Confidence Value
```
1. Enter confidence = 2.0
2. Click Filter
3. Expected: Validation error or capped to 1.0
```

## Part 7: Performance Tests

### Test 7.1: Large Date Range
```
1. Set date range to 30 days
2. Expected:
   - ✓ Page loads within 2 seconds
   - ✓ Shows pagination controls
   - ✓ Stats calculated quickly
```

### Test 7.2: Many Filters
```
1. Apply 5+ filters simultaneously
2. Expected:
   - ✓ Query completes in <1 second
   - ✓ Correct subset returned
```

### Test 7.3: Animation Playback
```
1. Open animation for track with 500+ positions
2. Expected:
   - ✓ Smooth playback at 1x speed
   - ✓ No stuttering at 3x speed
   - ✓ Scrubber responsive
```

## Part 8: Browser Compatibility

### Test 8.1: Chrome/Edge
```
- ✓ All animations smooth
- ✓ Canvas renders correctly
- ✓ Responsive design works
```

### Test 8.2: Firefox
```
- ✓ All features work
- ✓ CSS gradients display
- ✓ JavaScript executes without errors
```

### Test 8.3: Mobile (iOS/Android)
```
- ✓ Pages responsive
- ✓ Touch controls work
- ✓ Readable on small screens
```

## Part 9: Data Validation

### Test 9.1: Position History Parsing
```
1. Get animation data: /track-events/{id}/animation
2. Verify position_history is valid JSON array
3. Each element is [x, y] coordinate pair
```

### Test 9.2: Event Timestamps
```
1. Get /track-events (any page)
2. Verify all timestamps are ISO 8601 format
3. created_at ≤ timestamp for each track
```

### Test 9.3: Statistics Consistency
```
1. Get stats: /api/track-events/stats
2. Verify: sum(by_type counts) = total
3. Verify: sum(histogram counts) = total
```

## Part 10: Documentation Validation

### Test 10.1: Read Documentation
```
1. Open: docs/TRACK_EVENTS_ENHANCEMENTS.md
2. Expected:
   - ✓ Clear explanation of all features
   - ✓ API examples work as shown
   - ✓ Code snippets are accurate
```

### Test 10.2: Quick Reference
```
1. Open: docs/TRACK_EVENTS_QUICK_REF.md
2. Expected:
   - ✓ Common queries work
   - ✓ Filter examples produce correct results
   - ✓ Tips are helpful and accurate
```

## 📋 Test Results Template

```
Feature: [Name]
Test Date: [Date]
Tester: [Name]
Status: [PASS/FAIL]

Passed Tests:
- [x] Test 1
- [x] Test 2
- [ ] Test 3

Failed Tests:
- [ ] Test 3 (Issue: ...)

Notes:
```

## 🐛 Issue Reporting

If tests fail, report with:
1. **Test ID**: (e.g., 4.3)
2. **Expected**: What should happen
3. **Actual**: What actually happened
4. **Browser**: Chrome 120, Firefox 121, etc.
5. **Steps**: How to reproduce
6. **Screenshots**: If visual issue
7. **Error**: Console errors (F12 > Console)

## ✅ Sign-Off

- [ ] All tests passed
- [ ] Documentation verified
- [ ] No critical issues
- [ ] Ready for deployment

**Date**: ___________
**Tester**: ___________
**Comments**: ___________

---

**Testing Guide Version**: 1.0
**Last Updated**: February 19, 2026
