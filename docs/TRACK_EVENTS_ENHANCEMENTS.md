# Track Events Endpoint Enhancements

## Overview

The track-events endpoint has been significantly enhanced to provide comprehensive track lifecycle analytics with:

- **Advanced Filtering**: Multiple filtering options for precise data exploration
- **Pagination**: Efficient browsing of large datasets
- **Enhanced Statistics**: Distribution charts and recovery metrics
- **Track Animation**: Interactive visual timeline of each track's journey
- **JSON APIs**: Machine-readable endpoints for programmatic access
- **Performance Optimizations**: Batch queries and efficient database access

## New Features

### 1. Advanced Filtering

The main `/track-events` page now supports multiple filter criteria:

#### Filter Parameters

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `start_time` | ISO datetime | - | Time range start (defaults to 24h ago) |
| `end_time` | ISO datetime | - | Time range end (defaults to now) |
| `event_type` | string | `track_completed`, `track_lost`, `track_invalid` | Filter by completion status |
| `classification` | string | Any classification name | Filter by final classification result |
| `min_confidence` | float | 0.0 - 1.0 | Minimum detection confidence threshold |
| `min_duration` | float | seconds | Minimum track duration |
| `max_duration` | float | seconds | Maximum track duration |
| `entry_type` | string | `bottom_entry`, `thrown_entry`, `midway_entry` | How track entered the frame |
| `exit_direction` | string | `top`, `bottom`, `left`, `right`, `timeout` | Where/how track exited |
| `page` | int | 1+ | Pagination page number |
| `page_size` | int | 10-200 | Events per page (default: 50) |

#### Example Queries

```
# All completed tracks in the last 24 hours with classification "Wheatberry"
/track-events?event_type=track_completed&classification=Wheatberry

# Tracks lasting less than 1 second with high confidence
/track-events?max_duration=1&min_confidence=0.8

# Anomalous midway entries that were lost
/track-events?entry_type=midway_entry&event_type=track_lost

# All tracks that exited from the top
/track-events?exit_direction=top
```

### 2. Pagination

Large result sets are now paginated for better performance and UX:

- **Default page size**: 50 events per page
- **Adjustable**: 10-200 events per page via `page_size` parameter
- **Navigation**: First, Previous, Next, Last buttons
- **Total count**: Shows total events matching filters

### 3. Enhanced Statistics Dashboard

The stats section now includes:

#### Distribution Charts

1. **Top Classifications** - Bar chart showing most common classification results
2. **Duration Distribution** - Histogram showing track duration buckets:
   - 0-1s
   - 1-2s
   - 2-3s
   - 3-5s
   - 5s+

3. **Confidence Distribution** - Histogram showing detection confidence ranges:
   - <50%
   - 50-70%
   - 70-85%
   - 85-95%
   - 95-100%

#### Recovery Statistics

- **Ghost Recovered**: Tracks that were recovered after occlusion
- **Shadow Tracks**: Duplicate tracks merged with survivors

### 4. Track Lifecycle Animation

#### Animated Visualization Page

Each track now has a dedicated visualization page showing its journey:

**URL**: `/track-events/{track_id}/visualize`

**Features**:
- **SVG Canvas Animation**: Animated trajectory from entry to exit
- **Entry/Exit Markers**: Yellow circle for entry, red X for exit
- **ROI Overlays**: Dashed purple boxes for ROI collection points
- **Interactive Controls**:
  - Play/Pause button
  - Speed adjustment (0.5x - 3x)
  - Timeline scrubber for frame seeking
  - Current time / total duration display
- **Sidebar Information**:
  - Track summary (status, duration, distance)
  - Entry/exit positions
  - Recovery statistics
  - Event timeline
  - Legend explaining visual elements

#### Animation Data Flow

The animation is powered by the `/track-events/{track_id}/animation` API endpoint:

```json
{
  "track_id": 123,
  "summary": {
    "event_type": "track_completed",
    "entry": {"x": 100, "y": 200},
    "exit": {"x": 500, "y": 200},
    "entry_type": "bottom_entry",
    "exit_direction": "right",
    "duration_seconds": 2.5,
    "distance_pixels": 400,
    "classification": "Wheatberry",
    "created_at": "2026-02-19T14:30:00",
    "ended_at": "2026-02-19T14:30:02.5"
  },
  "position_history": [[100, 200], [150, 200], [200, 200], ...],
  "roi_events": [
    {
      "timestamp": "2026-02-19T14:30:01",
      "roi_index": 0,
      "bbox": {"x1": 90, "y1": 190, "x2": 110, "y2": 210},
      "quality_score": 0.95
    }
  ],
  "classification_events": [
    {
      "timestamp": "2026-02-19T14:30:01.5",
      "roi_index": 0,
      "class_name": "Wheatberry",
      "confidence": 0.98
    }
  ],
  "lifecycle_events": [
    {
      "timestamp": "2026-02-19T14:30:00",
      "step_type": "track_created"
    },
    {
      "timestamp": "2026-02-19T14:30:02.5",
      "step_type": "track_completed"
    }
  ],
  "occlusion_events": [],
  "merge_events": [],
  "animation": {
    "keyframes": [...],
    "total_distance": 400,
    "frame_count": 50,
    "suggested_duration_ms": 2500
  }
}
```

### 5. JSON API Endpoints

For programmatic access to track event data:

#### `/api/track-events` - Paginated Events

**Parameters**: Same as `/track-events` page

**Returns**: Paginated list of events with parsed JSON fields

```json
{
  "events": [
    {
      "track_id": 123,
      "event_type": "track_completed",
      "classification": "Wheatberry",
      "position_history": [[x1, y1], [x2, y2], ...],
      ...
    }
  ],
  "pagination": {
    "page": 1,
    "page_size": 50,
    "total_count": 1000,
    "total_pages": 20
  }
}
```

#### `/api/track-events/stats` - Enhanced Statistics

**Returns**: Comprehensive analytics data

```json
{
  "time_range": {
    "start": "2026-02-18T14:30:00",
    "end": "2026-02-19T14:30:00"
  },
  "stats": {
    "total": 1000,
    "by_type": {
      "track_completed": {
        "count": 800,
        "avg_duration": 2.3,
        "avg_distance": 400,
        "avg_confidence": 0.92
      },
      "track_lost": {
        "count": 150,
        ...
      },
      "track_invalid": {
        "count": 50,
        ...
      }
    },
    "by_classification": [
      {"classification": "Wheatberry", "count": 400, "avg_conf": 0.95},
      {"classification": "Brown_Orange", "count": 300, "avg_conf": 0.93},
      ...
    ],
    "by_entry_type": {
      "bottom_entry": 900,
      "thrown_entry": 80,
      "midway_entry": 20
    },
    "by_exit_direction": {
      "bottom": 800,
      "right": 150,
      "timeout": 50
    },
    "recovery_stats": {
      "recovered_tracks": 45,
      "total_recoveries": 67,
      "total_shadows": 23
    },
    "duration_histogram": {
      "0-1s": 200,
      "1-2s": 400,
      "2-3s": 300,
      "3-5s": 80,
      "5s+": 20
    },
    "confidence_histogram": {
      "< 50%": 10,
      "50-70%": 30,
      "70-85%": 100,
      "85-95%": 400,
      "95-100%": 460
    }
  }
}
```

#### `/track-events/{track_id}` - Full Track Lifecycle

**Returns**: Complete track summary and all lifecycle detail steps

```json
{
  "summary": {
    "track_id": 123,
    "event_type": "track_completed",
    "entry_x": 100,
    "entry_y": 200,
    "exit_x": 500,
    "exit_y": 200,
    ...
  },
  "details": [
    {
      "id": 1,
      "track_id": 123,
      "step_type": "track_created",
      "timestamp": "2026-02-19T14:30:00"
    },
    {
      "id": 2,
      "step_type": "roi_collected",
      "bbox_x1": 90,
      "bbox_y1": 190,
      "bbox_x2": 110,
      "bbox_y2": 210,
      "quality_score": 0.95
    },
    ...
  ]
}
```

#### `/track-events/{track_id}/animation` - Animation Data

**Returns**: Structured animation data for rendering

See JSON example under "Track Lifecycle Animation" section above.

## Performance Enhancements

### Database Optimizations

1. **Batch Queries**: `get_track_event_details_for_tracks()` fetches all details in a single query
   - Before: N+1 queries (one per track)
   - After: Single query with `IN` clause

2. **Composite Indexes**: Enhanced database schema with indexes on:
   - `track_events(timestamp, event_type)`
   - `track_events(track_id)`
   - `track_event_details(track_id, step_type)`

3. **Pagination**: Results limited to 50-200 per page vs. unbounded results

### Query Optimization

- **WHERE clause construction**: Dynamic filters only applied when specified
- **Prepared statements**: All queries use parameterized statements to prevent SQL injection
- **Efficient aggregation**: Statistics computed in database layer, not in Python

### Frontend Optimizations

- **Lazy loading**: Details rows expand on demand
- **Pagination**: Large datasets split across pages
- **Canvas optimization**: Animation uses requestAnimationFrame for smooth playback
- **Responsive design**: Adapts to mobile/tablet screens

## Usage Examples

### Example 1: Find all lost tracks in the last hour

```
GET /track-events?event_type=track_lost&start_time=2026-02-19T13:30:00&end_time=2026-02-19T14:30:00
```

### Example 2: Analyze classification accuracy for Wheatberry

```
GET /api/track-events/stats?classification=Wheatberry&start_time=2026-02-19T00:00:00&end_time=2026-02-19T23:59:59
```

### Example 3: Visualize a track's journey

```
# View animated timeline
GET /track-events/12345/visualize

# Get raw animation data
GET /track-events/12345/animation
```

### Example 4: Export data for external analysis

```javascript
// Fetch all completed tracks with classification
fetch('/api/track-events?event_type=track_completed&classification=Wheatberry&page_size=200')
  .then(r => r.json())
  .then(data => {
    console.log(`Found ${data.pagination.total_count} events`);
    // Process events
  });
```

## Architecture Changes

### Repository Layer (`track_lifecycle_repository.py`)

**New Methods**:
- `get_track_events_page()`: Paginated queries with advanced filtering
- `get_enhanced_stats()`: Comprehensive statistics with histograms
- `get_track_animation_data()`: Animation-specific data extraction
- `get_track_event_details_for_tracks()`: Batch detail queries
- `get_distinct_classifications()`: Dropdown options

**Benefits**:
- Separation of concerns: database queries isolated
- Reusable filter logic
- Efficient batch operations

### Service Layer (`track_lifecycle_service.py`)

**New Methods**:
- `get_lifecycle_data()`: Enhanced with pagination and filters
- `get_events_json()`: API-specific data serialization
- `get_track_animation()`: Animation data preparation with keyframes

**Benefits**:
- Business logic abstraction
- Data enrichment and transformation
- Template-agnostic data preparation

### Routes Layer (`track_lifecycle.py`)

**New Endpoints**:
- `GET /api/track-events`: JSON API for events
- `GET /api/track-events/stats`: JSON API for statistics
- `GET /track-events/{track_id}/animation`: Animation data
- `GET /track-events/{track_id}/visualize`: Animation visualization

**Benefits**:
- Backward compatible: existing endpoints unchanged
- Clear separation of concerns: HTML vs JSON
- Comprehensive API coverage

### Templates

**Updated**:
- `track_events.html`: Added filters, pagination, stats charts, animation links

**New**:
- `track_visualization.html`: Interactive animation viewer with controls

## Database Schema

No schema changes required. Existing tables are fully utilized:

- `track_events`: Summary information for each track
- `track_event_details`: Per-step lifecycle events (ROI collection, classification, etc.)
- Existing indexes support efficient filtering

## Backward Compatibility

All enhancements are backward compatible:
- Existing `/track-events` page still works with no parameters
- New filters are optional
- Old bookmarks and links continue to function
- No breaking changes to data structures

## Migration Guide

### For Dashboard Users

No action required. The enhanced UI is automatically available:

1. **Basic Usage**: Continue using `/track-events` as before
2. **Advanced Filtering**: Use new filter dropdowns to explore data
3. **Track Animation**: Click "Track ID" link in table to view animation

### For API Consumers

New API endpoints available alongside existing ones:

```python
# Old approach (still works)
response = requests.get('/track-events/123')
data = response.json()

# New approach (recommended)
response = requests.get('/api/track-events?page=1&page_size=100')
data = response.json()
events = data['events']
pagination = data['pagination']
```

### For Analytics Integration

Use new `/api/track-events/stats` endpoint for analytics:

```python
stats = requests.get('/api/track-events/stats').json()
completion_rate = stats['stats']['by_type']['track_completed']['count']
avg_duration = stats['stats']['by_type']['track_completed']['avg_duration']
```

## Future Enhancements

Potential improvements for future versions:

1. **Advanced Analytics**: Charts and graphs for KPI trending
2. **Export to CSV/Excel**: Bulk data export
3. **Multi-track Comparison**: Side-by-side trajectory comparison
4. **Heatmaps**: Position frequency heatmaps showing common paths
5. **Alerts**: Automated detection of anomalous tracks
6. **Dashboard Widgets**: Embeddable widgets for monitoring
7. **Real-time Updates**: WebSocket support for live streaming
8. **Custom Reports**: Template-based report generation

## Troubleshooting

### Slow Pagination Queries

**Symptom**: Page loads slowly with large time ranges

**Solution**:
- Reduce time range
- Use more specific filters (classification, entry_type)
- Check database indexes exist: `PRAGMA index_list(track_events);`

### Animation Playback Issues

**Symptom**: Animation stutters or doesn't play smoothly

**Solution**:
- Reduce animation speed via speed slider
- Try fullscreen mode
- Check browser performance (F12 > Performance tab)
- Ensure hardware acceleration enabled in browser

### Missing Statistics

**Symptom**: Distribution histograms don't appear

**Solution**:
- Verify time range has events
- Check filter criteria (may exclude all results)
- Refresh page to reload data

## Support

For issues or questions:
- Check logs: `tail -f data/logs/app.log`
- Review schema: `sqlite3 data/db/app.db ".schema track_events"`
- Test endpoints directly: `curl /api/track-events/stats`
