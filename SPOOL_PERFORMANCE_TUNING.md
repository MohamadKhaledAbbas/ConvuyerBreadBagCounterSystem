# Spool Processor Performance Tuning Guide

## Quick Reference: min_frame_interval_ms

The `min_frame_interval_ms` setting controls CPU throttling in FAST mode.

### Default Setting
```python
min_frame_interval_ms: float = 33.0  # ~30fps max (1000/33 = 30.3 fps)
delete_processed_segments: bool = True  # Immediately delete processed segments
```

### Performance Impact

| Setting | Max FPS | CPU Usage | Use Case |
|---------|---------|-----------|----------|
| 16.7 ms | ~60 fps | 80-95% | Maximum speed, powerful CPU |
| 25.0 ms | ~40 fps | 60-80% | Fast catch-up mode |
| **33.0 ms** | ~**30 fps** | **40-60%** | **Balanced (default)** |
| 50.0 ms | ~20 fps | 30-40% | Conservative, older hardware |
| 71.4 ms | ~14 fps | 10-20% | Match recording rate |

### Recommended Values

**For your setup (14 fps recording):**
```python
min_frame_interval_ms: float = 33.0  # ✅ Default - ~30 fps, good balance
```

**Rationale:**
- Recording rate: 14 fps (71ms per frame)
- Processing rate: ~30 fps (33ms interval)
- **Speedup: 2x faster than recording**
- CPU gets 33ms rest between frames
- Sustainable for 24/7 operation
- Segments are deleted immediately after processing

### Adjusting the Setting

**If CPU usage is too high (>80%):**
```python
min_frame_interval_ms: float = 20.0  # Increase to 20ms
```

**If you want maximum speed (powerful CPU):**
```python
min_frame_interval_ms: float = 5.0  # Decrease to 5ms
```

**To match recording rate exactly:**
```python
# Use REALTIME mode instead
playback_mode: PlaybackMode = PlaybackMode.REALTIME
# min_frame_interval_ms is ignored in REALTIME mode
```

### How It Works

```python
# In FAST mode processing loop
if self.config.playback_mode == PlaybackMode.FAST:
    if last_frame_time is not None:
        min_interval_s = self.config.min_frame_interval_ms / 1000.0
        elapsed = time.monotonic() - last_frame_time
        remaining = min_interval_s - elapsed
        
        if remaining > 0:
            time.sleep(remaining)  # CPU rest period
    
    last_frame_time = time.monotonic()
```

### Monitoring

```bash
# Watch CPU usage
top -p $(pgrep -f spool_processor)

# Monitor processing rate
tail -f data/logs/spool-processor.log | grep "segment_complete"

# Check ROS2 topic rate
ros2 topic hz /spool_image_ch_0
```

### Expected Behavior

**With 10ms throttling:**
```
Startup (catching up):
  Processing: 30-60 fps
  CPU usage: 50-70%
  Segments: Decreasing (catching up)

Steady state (caught up):
  Processing: ~14 fps (matches recording)
  CPU usage: 10-30%
  Segments: 1-3 on disk (just latest)
```

## Summary

✅ **Default 10ms setting is optimal for:**
- 14 fps recording rate
- 30-60 fps processing capability
- Reasonable CPU usage (50-70%)
- Fast catch-up (2-4x recording rate)
- Sustainable long-term operation

🎯 **No need to change unless:**
- CPU usage consistently >80% → Increase to 20ms
- Have powerful CPU and want max speed → Decrease to 5ms
- Need exact recording rate → Use REALTIME mode instead
