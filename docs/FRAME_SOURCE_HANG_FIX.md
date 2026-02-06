# Frame Source Hang Issue - Resolution

## Problem Analysis

The system was experiencing hang issues due to:

1. **No Backpressure**: Unbounded queue caused infinite memory growth when consumer (detector/tracker) was slower than producer (frame reader)
2. **No Graceful Shutdown**: Missing proper stop mechanism made it impossible to interrupt cleanly
3. **CPU Spinning**: No frame pacing caused 100% CPU usage with tight loops
4. **IO Blocking**: Queue operations could block indefinitely without timeout

## Root Causes (Comparison with v1)

After reviewing the [v1 OpenCvFrameSource.py](https://github.com/MohamadKhaledAbbas/BreadBagCounterSystem/blob/main/src/frame_source/OpenCvFrameSource.py), the key differences were:

### V1 Had:
- ✓ Proper frame pacing matching video FPS
- ✓ Simple blocking queue with bounded size
- ✓ CPU-friendly design

### V2 Had:
- ✗ Unbounded queue (queue_size=0)
- ✗ No timeout on queue operations
- ✗ No chunked sleeps for responsive shutdown
- ✗ Missing proper stop event mechanism

## Solutions Implemented

### 1. **Backpressure with Bounded Queue**

**File**: `src/frame_source/OpenCvFrameSource.py`

```python
# Before (v2):
self.queue = queue.Queue(maxsize=0)  # Unbounded = memory overflow
self.queue.put((frame, inter_frame_ms))  # Blocking without timeout

# After (Fixed):
self.queue = queue.Queue(maxsize=30)  # Bounded queue
try:
    self.queue.put((frame, inter_frame_ms), block=True, timeout=0.5)
except queue.Full:
    # Consumer slow, log and retry
    logger.warning("Queue full, waiting for consumer")
    time.sleep(0.01)
```

**Benefits**:
- Prevents unbounded memory growth
- Producer waits when consumer is slow (natural flow control)
- No frame drops - all frames processed

### 2. **Graceful Shutdown with Threading.Event**

```python
# Added:
self._stopped = threading.Event()

# In _read_frames loop:
while self.running and not self._stopped.is_set():
    # Check frequently
    if self._stopped.is_set():
        break

# In cleanup:
self._stopped.set()
self.read_thread.join(timeout=3.0)
```

**Benefits**:
- Can interrupt at any time
- No need to force-kill process
- Clean resource release

### 3. **CPU Management with Frame Pacing**

```python
# Calculate proper delay based on source FPS
self.frame_interval = 1.0 / self.source_fps

# Use chunked sleeps for responsive interruption
chunks = int(sleep_time / 0.01)  # 10ms chunks
for _ in range(chunks):
    if not self.running or self._stopped.is_set():
        break
    time.sleep(0.01)
```

**Benefits**:
- CPU gets rest between frames
- Can check stop signal frequently (every 10ms)
- Maintains proper video playback speed

### 4. **Configuration Options**

**File**: `src/config/settings.py`

```python
# Added configurable parameters:
frame_queue_size: int = int(os.getenv("FRAME_QUEUE_SIZE", "30"))
frame_target_fps: Optional[float] = None  # None = use source FPS
```

**Usage**:
```bash
# Set queue size via environment variable
set FRAME_QUEUE_SIZE=50

# Or modify settings.py directly
```

## Files Modified

1. **src/frame_source/OpenCvFrameSource.py**
   - Added `_stopped` threading.Event
   - Bounded queue with timeout
   - Chunked sleeps in frame pacing
   - Better error handling
   - Queue full event tracking

2. **src/config/settings.py**
   - Added `frame_queue_size` config
   - Added `frame_target_fps` config

3. **src/frame_source/FrameSourceFactory.py**
   - Pass `queue_size` parameter

4. **src/app/ConveyorCounterApp.py**
   - Pass config parameters to frame source

## Testing

Run the test script to verify fixes:

```bash
python test_frame_source_fixes.py
```

Tests verify:
1. Backpressure works with slow consumer
2. Shutdown is responsive (< 5 seconds)
3. Frame pacing maintains proper FPS

## Expected Behavior

### Before Fix:
- ❌ System hangs after some time
- ❌ Memory grows unbounded
- ❌ Cannot stop gracefully (need to kill process)
- ❌ CPU at 100%

### After Fix:
- ✅ All frames processed without hang
- ✅ Memory stays bounded (30 frames × frame size)
- ✅ Can stop anytime with Ctrl+C
- ✅ CPU usage normal (proportional to processing speed)

## Performance Notes

### Queue Size Tuning

- **Small (10-20)**: More backpressure, less memory, may slow down if consumer very slow
- **Medium (30-50)**: Good balance (recommended)
- **Large (100+)**: More buffering, higher memory, less backpressure

### Frame Rate

- **None (default)**: Use source video FPS (best for offline processing)
- **Custom (e.g., 15)**: Limit processing rate (useful for real-time systems)

## Monitoring

Check logs for these indicators:

```
[OpenCVFrameSource] Queue full events: <count>
```

- **0 events**: Consumer keeping up perfectly
- **Few events**: Occasional slowdown (normal)
- **Many events**: Consumer consistently slow (may need optimization)

## Additional Improvements

If issues persist, consider:

1. **Increase queue size**: `FRAME_QUEUE_SIZE=50`
2. **Reduce processing**: Lower detection confidence, skip frames
3. **Disable display**: Display adds latency
4. **Profile code**: Find bottlenecks in detector/tracker

## References

- [v1 OpenCvFrameSource.py](https://github.com/MohamadKhaledAbbas/BreadBagCounterSystem/blob/main/src/frame_source/OpenCvFrameSource.py)
- Python threading.Event: https://docs.python.org/3/library/threading.html#event-objects
- Queue with timeout: https://docs.python.org/3/library/queue.html
