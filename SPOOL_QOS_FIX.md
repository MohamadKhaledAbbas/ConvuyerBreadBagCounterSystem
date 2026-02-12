# Spool Recorder QoS Fix - Root Cause Analysis

## Problem
The spool_recorder_node was NOT receiving frames from hobot_rtsp_client despite:
- Topic `/rtsp_image_ch_0` was publishing at 13-14 fps
- Both nodes were using correct message type `img_msgs/msg/H26XFrame`
- Subscription was established

## Root Cause: QoS Incompatibility

**Publisher (hobot_rtsp_client):**
```
Reliability: RELIABLE
```

**Subscriber (spool_recorder_node) - BEFORE FIX:**
```
Reliability: BEST_EFFORT  ❌
```

### ROS2 QoS Compatibility Rules:
1. **BEST_EFFORT subscriber** → Can ONLY receive from **BEST_EFFORT publisher**
2. **RELIABLE subscriber** → Can receive from BOTH **RELIABLE** and **BEST_EFFORT** publishers

Since hobot_rtsp_client publishes with RELIABLE, the subscriber MUST use RELIABLE.

## Solution Applied

Changed both spool nodes to use RELIABLE QoS:

### 1. spool_recorder_node.py
```python
qos = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,  # Changed from BEST_EFFORT
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=self.config.qos_depth
)
```

### 2. spool_processor_node.py
```python
qos = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,  # Changed from BEST_EFFORT
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=self.config.qos_depth
)
```

## Complete Pipeline QoS Settings

```
hobot_rtsp_client (RELIABLE)
    ↓ /rtsp_image_ch_0
spool_recorder_node (RELIABLE) ✅
    ↓ disk segments
spool_processor_node (RELIABLE) ✅
    ↓ /spool_image_ch_0
hobot_codec
    ↓ /nv12_images
main.py (FrameServer)
```

## Testing on RDK

After syncing the updated code:

```bash
# 1. Restart spool_recorder
pkill -f spool_recorder
python3 -m src.spool.spool_recorder_node > data/logs/spool-recorder.log 2>&1 &

# 2. Check for "First frame received" message
tail -f data/logs/spool-recorder.log

# 3. Verify segments are being written
ls -lh /tmp/spool/

# 4. Check QoS match
ros2 topic info /rtsp_image_ch_0 -v
# Both Publisher and Subscriber should now show "Reliability: RELIABLE"
```

## Expected Results

- ✅ spool_recorder receives frames immediately
- ✅ Segments appear in `/tmp/spool/seg_*.bin`
- ✅ spool_processor can read and publish to `/spool_image_ch_0`
- ✅ hobot_codec receives frames and publishes to `/nv12_images`
- ✅ main.py receives NV12 frames for detection

## Files Modified

1. `src/spool/spool_recorder_node.py` - QoS changed to RELIABLE + timestamp conversion fix
2. `src/spool/spool_processor_node.py` - QoS changed to RELIABLE + timestamp creation fix + REALTIME pacing
3. `src/frame_source/Ros2FrameServer.py` - Message import fixed to img_msgs
4. `src/ros2/IPC.py` - Added ROS2 context initialization
5. `src/app/ConveyorCounterApp.py` - Added is_development check and ROS2 init

## Additional Fix: Timestamp Conversion

### Issue 1: spool_recorder_node.py
After the QoS fix, frames were being received but there was an error:
```
'>' not supported between instances of 'Time' and 'int'
```

**Root Cause:** The code was comparing ROS2 `Time` objects (`msg.dts`, `msg.pts`) directly with integers.

**Solution:** Convert ROS2 Time objects to nanoseconds before comparison:
```python
# Before:
if hasattr(msg, 'dts') and msg.dts > 0:  # ❌ Comparing Time with int

# After:
if hasattr(msg, 'dts'):
    if hasattr(msg.dts, 'sec'):
        dts_ns = msg.dts.sec * 1_000_000_000 + msg.dts.nanosec  # ✅ Convert to int
```

### Issue 2: spool_processor_node.py (Timestamp Creation)
Spool processor was failing to publish frames with error:
```
The 'dts' field must be a sub message of type 'Time'
```

**Root Cause:** The code was setting `msg.dts` and `msg.pts` as integers instead of ROS2 `Time` objects.

**Solution:** Create ROS2 Time objects for timestamps:
```python
# Before:
msg.dts = record.dts_ns  # ❌ Integer

# After:
from builtin_interfaces.msg import Time
dts_time = Time()
dts_time.sec = record.dts_sec
dts_time.nanosec = record.dts_nsec
msg.dts = dts_time  # ✅ Time object
```

### Issue 3: spool_processor_node.py (FPS Too Fast)
Spool processor was publishing at 60 fps instead of matching the recording rate (~13-14 fps).

**Root Cause:** Adaptive pacing mode was speeding up to max_fps when queue was empty.

**Solution:** Changed to REALTIME mode with timestamp-based pacing:
```python
# Changed default mode from ADAPTIVE to REALTIME
playback_mode: PlaybackMode = PlaybackMode.REALTIME

# Added timestamp-based pacing in _process_loop:
if self.config.playback_mode == PlaybackMode.REALTIME:
    # Calculate inter-frame delay from actual timestamps
    elapsed_since_first_ns = frame_timestamp_ns - first_frame_timestamp_ns
    elapsed_since_first_s = elapsed_since_first_ns / 1_000_000_000.0
    wall_time_elapsed = time.monotonic() - first_frame_wall_time
    time_to_wait = elapsed_since_first_s - wall_time_elapsed
    if time_to_wait > 0:
        time.sleep(time_to_wait)
```

This ensures the processor publishes frames at the same rate they were recorded (~13-14 fps).

