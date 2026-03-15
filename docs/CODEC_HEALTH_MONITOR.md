# Codec Health Monitor

## Overview

The Codec Health Monitor automatically detects and recovers from `hobot_codec` VPU decoder stalls. This addresses a known issue on RDK platforms where the hardware H.264 decoder can enter a hung state.

## Root Cause

The RDK's `hobot_codec` uses the hardware VPU for H.264 decoding. The VPU can enter a stalled state when:

1. **VPU Close/Reopen Cycle**: If the VPU is closed and reopened (visible in `dmesg` as `vpu_close_instance` / `vpu_open_instance`), the decoder may lose its internal state.

2. **Missing IDR Keyframe**: After a restart or state loss, the decoder needs an IDR (I-frame/keyframe) to reinitialize. If it restarts mid-GOP (between keyframes), it receives P/B frames that reference a keyframe it never saw.

3. **Resource Contention**: If another process briefly claims the VPU hardware, the existing decode session is terminated.

When stalled, the `hobot_codec` node:
- Remains registered in ROS2 (`ros2 node list` shows it)
- Continues receiving input frames from `/spool_image_ch_0`
- Produces **zero output** on `/nv12_images`
- Reports no errors (it's waiting for decode data that will never be valid)

## Symptoms

```bash
# Input topic is alive
$ ros2 topic hz /spool_image_ch_0
average rate: 18.394

# Output topic is dead
$ ros2 topic hz /nv12_images
# (no output — 0 Hz)

# Node appears healthy
$ ros2 node list
/hobot_codec
/spool_processor_node
/frame_server
```

Application logs will show:
```
[Snapshot] Timed out after 3.03s (58 polls). old_ts=X, final_ts=X, DB flag now=1.
Likely cause: main.py is not running or not processing frames.
```

## Solution

The monitor checks `/nv12_images` periodically and restarts `hobot_codec` when it stops publishing.

### Integration Options

#### Option 1: Integrated with main.py (Recommended)

Add to your `main.py`:

```python
from src.ros2.codec_health_monitor import CodecHealthMonitor, MonitorConfig

# During initialization
monitor = CodecHealthMonitor(
    config=MonitorConfig(
        topic="/nv12_images",
        message_timeout_sec=10.0,
        check_interval_sec=15.0,
        failure_threshold=2,
    )
)
monitor.start()

# During shutdown
monitor.stop()
```

#### Option 2: Standalone systemd Service

```bash
# Copy service file
sudo cp systemd/codec-health-monitor.service /etc/systemd/system/

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable codec-health-monitor.service
sudo systemctl start codec-health-monitor.service

# Check status
sudo systemctl status codec-health-monitor.service
journalctl -u codec-health-monitor -f
```

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `CODEC_MONITOR_TOPIC` | `/nv12_images` | Topic to monitor |
| `CODEC_MONITOR_TIMEOUT` | `10` | Seconds to wait for a message |
| `CODEC_MONITOR_INTERVAL` | `15` | Seconds between health checks |
| `CODEC_MONITOR_THRESHOLD` | `2` | Consecutive failures before restart |
| `CODEC_MONITOR_COOLDOWN` | `30` | Seconds to wait after restart |
| `CODEC_MONITOR_MAX_RESTARTS` | `5` | Max restarts per hour (circuit breaker) |
| `CODEC_MONITOR_ENABLE_RESTART` | `true` | Enable automatic restart |
| `CODEC_MONITOR_VERBOSE` | `false` | Enable verbose logging |

## Monitoring

### Check Health Status

```bash
# View recent logs
journalctl -u codec-health-monitor --since "10 min ago"

# Check for restarts
grep "RESTART" /path/to/app/logs/*.log
```

### Manual Restart

If you need to manually trigger a codec restart:

```bash
# Kill the codec process (launch system should respawn it)
pkill -9 -f hobot_codec
```

Or via the monitor API:

```python
from src.ros2.codec_health_monitor import CodecHealthMonitor
monitor = CodecHealthMonitor()
monitor.force_restart()
```

## Diagnostics

### Check VPU Status

```bash
# Recent VPU events
dmesg | grep -i vpu | tail -10

# Check for close/reopen cycles
dmesg | grep -i "vpu_close_instance\|vpu_open_instance"
```

### Check Process Status

```bash
# Is hobot_codec running?
ps aux | grep hobot_codec

# Process details
cat /proc/$(pgrep -f hobot_codec)/status | grep -i "state\|threads"
```

### Check Topic Flow

```bash
# Input (should show ~17-18 Hz)
ros2 topic hz /spool_image_ch_0

# Output (should show ~17-18 Hz, 0 = stalled)
ros2 topic hz /nv12_images

# Topic details
ros2 topic info /nv12_images --verbose
```

## Rate Limiting

The monitor includes a circuit breaker to prevent restart loops:

- **Max 5 restarts per hour**: If exceeded, restarts are blocked and an error is logged
- **30-second cooldown**: After each restart, health checks pause to allow recovery
- **2 consecutive failures required**: Single transient failures don't trigger restarts

If you see `rate_limit_exceeded` in logs, there's likely a deeper issue:
1. Check camera/RTSP source
2. Check disk space (`df -h`)
3. Check memory (`free -m`)
4. Check system logs (`dmesg | tail -50`)

## Why Restart Works

When `hobot_codec` is killed:

1. The ROS2 launch system respawns it automatically
2. The new instance opens a fresh VPU session
3. It waits for the next IDR keyframe from `/spool_image_ch_0`
4. Once it receives an IDR, decoding resumes normally

The RTSP camera typically sends an IDR every 17-30 frames (0.5-1 second at 30fps), so recovery is usually fast.

## Troubleshooting

### Monitor Not Starting

```bash
# Check Python path
python3 -c "from src.ros2.codec_health_monitor import CodecHealthMonitor; print('OK')"

# Check ROS2 CLI available
ros2 topic list
```

### Restarts Not Helping

If restarts don't recover the system:

1. **Check the spool processor**: Is `/spool_image_ch_0` actually publishing?
   ```bash
   ros2 topic hz /spool_image_ch_0
   ```

2. **Check disk space**: Spool needs disk space
   ```bash
   df -h /tmp/spool
   ```

3. **Check RTSP source**: Is the camera accessible?
   ```bash
   ros2 topic hz /rtsp_image_ch_0
   ```

4. **Full system restart**: If all else fails
   ```bash
   sudo systemctl restart conveyor-counting.service
   ```

