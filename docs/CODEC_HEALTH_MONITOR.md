# Codec Health Monitor

## Overview

The Codec Health Monitor provides **multi-point pipeline health detection** and **staged recovery** for the ConveyorBreadBagCounterSystem media pipeline on RDK platforms. It watches three checkpoints along the data flow and escalates through increasingly aggressive recovery stages when failures persist.

## Architecture

### Data Flow

```
RTSP Camera
  │
  ▼
hobot_rtsp_client          ── /rtsp_image_ch_0      [HealthCheckpoint.RTSP_INGEST]
  │
  ▼
SpoolRecorderNode          ── writes H.264 to /tmp/spool/
  │
  ▼
SpoolProcessorNode         ── /spool_image_ch_0      [HealthCheckpoint.SPOOL_INPUT]
  │
  ▼
hobot_codec (VPU decoder)  ── /nv12_images           [HealthCheckpoint.CODEC_OUTPUT]
  │
  ▼
ConveyorCounterApp         ── detection & counting
```

### Supervisor Services

| Service | Process | Restartable Independently? |
|---|---|---|
| `breadcount-ros2` | hobot_rtsp_client + hobot_codec (via ROS2 launch) | Yes |
| `breadcount-spool-recorder` | SpoolRecorderNode | Yes (preserved in stages 1-2) |
| `breadcount-spool-processor` | SpoolProcessorNode | Yes |
| `breadcount-main` | ConveyorCounterApp | Yes |
| `breadcount-uvicorn` | FastAPI server | Never restarted by monitor |

## Failure Domains and Recovery Boundaries

| Domain | Symptoms | First Recovery Stage |
|---|---|---|
| **VPU stall** | `/nv12_images` dead, process alive | Stage 1 (CODEC_ONLY) |
| **Codec crash** | Process missing | Stage 1 (CODEC_ONLY) |
| **Spool pipeline** | `/spool_image_ch_0` dead | Stage 2 (MEDIA_CONSUMERS) |
| **RTSP ingest** | `/rtsp_image_ch_0` dead | Stage 3 (FULL_MEDIA_STACK) |
| **Broad corruption** | Multiple failures after stage 3 | Stage 4 (BROAD_SERVICES) |
| **Hardware / OS** | All stages exhausted | Stage 5 (REBOOT_RECOMMENDED) |

## Staged Recovery Strategy

### Stage 1: CODEC_ONLY

- **Action**: SIGTERM → wait 5 s → SIGKILL `hobot_codec`
- **Scope**: Only the VPU decoder process
- **spool_record**: Kept alive
- **When**: Default first response to `/nv12_images` stall

### Stage 2: MEDIA_CONSUMERS

- **Action**: `supervisorctl restart breadcount-ros2 breadcount-spool-processor`
- **Scope**: ROS2 launch (rtsp + codec) and spool processor
- **spool_record**: Kept alive (continues writing to disk)
- **When**: Stage 1 failed to restore health

### Stage 3: FULL_MEDIA_STACK

- **Action**: `supervisorctl restart breadcount-ros2 breadcount-spool-processor breadcount-spool-recorder`
- **Scope**: Entire media pipeline including spool recorder
- **spool_record**: Restarted
- **When**: Stage 2 failed — likely RTSP or spool corruption

### Stage 4: BROAD_SERVICES

- **Action**: `supervisorctl restart breadcount-ros2 breadcount-spool-processor breadcount-spool-recorder breadcount-main`
- **Scope**: All services except uvicorn (API stays up for diagnostics)
- **spool_record**: Restarted
- **When**: Stage 3 failed — broad pipeline corruption

### Stage 5: REBOOT_RECOMMENDED

- **Action**: Logs `CRITICAL` message recommending manual reboot
- **No automatic action** — requires human intervention
- **When**: All automated stages exhausted

### Escalation Flow

```
Health Check Failed (consecutive >= threshold)
  │
  ├─ Stage 1 ──→ success? ──→ reset to Stage 1, continue monitoring
  │                  │
  │                  └─ fail ──→ escalate to Stage 2
  │
  ├─ Stage 2 ──→ success? ──→ reset to Stage 1
  │                  │
  │                  └─ fail ──→ escalate to Stage 3
  │
  ├─ Stage 3 ──→ success? ──→ reset to Stage 1
  │                  │
  │                  └─ fail ──→ escalate to Stage 4
  │
  ├─ Stage 4 ──→ success? ──→ reset to Stage 1
  │                  │
  │                  └─ fail ──→ escalate to Stage 5
  │
  └─ Stage 5 ──→ LOG REBOOT RECOMMENDED (no auto-action)
```

When the pipeline returns to healthy, escalation resets to Stage 1.

## Health Checkpoints

| Checkpoint | Topic | What It Proves |
|---|---|---|
| `RTSP_INGEST` | `/rtsp_image_ch_0` | Camera feed arriving |
| `SPOOL_INPUT` | `/spool_image_ch_0` | Spool processor publishing decoded H.264 |
| `CODEC_OUTPUT` | `/nv12_images` | VPU decoder producing NV12 frames |

The monitor also detects **stalled processes** — where `hobot_codec` is running (PID exists) but producing no output.

## Clean Boot Handling

On startup the monitor (and `run_app.sh`) remove stale artifacts:

| Artifact | Location | Why |
|---|---|---|
| Status file | `/tmp/codec_health_status.json` | Stale state from previous run |
| Spool temps | `/tmp/spool/*.tmp` | Partially written spool segments |
| FastDDS SHM | `/dev/shm/fastrtps_*`, `/dev/shm/fast_datasharing_*` | Shared memory from crashed DDS |

Call `perform_startup_cleanup()` from Python or rely on the shell cleanup in `run_app.sh`.

## spool_record Preservation

| Stage | spool_record |
|---|---|
| 1 (CODEC_ONLY) | **Preserved** — recording continues |
| 2 (MEDIA_CONSUMERS) | **Preserved** — recording continues |
| 3 (FULL_MEDIA_STACK) | Restarted |
| 4 (BROAD_SERVICES) | Restarted |
| 5 (REBOOT_RECOMMENDED) | N/A |

Each recovery event logs whether `spool_record` was alive before and after the action.

## Observability

### Recovery Events

The monitor keeps the last 20 `RecoveryEvent` records with:

- `timestamp` / `time_iso`
- `stage` (1-5)
- `action` (stage name)
- `success` (bool)
- `spool_record_alive` (bool)
- `details`

### Health Status API

| Endpoint | Method | Description |
|---|---|---|
| `/api/health/codec` | GET | Full codec health + recovery_events + health_checkpoints |
| `/api/health/pipeline` | GET | Overall pipeline health (codec + spool + rtsp + recovery stage) |
| `/api/health/codec/restart` | POST | Manually kill hobot_codec |
| `/api/health/pipeline/recover` | POST | Trigger recovery at a specific stage (`{"stage": 1}` … `{"stage": 4}`) |

### Shared Status File

Written to `/tmp/codec_health_status.json` after every health check cycle. Contains all fields from `get_stats()` including `health_checkpoints`, `recovery_events`, `current_recovery_stage`, and `escalation_count`.

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `CODEC_MONITOR_TOPIC` | `/nv12_images` | Primary topic to monitor (codec output) |
| `CODEC_MONITOR_SPOOL_TOPIC` | `/spool_image_ch_0` | Spool input topic |
| `CODEC_MONITOR_RTSP_TOPIC` | `/rtsp_image_ch_0` | RTSP ingest topic |
| `CODEC_MONITOR_TIMEOUT` | `10` | Seconds to wait for a message |
| `CODEC_MONITOR_INTERVAL` | `15` | Seconds between health checks |
| `CODEC_MONITOR_THRESHOLD` | `2` | Consecutive failures before recovery |
| `CODEC_MONITOR_COOLDOWN` | `30` | Seconds to wait after recovery |
| `CODEC_MONITOR_MAX_RESTARTS` | `5` | Max restarts per hour (circuit breaker) |
| `CODEC_MONITOR_ENABLE_RESTART` | `true` | Enable automatic recovery |
| `CODEC_MONITOR_VERBOSE` | `false` | Enable verbose debug logging |
| `CODEC_RESTART_COMMAND` | *(empty)* | Command to run when process is absent |
| `CODEC_MONITOR_PROCESS_START_TIMEOUT` | `10` | Seconds to wait for process after restart |
| `CODEC_MONITOR_GRACEFUL_KILL_TIMEOUT` | `5` | Seconds between SIGTERM and SIGKILL |
| `CODEC_MONITOR_ENABLE_STARTUP_CLEANUP` | `true` | Run cleanup on startup |
| `CODEC_MONITOR_MAX_RECOVERY_EVENTS` | `20` | Max recovery events to keep in history |

### Integration (main.py)

```python
from src.ros2.codec_health_monitor import CodecHealthMonitor, MonitorConfig

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

### Standalone systemd Service

```bash
sudo cp systemd/codec-health-monitor.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now codec-health-monitor.service
```

## Troubleshooting

### Monitor Not Starting

```bash
python3 -c "from src.ros2.codec_health_monitor import CodecHealthMonitor; print('OK')"
ros2 topic list
```

### Rate Limit Exceeded

If you see `rate_limit_exceeded` in logs (>5 restarts/hour), investigate deeper issues:

1. Check camera/RTSP source
2. Check disk space: `df -h /tmp/spool`
3. Check memory: `free -m`
4. Check VPU: `dmesg | grep -i vpu | tail -10`
5. Check system logs: `dmesg | tail -50`

### Recovery Not Escalating

The monitor only escalates when a recovery action **succeeds** (command ran) but health is **still not restored** at the next check. If the command itself fails, escalation still happens.

### Restarts Not Helping

1. Check spool processor: `ros2 topic hz /spool_image_ch_0`
2. Check disk space: `df -h /tmp/spool`
3. Check RTSP source: `ros2 topic hz /rtsp_image_ch_0`
4. Trigger manual recovery: `curl -X POST http://localhost:8080/api/health/pipeline/recover -H 'Content-Type: application/json' -d '{"stage": 3}'`

### Check Topic Flow

```bash
# RTSP input (~17-18 Hz expected)
ros2 topic hz /rtsp_image_ch_0

# Spool output (~17-18 Hz expected)
ros2 topic hz /spool_image_ch_0

# Codec output (~17-18 Hz expected, 0 = stalled)
ros2 topic hz /nv12_images
```

### Check VPU Status

```bash
dmesg | grep -i vpu | tail -10
dmesg | grep -i "vpu_close_instance\|vpu_open_instance"
```

## Recovery Flow Diagram

```
                    ┌──────────────┐
                    │  System Boot │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ run_app.sh   │
                    │ cleanup stale│
                    │ artifacts    │
                    └──────┬───────┘
                           │
                    ┌──────▼───────────┐
                    │ perform_startup_ │
                    │ cleanup() in     │
                    │ main() entry     │
                    └──────┬───────────┘
                           │
                    ┌──────▼───────┐
                    │  30s startup │
                    │  delay       │
                    └──────┬───────┘
                           │
              ┌────────────▼────────────┐
              │   Health Check Loop     │◄────────────┐
              │  (every 15s)            │             │
              └────────────┬────────────┘             │
                           │                          │
                   ┌───────▼───────┐                  │
                   │  All checks   │                  │
                   │  pass?        │                  │
                   └───┬───────┬───┘                  │
                   yes │       │ no                    │
                       │       │                      │
              ┌────────▼──┐  ┌─▼──────────────┐       │
              │ HEALTHY   │  │ consecutive    │       │
              │ reset     │  │ failures >=    │       │
              │ escalation│  │ threshold?     │       │
              └───────────┘  └──┬──────────┬──┘       │
                            yes │          │ no       │
                                │          │          │
                       ┌────────▼──────┐   │          │
                       │ can_restart?  │   │          │
                       │ (rate limit)  │   │          │
                       └───┬────────┬──┘   │          │
                       yes │        │ no   │          │
                           │        │      │          │
                  ┌────────▼─────┐  │      │          │
                  │ execute      │  │      │          │
                  │ recovery at  │  │      │          │
                  │ current stage│  │      │          │
                  └───┬──────┬───┘  │      │          │
                  ok  │      │ fail │      │          │
                      │      │      │      │          │
             ┌────────▼──┐ ┌─▼──────▼──┐   │          │
             │ cooldown  │ │ escalate  │   │          │
             │ + reset   │ │ to next   │   │          │
             │ failures  │ │ stage     │   │          │
             └───────┬───┘ └───────┬───┘   │          │
                     │             │       │          │
                     └─────────────┴───────┴──────────┘
```

