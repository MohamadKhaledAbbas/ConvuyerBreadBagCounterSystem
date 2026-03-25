# Pipeline-Wide Power Save (Sentinel Mode)

Extends the adaptive frame throttle from an app-only skip mechanism into a
**pipeline-wide** power-saving strategy that idles the SpoolProcessor, VPU
decoder (hobot_codec), and detection pipeline during conveyor inactivity.

## Problem

The original `AdaptiveFrameThrottle` only throttled frame processing at the
ConveyorCounterApp level.  In DEGRADED mode:

| Component             | Old Behaviour            | Power during DEGRADED |
|-----------------------|--------------------------|-----------------------|
| hobot_rtsp_client     | Always running           | ~2 %                  |
| SpoolRecorderNode     | Always recording         | ~2 %                  |
| SpoolProcessorNode    | **Processing ALL frames**| ~15 %                 |
| hobot_codec (VPU)     | **Decoding ALL frames**  | ~25 %                 |
| ConveyorCounterApp    | Skip 4/5 frames          | ~20 %                 |
| **Total**             |                          | **~64 %**             |

Only ~36 % of hardware was actually resting.

## Solution: Sentinel Mode

```
┌──────────────┐   always on     ┌──────────────┐   always on
│ RTSP Camera  │ ──────────────▶ │ SpoolRecorder │ ──▶ disk
└──────────────┘                 └──────────────┘
                                        │
                              ┌─────────▼──────────┐
                              │  SpoolProcessor     │
                              │  ┌────────────────┐ │
                              │  │ FULL: all segs  │ │ ◀── reads /tmp/pipeline_throttle.json
                              │  │ SENTINEL: 1 f/s │ │
                              │  └────────────────┘ │
                              └─────────┬──────────┘
                                        │
                              ┌─────────▼──────────┐
                              │  hobot_codec (VPU)  │  naturally idles when
                              │  Decodes published  │  processor stops feeding
                              │  frames only        │
                              └─────────┬──────────┘
                                        │
                              ┌─────────▼──────────┐
                              │  ConveyorCounterApp │  processes every sentinel
                              │  Detection/Tracking │  frame for wake detection
                              └─────────────────────┘
```

### Power savings with sentinel mode

| Component             | New Behaviour                | Power during DEGRADED |
|-----------------------|------------------------------|-----------------------|
| hobot_rtsp_client     | Always running               | ~2 %                  |
| SpoolRecorderNode     | Always recording             | ~2 %                  |
| SpoolProcessorNode    | **1 frame/s (sentinel)**     | ~1 %                  |
| hobot_codec (VPU)     | **1 decode/s**               | ~2 %                  |
| ConveyorCounterApp    | **Processes all sentinels**  | ~6 %                  |
| **Total**             |                              | **~13 %**             |

**~87 % of hardware is resting** (up from ~36 %).

## Architecture

### Cross-Process Coordination

The throttle decision lives in ConveyorCounterApp (the main process).  The
SpoolProcessorNode runs in a **separate supervisord process**.  Coordination
uses a shared JSON state file:

```
/tmp/pipeline_throttle.json
{
    "mode": "full" | "degraded",
    "updated_at": 1711396800.0,
    "sentinel_interval_s": 1.0,
    "skip_n": 5
}
```

- **Writer**: ConveyorCounterApp — on mode transitions + periodic heartbeat (~25 s)
- **Reader**: SpoolProcessorNode — polls every ~2 s

### Sentinel Frame Selection

In DEGRADED mode, the SpoolProcessor:

1. Lists all complete segments on disk (`seg_NNNNNN.bin`)
2. Picks the **second-to-last** segment (the latest *complete* one; the very
   last may still be actively written by the recorder)
3. Reads only the **first frame** (IDR keyframe — segments are IDR-aligned)
4. Publishes it to `/spool_image_ch_0`
5. Sleeps for `sentinel_interval_s` (default 1.0 s)
6. Repeats

This gives the app a near-real-time probe frame to run detection on.

### Wake Sequence

```
1. Sentinel frame arrives at app
2. Detection model finds a bag → Signal A (report_detection)
3. AdaptiveFrameThrottle: DEGRADED → FULL
4. on_mode_change callback writes {"mode": "full"} to state file
5. SpoolProcessor reads "full" within ~2 s
6. SpoolProcessor exits sentinel mode:
   a. Advances cursor to (latest_segment - 3) to skip idle footage
   b. Deletes skipped idle segments via retention
   c. Resumes FAST sequential processing for catch-up
7. Pipeline is at full speed within ~3 s
```

### Idle Segment Skip on Wake

When transitioning from sentinel → full mode, the SpoolProcessor **skips**
accumulated idle segments to avoid reprocessing hours of empty belt footage:

- Keeps the last `sentinel_wake_buffer_segments` (default 3) segments before
  the latest as a safety buffer (~15 s of video)
- Deletes everything between the saved cursor and the resume point
- This means wake-to-real-time latency is only ~15 s of catch-up, regardless
  of how long the belt was idle
- Processing uses **FAST mode** (~30 fps, 33 ms min interval) which is faster
  than the recording rate (~17 fps), so the catch-up completes in ~8.5 seconds

## Retention Policy

Segments are cleaned up aggressively to prevent filling the eMMC:

| Parameter           | Default   | Description |
|---------------------|-----------|-------------|
| `max_age_hours`     | 0.083 (5 min) | Delete segments older than 5 minutes |
| `max_storage_bytes` | 1 GB      | Hard cap on total spool storage |
| `min_segments_keep` | 5         | Always keep at least 5 most recent segments |
| `check_interval_seconds` | 30 s | Background cleanup runs every 30 s |
| `only_delete_processed`  | `True` | Only delete segments that have been processed |

On wake from sentinel mode, skipped idle segments are deleted **immediately**
(not waiting for the background cleanup cycle), so disk space is reclaimed
within milliseconds of the transition.

## Health Endpoint: Comprehensive System Status

Every pipeline component writes a cross-process status file to `/tmp` (RAM-
backed tmpfs on RDK, no eMMC wear).  The FastAPI health endpoints read these
files to provide a complete picture of the system.

### Cross-Process Status Files

| File | Writer | Interval | Key metrics |
|------|--------|----------|-------------|
| `/tmp/spool_recorder_status.json` | SpoolRecorderNode | ~5 s | `avg_fps`, `frames_received`, `segments_completed`, `write_queue_size`, `write_queue_hwm` |
| `/tmp/spool_processor_status.json` | SpoolProcessorNode | ~5 s | `current_fps`, `time_behind_recorder_s`, `segments_behind`, `sentinel_active`, `sentinel_frames_sent` |
| `/tmp/codec_health_status.json` | Codec health monitor | ~10 s | `state`, `restarts_total`, `health_checkpoints` |
| `/tmp/pipeline_throttle.json` | ConveyorCounterApp | on transition + ~15 s heartbeat | `mode`, `sentinel_interval_s` |
| `data/pipeline_state.json` | ConveyorCounterApp | ~15 s | Counts, `frame_throttle`, `app_metrics` (FPS, active_tracks, etc.) |

### API Endpoints

| Endpoint | Description |
|----------|-------------|
| **`GET /health`** | Main health page endpoint — version, uptime, DB, pipeline counts, **throughput** (FPS at each stage), **spool_recorder**, **spool_processor**, **app_metrics**, components, system_info |
| **`GET /api/health/pipeline`** | Comprehensive pipeline health — all 5 components, throughput summary, power_save status, recovery stage |
| **`GET /api/health/spool`** | Dedicated spool processor stats |
| **`GET /api/health/recorder`** | Dedicated spool recorder (RTSP ingestion) stats |
| **`GET /api/health/codec`** | Codec VPU decoder health |
| **`GET /api/health/logs`** | Monitoring log entries from database |

### Throughput Summary (in `/health` and `/api/health/pipeline`)

```json
"throughput": {
    "recorder_fps": 17.1,
    "processor_fps": 17.0,
    "app_fps": 16.8,
    "time_behind_recorder_s": 5.0,
    "segments_behind": 1
}
```

Shows end-to-end FPS at each pipeline stage:
- **recorder_fps** — RTSP frames received from camera per second
- **processor_fps** — Frames published to codec per second (rolling 10 s window)
- **app_fps** — Frames processed by detection/tracking per second
- **time_behind_recorder_s** — How far behind real-time the processor is
- **segments_behind** — Unprocessed segments on disk

### App Metrics (in `/health` and `/api/health/pipeline`)

```json
"app_metrics": {
    "fps": 16.8,
    "frame_count": 51234,
    "processing_time_ms": 42.3,
    "active_tracks": 2,
    "pending_classifications": 0,
    "total_counted": 847,
    "lost_track_count": 12,
    "rejected_count": 3
}
```

### Key Health Metrics

| Metric                  | Description |
|-------------------------|-------------|
| `current_fps`           | Rolling 10-second FPS (real throughput, not lifetime avg) |
| `time_behind_recorder_s`| Estimated seconds the processor is behind the live RTSP feed |
| `segments_behind`       | Number of unprocessed segments on disk |
| `sentinel_active`       | Whether the processor is in sentinel power-save mode |
| `healthy`               | `true` if not stale AND (sentinel OR time_behind < 30 s) |

## Safety Mechanisms

### Staleness Timeout

If ConveyorCounterApp crashes while the state file says `"degraded"`, the
SpoolProcessor would be stuck in sentinel mode indefinitely.  To prevent this:

- `read_throttle_state()` checks the `updated_at` timestamp
- If the file hasn't been refreshed within **120 seconds**, mode is treated
  as `"full"` regardless of what the file says
- The main app refreshes the file every ~25 s (via `_maybe_publish_state_periodic`)

### Shutdown Cleanup

On graceful shutdown, ConveyorCounterApp writes `"full"` to the state file
via `cleanup_throttle_state()`, so the SpoolProcessor reverts immediately.

### Missing State File

If the state file doesn't exist (first boot, manual deletion),
`read_throttle_state()` returns `"full"`.  The SpoolProcessor runs at
full speed — the safe default.

### Codec Health Monitor Compatibility

During sentinel mode, the codec receives ~1 frame/s.  The codec health
monitor's `message_timeout_sec=10.0` is well above the 1 s sentinel
interval, so no false VPU-stall restarts are triggered.

## Configuration

| Environment Variable                      | Default | Description |
|-------------------------------------------|---------|-------------|
| `FRAME_THROTTLE_ENABLED`                  | `True`  | Master switch for the entire throttle system |
| `FRAME_THROTTLE_IDLE_TIMEOUT_S`           | `900`   | Seconds of no activity before DEGRADED |
| `FRAME_THROTTLE_SKIP_N`                   | `5`     | App-level frame skip factor (belt-and-suspenders) |
| `FRAME_THROTTLE_HYSTERESIS_S`             | `60`    | Stay in FULL for at least this many seconds after wake |
| `FRAME_THROTTLE_SENTINEL_INTERVAL_S`      | `1.0`   | Seconds between sentinel probe frames |
| `FRAME_THROTTLE_WAKE_BUFFER_SEGMENTS`     | `3`     | Segments to keep for catch-up on wake |

## Files Modified

| File | Change |
|------|--------|
| `src/app/pipeline_throttle_state.py` | **New** — shared state file I/O |
| `src/app/adaptive_frame_throttle.py` | Added `on_mode_change` callback |
| `src/app/ConveyorCounterApp.py` | Writes state file on transitions + heartbeat + cleanup |
| `src/spool/spool_processor_node.py` | Sentinel mode in `_process_loop` + status file writer + `RollingFPSCounter` |
| `src/spool/segment_io.py` | Added `read_first_record()` helper |
| `src/spool/retention.py` | Tightened defaults: 5 min max age, 1 GB max storage, 30 s check interval |
| `src/config/tracking_config.py` | New config: `sentinel_interval_s`, `wake_buffer_segments` |
| `src/endpoint/routes/health.py` | Added `/api/health/spool` endpoint + spool_processor in `/api/health/pipeline` |

## Development Mode

On non-RDK platforms (Windows/Linux), there is no spool pipeline.  The
pipeline-wide throttle has no effect — the app-level throttle (`should_process`)
continues to work as before.  The state file is still written but there is no
SpoolProcessor to read it.



