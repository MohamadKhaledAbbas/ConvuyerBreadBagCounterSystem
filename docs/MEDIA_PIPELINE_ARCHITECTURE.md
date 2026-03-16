# Media Pipeline Architecture

This document describes the runtime architecture of the ConveyorBreadBagCounterSystem media pipeline, its services, data flow, failure domains, and recovery strategy.

## Service Map

All services are managed by **supervisord**:

| Supervisor Name | Process(es) | Role |
|---|---|---|
| `breadcount-ros2` | `hobot_rtsp_client` + `hobot_codec` (via ROS2 launch) | RTSP ingest & H.264 → NV12 decode |
| `breadcount-spool-recorder` | `SpoolRecorderNode` | Records raw RTSP H.264 frames to `/tmp/spool/` |
| `breadcount-spool-processor` | `SpoolProcessorNode` | Reads spool segments, publishes to `/spool_image_ch_0` |
| `breadcount-main` | `main.py` (ConveyorCounterApp) | Detection, tracking, counting |
| `breadcount-uvicorn` | FastAPI via Uvicorn | REST API for UI, health, analytics |

## Data Flow

```
┌─────────────────┐
│  RTSP Camera    │  (IP camera, H.264 stream)
└────────┬────────┘
         │  RTSP/TCP
         ▼
┌─────────────────┐
│ hobot_rtsp_     │  breadcount-ros2
│ client          │  Publishes: /rtsp_image_ch_0
└────────┬────────┘
         │
    ┌────▼────┐
    │  Fork   │  (ROS2 topic pub/sub)
    ├─────────┤
    │         │
    ▼         ▼
┌────────┐  ┌───────────────┐
│ hobot_ │  │ SpoolRecorder │  breadcount-spool-recorder
│ codec  │  │ Node          │
│ (VPU)  │  │ Writes .h264  │
│        │  │ to /tmp/spool │
│        │  └───────┬───────┘
│        │          │  (disk)
│        │          ▼
│        │  ┌───────────────┐
│        │  │ SpoolProcessor│  breadcount-spool-processor
│        │  │ Node          │
│        │  │ Publishes:    │
│        │  │ /spool_image_ │
│        │  │ ch_0          │
│        │  └───────┬───────┘
│        │          │  (ROS2 topic)
│        │◄─────────┘
│        │  (hobot_codec subscribes to /spool_image_ch_0)
│        │
│ Publish│
│ /nv12_ │
│ images │
└────┬───┘
     │
     ▼
┌────────────────┐
│ ConveyorCounter│  breadcount-main
│ App            │
│ (detection,    │
│  tracking,     │
│  counting)     │
└────────┬───────┘
         │
         ▼
┌────────────────┐
│ FastAPI /      │  breadcount-uvicorn
│ Uvicorn        │
│ (REST API,     │
│  health, UI)   │
└────────────────┘
```

### Topic Summary

| Topic | Publisher | Subscriber(s) | Format |
|---|---|---|---|
| `/rtsp_image_ch_0` | `hobot_rtsp_client` | `SpoolRecorderNode` | Raw H.264 NAL units |
| `/spool_image_ch_0` | `SpoolProcessorNode` | `hobot_codec` | H.264 NAL units (from spool) |
| `/nv12_images` | `hobot_codec` | `ConveyorCounterApp` | NV12 decoded frames |

## Failure Domains

The pipeline can be divided into four failure domains. Each domain has a different blast radius and recovery cost.

### Domain 1: VPU / Codec

**Components**: `hobot_codec` process only

**Failure Modes**:
- VPU stall (process alive but no output)
- Process crash (PID disappears)
- Memory pressure causing VPU close/reopen

**Recovery**: Kill `hobot_codec`; ROS2 launch respawns it. Spool continues recording.

**Impact**: Brief frame gap (~1-2 s) while codec reinitialises and waits for IDR keyframe.

### Domain 2: Media Consumers

**Components**: `breadcount-ros2` + `breadcount-spool-processor`

**Failure Modes**:
- `SpoolProcessorNode` crash or stall
- ROS2 DDS communication failure
- Stale FastDDS shared memory after power loss

**Recovery**: `supervisorctl restart breadcount-ros2 breadcount-spool-processor`. Spool recorder keeps writing to disk so no frames are lost permanently.

**Impact**: ~5-10 s pipeline interruption.

### Domain 3: Full Media Stack

**Components**: `breadcount-ros2` + `breadcount-spool-processor` + `breadcount-spool-recorder`

**Failure Modes**:
- RTSP source disconnection
- Spool recorder corruption
- Disk full preventing spool writes
- Cascading DDS failures

**Recovery**: Restart all three media services. This is the first stage that restarts spool_recorder.

**Impact**: ~10-15 s pipeline interruption; spool recording gap during restart.

### Domain 4: Application-Wide

**Components**: All except `breadcount-uvicorn`

**Failure Modes**:
- Cascading failure across detection + media
- Corrupted application state
- Shared memory leaks

**Recovery**: Restart everything except the API server (so diagnostics remain accessible).

**Impact**: Full pipeline restart; ~15-30 s downtime.

## Service Dependency Graph

```
breadcount-uvicorn  (independent – never auto-restarted)
     │
     │  reads /tmp/codec_health_status.json
     ▼
breadcount-main
     │
     │  subscribes to /nv12_images
     ▼
breadcount-ros2  (hobot_rtsp_client + hobot_codec)
     │
     │  hobot_codec subscribes to /spool_image_ch_0
     ▼
breadcount-spool-processor
     │
     │  reads from /tmp/spool/
     ▼
breadcount-spool-recorder
     │
     │  subscribes to /rtsp_image_ch_0 (from breadcount-ros2)
     ▼
(RTSP Camera – external)
```

## Recovery Stage Explanations

| Stage | Name | Services Restarted | Tradeoff |
|---|---|---|---|
| 1 | CODEC_ONLY | hobot_codec process | Minimal impact; spool continues |
| 2 | MEDIA_CONSUMERS | breadcount-ros2, breadcount-spool-processor | Slightly wider; spool recorder still writes |
| 3 | FULL_MEDIA_STACK | breadcount-ros2, breadcount-spool-processor, breadcount-spool-recorder | Spool recording interrupted |
| 4 | BROAD_SERVICES | All except breadcount-uvicorn | Full restart; API stays for diagnostics |
| 5 | REBOOT_RECOMMENDED | None (logged only) | Requires human intervention |

### When spool_record Can Remain Alive

**Stages 1 and 2**: The spool recorder's only dependency is `/rtsp_image_ch_0`. If `hobot_rtsp_client` is restarted (as part of `breadcount-ros2`), the recorder reconnects via DDS. This means recording continues during codec-only and media-consumer restarts.

### When spool_record Must Restart

**Stage 3+**: If the spool recorder itself is corrupted (e.g., stuck writing a partial segment, disk errors) or if RTSP ingest is completely dead, the recorder must be restarted to clear its state.

### When Full Reboot Is Needed

A reboot is recommended (Stage 5) when:

- All four automated recovery stages have been attempted and failed
- VPU hardware is in an unrecoverable state (visible in `dmesg`)
- Kernel-level resource exhaustion (file descriptors, shared memory segments)
- Power loss left the system in an inconsistent state that cleanup cannot fix

## Cross-Process Communication

The codec health monitor (running inside `breadcount-main`) writes its status to `/tmp/codec_health_status.json`. The FastAPI server (`breadcount-uvicorn`) reads this file to serve health endpoints. This avoids needing shared memory between the two processes.

The status file includes:
- Current `HealthState`
- `health_checkpoints` for all three topics
- `recovery_events` (last 20)
- `current_recovery_stage` and `escalation_count`
- Standard counters (checks, failures, restarts)

## Clean Boot Sequence

1. `run_app.sh` removes stale artifacts (`/tmp/codec_health_status.json`, `/tmp/spool/*.tmp`, `/dev/shm/fastrtps_*`, `/dev/shm/fast_datasharing_*`)
2. All services are stopped for a clean start
3. Services are started based on development/production mode
4. `main()` in `codec_health_monitor.py` calls `perform_startup_cleanup()` as a safety net
5. Monitor waits 30 s for the pipeline to stabilise before first health check
