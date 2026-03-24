# Adaptive Frame Throttle — Idle Power-Saving Mode

## Overview

The conveyor pipeline processes camera frames at ~17 FPS. When the production
line is idle (no bread bags on the conveyor), all that processing is wasted
CPU/power. The **Adaptive Frame Throttle** detects idle periods and
automatically degrades to processing every 5th frame (~3.4 FPS), reducing CPU
load by ~80%.

The moment a bag is detected on the conveyor, the system **immediately** wakes
back to full 17 FPS processing — zero bags are missed.

## How It Works

```
┌─────────────────────────────────────────────────────────┐
│                    State Machine                        │
│                                                         │
│   FULL ──── (30 min no detections) ────→ DEGRADED       │
│    ↑                                        │           │
│    └──── (any detection found) ─────────────┘           │
│           (stays FULL for 60s hysteresis)                │
└─────────────────────────────────────────────────────────┘
```

| Mode       | Frames Processed | Effective FPS | CPU Load |
|------------|-----------------|---------------|----------|
| **FULL**   | Every frame     | ~17 FPS       | 100%     |
| **DEGRADED** | Every 5th frame | ~3.4 FPS    | ~20%     |

### Key Design Decisions

1. **Frame skipping at the processing level, NOT the source level** — the
   camera feed continues to be read and drained at full rate. Only the
   expensive detection/tracking/classification is skipped. This prevents
   buffer overflows in the frame source queue.

2. **Immediate wake-up** — `report_activity()` is called the instant
   detections are found. The throttle switches back to FULL mode within the
   same loop iteration, so the very next frame is fully processed.

3. **Hysteresis window** — after waking from DEGRADED, the throttle stays in
   FULL mode for at least 60 seconds. This prevents rapid oscillation from
   single spurious detections (e.g., a hand reaching into the frame).

4. **Aligned with smoother timeout** — the 30-minute idle timeout matches the
   existing smoother inactivity flush timeout
   (`BIDIRECTIONAL_INACTIVITY_TIMEOUT_MS`), so both systems agree on what
   "idle" means.

## Configuration

All settings are configurable via environment variables:

| Environment Variable             | Default  | Description                                      |
|----------------------------------|----------|--------------------------------------------------|
| `FRAME_THROTTLE_ENABLED`         | `true`   | Master switch (set `false` to disable)           |
| `FRAME_THROTTLE_IDLE_TIMEOUT_S`  | `1800`   | Seconds of no detections before degrading (30m)  |
| `FRAME_THROTTLE_SKIP_N`          | `5`      | In degraded mode, process every Nth frame        |
| `FRAME_THROTTLE_HYSTERESIS_S`    | `60`     | Min seconds to stay FULL after waking            |

### Examples

```bash
# Disable throttle entirely (always process all frames)
FRAME_THROTTLE_ENABLED=false

# Degrade after 15 minutes instead of 30
FRAME_THROTTLE_IDLE_TIMEOUT_S=900

# Process every 3rd frame in idle (less aggressive savings)
FRAME_THROTTLE_SKIP_N=3

# No hysteresis (immediate re-degradation allowed)
FRAME_THROTTLE_HYSTERESIS_S=0
```

## Observability

### Pipeline State (JSON)

The throttle state is exposed in the pipeline state file
(`data/pipeline_state.json`) under the `frame_throttle` key:

```json
{
  "frame_throttle": {
    "enabled": true,
    "mode": "degraded",
    "idle_seconds": 2145.3,
    "idle_timeout_s": 1800.0,
    "skip_n": 5,
    "hysteresis_s": 60.0,
    "frames_skipped": 34521,
    "total_frames_seen": 43152,
    "degraded_transitions": 2,
    "wake_transitions": 1
  }
}
```

### Log Messages

```
[FrameThrottle] DEGRADE → processing every 5th frame | No detections for 30.0 min (threshold=30 min). CPU load reduced ~80%
[FrameThrottle] Still DEGRADED | idle for 45 min, skipped 15300 frames total
[FrameThrottle] WAKE → FULL | Detection found in degraded mode, resuming full-rate processing (skipped 15300 frames during idle period)
```

### Final Shutdown Stats

```
Frame throttle: mode=full, skipped=34521/43152 frames, degraded_transitions=2, wake_transitions=1
```

## Files Modified

| File | Change |
|------|--------|
| `src/app/adaptive_frame_throttle.py` | **NEW** — `AdaptiveFrameThrottle` class |
| `src/config/tracking_config.py` | Added 4 config fields |
| `src/app/ConveyorCounterApp.py` | Integrated throttle into main loop |
| `test_adaptive_frame_throttle.py` | **NEW** — 23 tests (incl. thread-safety) |

## Testing

```bash
python -m pytest test_adaptive_frame_throttle.py -v
```

23 tests covering:
- Basic mode behavior (FULL / DEGRADED)
- Idle timeout and degradation
- Frame skip patterns
- Wake-up on detection
- Hysteresis window
- State/observability
- Thread safety (concurrent access)
- Edge cases (clamping, lifecycle)

