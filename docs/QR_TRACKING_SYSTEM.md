# QR Tracking System — صالة (Sale Point)

> Horizontal container-counting pipeline using an overhead QR camera and,
> optionally, a second 3D-angle content camera that records what was
> inside each container.

---

## 1. Overview

The sale point tracks five physical containers, each printed with a
unique QR code (values **1–5**). The overhead camera at
`192.168.2.118` watches them pass through the sale area. Every time a
container crosses the frame and exits, the app:

1. Reads the QR, tracks the container horizontally across the frame.
2. Decides **direction** (filled-out or empty-return).
3. Stores a lifetime snapshot of that container's track (JPEG frames +
   metadata).
4. Optionally triggers a short MP4 recording from a **second** camera
   (`192.168.2.128`) that shows the container's contents from a 3D
   angle — pre-roll of ~3 s and post-roll of ~2 s, so operators can
   audit what each container actually held.
5. Writes the event to SQLite and publishes live state for the
   dashboard.

```
   ┌────────────────────┐
   │ 192.168.2.118 (QR) │  ceiling, top-down
   └─────────┬──────────┘
             │ frames (ROS2 or OpenCV)
             ▼
   ┌────────────────────┐          ┌────────────────────────┐
   │ QRCodeDetector     │          │ 192.168.2.128          │
   │ (WeChatQRCode)     │          │ (Content camera, 3D)   │
   └─────────┬──────────┘          └──────────┬─────────────┘
             ▼                                 │ rolling 5s buffer
   ┌────────────────────┐                     │
   │ ContainerTracker   │                     ▼
   │ (X-axis, 5 tracks) │          ┌────────────────────────┐
   └─────────┬──────────┘          │ ContentCameraRecorder  │
             │ event                │ (pre+post → .mp4)      │
             ▼                     └──────────┬─────────────┘
   ┌────────────────────┐                     │
   │ _handle_event      │◄────────────────────┘
   │  - lifetime snap   │ trigger_recording(event_id)
   │  - DB insert       │
   │  - state publish   │
   └────────────────────┘
```

---

## 2. Tracking axis & direction

The overhead camera is mounted on the ceiling, so the containers travel
**horizontally** across the frame:

| Direction of motion | Label      | Meaning                                 |
| ------------------- | ---------- | --------------------------------------- |
| right → left        | `positive` | **Filled** container leaving the point  |
| left → right        | `negative` | **Empty** container returning           |
| never reached edge  | `unknown`  | Lost track — QR lost before exit zone   |

> **Note on the DB schema:** the `container_events` table still has
> columns named `entry_y` / `exit_y` from the original vertical layout.
> They now store **X** coordinates. Renaming would touch migrations,
> repositories, and templates — deferred as a cosmetic cleanup.
> See `FIX_T13_T14_MERGE_ISSUE.md` style docs if renamed later.

---

## 3. QR detection

* **Engine:** `cv2.wechat_qrcode.WeChatQRCode` (OpenCV-contrib).
* **Fallback:** `cv2.QRCodeDetector` — less robust on tilted / blurry
  codes.
* Model files live under `data/model/wechat_qrcode/`.
* Detection runs every **`container_detect_interval`** frames
  (default **3**) — intermediate frames use a lightweight linear
  predictor so the tracker sees positions every frame without burning
  CPU on QR decoding.

### `_LinearPredictor`

A per-QR EMA-smoothed X-velocity predictor:

* On each real detection, it updates `(cx, vx)` for that QR.
* Between detections, `advance()` emits
  `cx_next = cx + vx * elapsed_frames` with `cy` held constant (camera
  view is 2D top-down).
* Predicted detections are tagged with `confidence = 0.0` so downstream
  logic can distinguish them.
* A prediction is dropped after `MAX_PRED_FRAMES` without a real
  detection, preventing "ghost" tracks.

---

## 4. Container tracking

Module: `src/container/tracking/ContainerTracker.py`.

Each of the 5 QR values maps to at most **one active track**. The
tracker tracks `(cx, cy, bw, bh)` per track and checks:

| State       | Trigger                                               |
| ----------- | ----------------------------------------------------- |
| **Active**  | QR seen within `lost_timeout` seconds                 |
| **Exited**  | `cx ≤ left_exit_x` or `cx ≥ right_exit_x`             |
| **Lost**    | No QR for `lost_timeout` seconds AND not exited       |

Exit boundaries are derived from
`exit_zone_ratio` × `frame_width` (default **15 %** on each side).

Minimum displacement to count: `min_displacement_ratio` × `frame_width`
(default **30 %**). Prevents spurious events from jitter near the
entry edge.

---

## 5. Snapshot pipeline

On every exit/lost event `ContainerCounterApp._save_track_snapshot()`
writes a directory:

```
data/container_snapshots/
    qr{N}_{direction}_{YYYYMMDD_HHMMSS_ffffff}/
        frames/
            frame_0000.jpg
            frame_0001.jpg
            …
        metadata.json
```

* **Frames** are half-resolution BGR captures, one per detect cycle.
* **JPEG encoding** happens entirely in a `ThreadPoolExecutor`
  (`_snapshot_io`) — the main loop is never blocked on `cv2.imencode`
  or disk writes.
* `metadata.json` stores `{event_id, qr_value, direction, trigger_time,
  frame_count, fps, duration_seconds, track_id, entry_x, exit_x}`.

---

## 6. Content camera (second camera, 3D angle)

Optional. Enabled via the config key
`content_recording_enabled = '1'`.

### 6.1 What it does

`ContentCameraRecorder`
(`src/container/content/ContentCameraRecorder.py`):

* Reader thread pulls frames from the content RTSP stream at
  `content_video_fps` (default **15 fps**) and pushes them into a
  thread-safe `deque(maxlen = buffer_seconds × fps)` — a rolling
  **~5 s** ring buffer is always hot in memory.
* When `trigger_recording(event_id)` is called, it snapshots the ring
  buffer (the last `content_pre_event_seconds`, default **3 s**) and
  schedules the reader to keep appending frames for
  `content_post_event_seconds` (default **2 s**) more.
* A dedicated writer thread drains completed recordings to
  `data/container_content_videos/{event_id}.mp4` using
  `cv2.VideoWriter` with the `mp4v` fourcc. Atomic rename
  (`.mp4.part` → `.mp4`) prevents partial files.
* **Auto-reconnect:** if the RTSP read fails, the reader waits
  `reconnect_delay` seconds and retries. The ring buffer does not grow
  during the disconnect but old frames remain available for triggers.

### 6.2 Configuration keys

| Key                          | Default        | Meaning                                   |
| ---------------------------- | -------------- | ----------------------------------------- |
| `content_recording_enabled`  | `0`            | `1` to enable the second-camera recorder  |
| `content_rtsp_host`          | `192.168.2.128`| Camera IP                                 |
| `content_rtsp_port`          | `554`          | RTSP port                                 |
| `content_rtsp_username`      | `""`           | Credentials                               |
| `content_rtsp_password`      | `""`           | Credentials                               |
| `content_buffer_seconds`     | `5.0`          | Ring-buffer capacity in seconds           |
| `content_pre_event_seconds`  | `3.0`          | Pre-roll included in each recording       |
| `content_post_event_seconds` | `2.0`          | Post-event capture duration               |
| `content_video_fps`          | `15`           | Reader sampling rate & output video FPS   |

### 6.3 RTSP URL template

```
rtsp://{user}:{pwd}@{host}:{port}/cam/realmonitor?channel=1&subtype=0
```

Matches the pattern already used by the overhead camera and the bread
camera.

### 6.4 Endpoints

| Path                                   | Purpose                                   |
| -------------------------------------- | ----------------------------------------- |
| `GET /container/content`               | HTML page — video wall with filters       |
| `GET /container/api/content/list`      | JSON: list of recordings (newest first)   |
| `GET /container/content/video/{id}`    | Streaming MP4 (Range-request capable)     |

Event IDs are validated against `^[A-Za-z0-9_\-]+$` and paths are
`realpath`-checked against `CONTAINER_CONTENT_VIDEOS_DIR` to block
directory traversal.

---

## 7. Database

Table: `container_events` (in `data/db/bag_events.db`).

```
id, timestamp, qr_code_value, direction, track_id,
entry_y, exit_y,                -- legacy names, store X values
duration_seconds, snapshot_path, is_lost, metadata
```

The `metadata` JSON column carries the content-video filename:

```json
{
  "position_count": 87,
  "entry_x": 120,
  "exit_x": 1180,
  "content_video": "qr3_positive_20251105_143001_123456.mp4"
}
```

---

## 8. Config keys (tracking side)

| Key                            | Default | Meaning                                         |
| ------------------------------ | ------- | ----------------------------------------------- |
| `container_exit_zone_ratio`    | `0.15`  | Exit-band width as fraction of frame width      |
| `container_lost_timeout`       | `2.0`   | Seconds without QR before track is lost         |
| `container_pre_event_seconds`  | `5.0`   | Pre-event snapshot window (main camera)         |
| `container_post_event_seconds` | `5.0`   | Post-event snapshot window (main camera)        |
| `container_detect_interval`    | `3`     | Run real QR detection every N-th frame          |
| `container_enable_display`     | `0`     | `1` to open a debug OpenCV window               |

---

## 9. Operational notes & known limitations

* **Schema cosmetic debt:** `entry_y` / `exit_y` columns store X values.
  Not wrong, just poorly named. Rename only with a proper migration.
* **Lost-track direction ambiguity:** if a container is lost *before*
  entering the exit zone, its direction is `unknown`. This is
  deliberate — we refuse to guess.
* **Content camera is optional and best-effort.** If the second camera
  is offline at startup, the recorder reconnects in the background.
  Events still record successfully without a content video; the
  `metadata.content_video` field is simply `null`.
* **Codec:** the default `mp4v` works everywhere but produces larger
  files than `avc1`. Switch via `ContentRecorderConfig.codec` if your
  OpenCV build has h264 available.
* **Storage growth:** each content MP4 is ~0.5–2 MB at 15 fps, 5 s.
  Add a cron retention policy if you expect hundreds of events per day.

---

## 10. Quick troubleshooting

| Symptom                                          | Check                                                               |
| ------------------------------------------------ | ------------------------------------------------------------------- |
| No content videos written                        | `content_recording_enabled = 1`? Camera reachable? Logs contain `[ContentRecorder] Connected`? |
| All events show `content_video: null`            | Recorder failed to start. Check logs around `_init_content_recorder`. |
| Content page empty but files exist on disk       | Filenames must match `^[A-Za-z0-9_\-]+\.mp4$`.                      |
| Main loop FPS drops                              | JPEG encoding & video writing both run off-thread — check for disk I/O saturation. |
| Wrong direction labels                           | Verify camera orientation — right→left must be `positive`.          |
| Tracker ghost predictions                        | Lower `_LinearPredictor.MAX_PRED_FRAMES` or increase detection rate. |
