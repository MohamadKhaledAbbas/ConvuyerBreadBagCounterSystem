# Conveyor Bread Bag Counter System (v2) — Full Analysis Report

Created: 2026-02-03

This document is a **deep, end-to-end analysis** of the Conveyor Bread Bag Counter System v2. It explains how the system is organized, how data flows through it at runtime, and how each small component works—including configuration knobs, edge cases, and the purpose of each module.

> Audience: engineers who need to understand, maintain, deploy, or extend the system.

---

## Table of contents

1. [What the system does (executive summary)](#what-the-system-does-executive-summary)
2. [Core idea: detect → track → classify (after exit) → smooth → count](#core-idea-detect--track--classify-after-exit--smooth--count)
3. [Repository map (where everything lives)](#repository-map-where-everything-lives)
4. [Runtime entrypoints and modes](#runtime-entrypoints-and-modes)
5. [End-to-end runtime workflow (step-by-step)](#end-to-end-runtime-workflow-step-by-step)
6. [Data model / key entities (what’s passed between components)](#data-model--key-entities-whats-passed-between-components)
7. [Frame acquisition subsystem (`src/frame_source/`)](#frame-acquisition-subsystem-srcframe_source)
8. [Detection subsystem (`src/detection/`)](#detection-subsystem-srcdetection)
9. [Tracking subsystem (`src/tracking/`)](#tracking-subsystem-srctracking)
10. [ROI collection subsystem (`src/classifier/ROICollectorService.py`)](#roi-collection-subsystem-srcclassifierroicollectorservicepy)
11. [Async classification subsystem (`src/classifier/ClassificationWorker.py`)](#async-classification-subsystem-srcclassifierclassificationworkerpy)
12. [Classification models / platform abstraction (`ClassifierFactory`)](#classification-models--platform-abstraction-classifierfactory)
13. [Bidirectional batch smoothing (`src/tracking/BidirectionalSmoother.py`)](#bidirectional-batch-smoothing-srctrackingbidirectionalsmootherpy)
14. [Counting + persistence (SQLite) (`src/logging/Database.py`)](#counting--persistence-sqlite-srcloggingdatabasepy)
15. [Observability (logs + structured events)](#observability-logs--structured-events)
16. [Spool / segment recording pipeline (RDK-focused) (`src/spool/`)](#spool--segment-recording-pipeline-rdk-focused-srcspool)
17. [Separate RTSP H.264 recorder (`rtsp_h264_recorder.py`)](#separate-rtsp-h264-recorder-rtsp_h264_recorderpy)
18. [API / analytics endpoint (`src/endpoint/`)](#api--analytics-endpoint-srcendpoint)
19. [Configuration reference and tuning guide](#configuration-reference-and-tuning-guide)
20. [Failure modes, edge cases, and operational notes](#failure-modes-edge-cases-and-operational-notes)
21. [Extensibility guide (how to add/change components safely)](#extensibility-guide-how-to-addchange-components-safely)

---

## What the system does (executive summary)

The system counts bread bags moving on a conveyor belt using a computer-vision pipeline:

- **Frame source** reads frames from a video file / camera / RTSP (OpenCV) or ROS2 on RDK.
- **Detector** finds bread bags with a single-class object detector (YOLO style).
- **Tracker** assigns detections to object tracks over time using IoU + velocity prediction.
- **ROI collector** grabs good-quality crops while an object is being tracked.
- When a track completes (object exits or is lost), the **best ROI** is sent to an **async classifier**.
- Classification results can be **batch-smoothed** to remove low-confidence outliers.
- The system **increments counts** and writes events to **SQLite** (`events` table joined with `bag_types`).

Key “v2” design: **classification happens after the track ends** (not per-frame), which keeps the main loop fast and stable for conveyor motion.

Primary orchestration file: `src/app/ConveyorCounterApp.py`.

---

## Core idea: detect → track → classify (after exit) → smooth → count

The README captures the intended architecture:

- Frame Source → Detector → ConveyorTracker
- Track Completed Event → ClassifierService/Worker → Bidirectional Smoother
- Count & Log → Database

In the actual implementation, the responsibilities are split as:

- **`PipelineCore`** (`src/app/pipeline_core.py`): detection + tracking + ROI collection + submission of completed tracks to async classification.
- **`PipelineVisualizer`** (`src/app/pipeline_visualizer.py`): purely drawing/GUI.
- **`ConveyorCounterApp`**: orchestrates everything, handles smoothing + DB recording + lifecycle.

---

## Repository map (where everything lives)

Top-level entry scripts:

- `main.py`: run the counting pipeline.
- `run_endpoint.py`: run the analytics API server (FastAPI).
- `rtsp_h264_recorder.py`: separate FFmpeg-based RTSP recorder.

Main packages under `src/`:

- `src/app/`
  - `ConveyorCounterApp.py`: main orchestrator.
  - `pipeline_core.py`: core pipeline logic.
  - `pipeline_visualizer.py`: GUI overlay/display.
- `src/config/`
  - `settings.py`: `AppConfig` (paths, classes, DB path, platform-aware defaults).
  - `tracking_config.py`: `TrackingConfig` (tracker, ROI quality thresholds, smoothing/spool params).
- `src/frame_source/`: OpenCV / ROS2 frame ingestion.
- `src/detection/`: detector interface + factories + backends.
- `src/tracking/`: tracker and smoother.
- `src/classifier/`: ROI collector + worker thread + classifier interface + backends.
- `src/logging/`: SQLite manager and schema.
- `src/spool/`: RDK spool recording/retention + segment format.
- `src/endpoint/`: FastAPI server and analytics rendering.
- `src/utils/`: logging, platform detection, and generic helpers.

---

## Runtime entrypoints and modes

### Counting pipeline (`main.py`)

`main.py` is a CLI wrapper that:

- reads CLI args
- constructs `AppConfig()` and `TrackingConfig()`
- applies CLI overrides
- constructs `ConveyorCounterApp(...)`
- calls `app.run(...)`

Important CLI switches:

- `--source`: camera index (e.g., `0`), file path, or RTSP URL.
- `--testing`: enables synchronous frame reading (no drops) in the OpenCV frame source.
- `--no-display`: disables OpenCV GUI window.
- `--no-recording`: disables spool/ROI saving behaviors.
- thresholds:
  - `--detection-conf`
  - `--classification-conf` (mapped to `TrackingConfig.high_confidence_threshold`)
- output:
  - `--output-dir` (mapped to `TrackingConfig.spool_dir`)
  - `--database` exists in CLI but note: `main.py` currently does **not** map it to `AppConfig.db_path`. The app uses `self.app_config.db_path` in `ConveyorCounterApp`.

### Endpoint server (`run_endpoint.py`)

Starts `uvicorn` pointing at `src.endpoint.server:app`.

- exposes `/health`
- exposes analytics pages via `src/endpoint/routes/analytics.py`
- serves static mounts for known/unknown class images.

### RTSP recorder (`rtsp_h264_recorder.py`)

Independent process that uses FFmpeg to record the RTSP H264 stream directly to files (rotate/retention supported). It is explicitly documented as **not the spool system**.

---

## End-to-end runtime workflow (step-by-step)

This section describes what happens when you run:

- `python main.py --source ...`

### 0) Startup

`ConveyorCounterApp.run()` calls `_init_components()`:

1. **Frame source**
   - If none supplied, `FrameSourceFactory.create('opencv', source=..., testing_mode=...)` is used.
2. **Detector**
   - `DetectorFactory.create(config=app_config, confidence_threshold=tracking_config.min_detection_confidence)`.
3. **Classifier**
   - `ClassifierFactory.create(config=app_config)`.
4. **Tracker**
   - `ConveyorTracker(config=tracking_config)`.
5. **ROI Collector**
   - `ROICollectorService(quality_config=ROIQualityConfig(...))`.
6. **Async classification worker thread**
   - `ClassificationWorker(...).start()`.
7. **PipelineCore**
   - constructed with detector, tracker, ROI collector, classification worker.
   - sets `PipelineCore.on_track_completed = ConveyorCounterApp._on_classification_completed`.
8. **Visualizer** (optional)
   - constructed only if `enable_display`.
9. **Bidirectional smoother**
   - `BidirectionalSmoother(...)` configured from `TrackingConfig`.
10. **Database**
   - `DatabaseManager(self.app_config.db_path)` loads `schema.sql` and ensures tables exist.

### 1) Main frame loop

`ConveyorCounterApp.run()` loops:

- for each `(frame, latency_ms)` yielded by the frame source:
  1. `annotated = _process_frame(frame)`
  2. `PipelineCore.process_frame(frame)` runs:
     - detection
     - tracking update
     - ROI collection for confirmed tracks
     - emits any completed tracks and submits to background classification
  3. optional: visual overlay + `cv2.imshow` in `PipelineVisualizer`
  4. periodic log every 100 frames

### 2) What “completed track” means

In `ConveyorTracker`, a track is completed when (high level):

- it hasn’t been updated for `max_frames_without_detection`, OR
- it is deemed to have exited the frame (near edge within `exit_margin_pixels`), OR
- it gets dropped by constraints (e.g., too old, too noisy), depending on tracker internals.

When completed, a `TrackEvent` is created with full bbox/position history.

### 3) ROI selection and submission

When `PipelineCore` sees a completed `TrackEvent`:

- it asks `ROICollectorService.get_best_roi(track_id)`
- if none exists: log a warning and cleanup track collection
- else: submit `(track_id, best_roi, bbox_history, callback=...)` to the `ClassificationWorker`

### 4) Background classification

`ClassificationWorker` runs in a daemon thread:

- waits on a bounded queue
- classifies ROI using `BaseClassifier.classify()` (Ultralytics or BPU)
- calls callback with `(track_id, class_name, confidence)`

### 5) Smoothing + counting + DB event

When classification completes, `ConveyorCounterApp._on_classification_completed()` is called (from worker thread).

- It calls `BidirectionalSmoother.add_classification(...)`.
- Two possibilities:
  - Batch finalized → returns list of `ClassificationRecord`s → each is recorded.
  - Not finalized → immediately record the single classification.

Recording (`_record_count`):

- increments in-memory counts (`CounterState.increment_count`, thread-safe lock)
- writes to SQLite using `DatabaseManager.add_event(...)`
  - `timestamp`: ISO string
  - `bag_type_name`: class name
  - `confidence`
  - `track_id`
  - `metadata`: JSON string with vote_ratio/smoothed/original_class
- logs a human-readable COUNT message

### 6) Shutdown

On stop (signal or quit), `_cleanup()`:

- finalizes any remaining smoothing buffer
- cleans core pipeline (`PipelineCore.cleanup()` stops worker, cleans tracker/detector)
- closes frame source
- closes DB
- closes UI windows
- logs final statistics

---

## Data model / key entities (what’s passed between components)

### `Detection` (`src/detection/BaseDetection.py`)

Represents a single detector output:

- `bbox`: `(x1, y1, x2, y2)` integer pixel coordinates
- `confidence`: float
- `class_id`, `class_name` (usually single class `bread-bag`)

Used by `ConveyorTracker.update(detections, frame_shape=...)`.

### `TrackedObject` (`src/tracking/ConveyorTracker.py`)

Represents an active track:

- identifiers and bbox/confidence
- life counters: `age`, `hits`, `time_since_update`
- history deques: `position_history`, `bbox_history`
- motion estimate via `velocity` property (estimated from last N centers)

This is what visualizer draws and what ROI collector uses to crop.

### `TrackEvent` (`src/tracking/ConveyorTracker.py`)

Produced when a track completes:

- `track_id`
- `event_type` (e.g., `track_completed`)
- histories: `bbox_history`, `position_history`
- `total_frames`, `created_at`, `ended_at`
- optional analytics: `avg_confidence`, `exit_direction`

This is what triggers async classification.

### ROI entities (`src/classifier/ROICollectorService.py`)

- `ROIQualityConfig`: thresholds for sharpness/brightness and size.
- `TrackROICollection`: stores many ROI crops + quality measure, tracks `best_roi`.

### Classification entities (`src/classifier/BaseClassifier.py`)

- `ClassificationResult`: `(class_id, class_name, confidence)`
- `EvidenceAccumulator`: **present, but v2’s current async-only path uses a single ROI** (vote_ratio is set to `1.0` in `ConveyorCounterApp`). Evidence accumulation can be reintroduced by submitting multiple ROIs.

### Smoothing entities (`src/tracking/BidirectionalSmoother.py`)

- `ClassificationRecord`: stored in batches, may be overwritten to dominant class.

---

## Frame acquisition subsystem (`src/frame_source/`)

### Factory (`FrameSourceFactory.create`)

- `source_type='opencv'`: creates `OpenCVFrameSource`.
- `source_type='ros2'`: creates `Ros2FrameServer` only on RDK.

### OpenCV source (`OpenCvFrameSource.py`)

Two modes:

1. **Production mode** (default)
   - background thread reads frames and pushes them to a queue
   - the consumer iterates `frames()` and blocks on queue
   - supports pacing to `target_fps`

2. **Testing mode** (`--testing`)
   - synchronous reads from `cv2.VideoCapture.read()`
   - no background queue
   - processes all frames without “producer/consumer skew”

Important behaviors:

- frames are resized to `(1280, 720)` always.
- yields `(frame, inter_frame_ms)` where `inter_frame_ms` is time between reads.

Operational notes:

- For RTSP sources, OpenCV’s reconnect behavior depends on the underlying backend; if you need robust reconnection, the FFmpeg-based recorder approach or a custom OpenCV reconnect loop is safer.

---

## Detection subsystem (`src/detection/`)

### Interface (`BaseDetector`)

`BaseDetector.detect(frame) -> List[Detection]` and `cleanup()`.

### Factory (`DetectorFactory`)

- If `IS_RDK`: return `BpuDetector`.
- Else: return `UltralyticsDetector`.

### Ultralytics detector (`UltralyticsDetector.py`)

- loads `.pt` model via `ultralytics.YOLO(model_path)`
- picks `device` automatically: `cuda` if available else `cpu`
- `detect(frame)` runs model inference and converts output to `Detection` objects.

Key thresholds:

- `confidence_threshold` passed from `TrackingConfig.min_detection_confidence`

---

## Tracking subsystem (`src/tracking/`)

### Goal

Maintain stable identities for bags as they move linearly across the frame.

### ConveyorTracker (`ConveyorTracker.py`)

Core logic (conceptual):

1. **Predict** each track’s next bbox using velocity (center-history based).
2. Build a **cost matrix** where cost = `1 - IoU(predicted_bbox, detection_bbox)`.
3. Run assignment:
   - if SciPy is installed: Hungarian algorithm (`linear_sum_assignment`).
   - else: greedy fallback.
4. Accept matches under threshold where `cost <= (1 - config.iou_threshold)`.
5. Update matched tracks.
6. Create new tracks for unmatched detections.
7. Mark missed tracks. If missed beyond `max_frames_without_detection` or exiting, emit a `TrackEvent`.
8. Enforce constraints like max active tracks.

Exit behavior:

- `_get_exit_direction()` checks track center vs frame edges with `exit_margin_pixels`.

Noise filtering:

- `min_track_duration_frames` prevents counting very short tracks.

Important config knobs (from `TrackingConfig`):

- `iou_threshold`
- `max_frames_without_detection`
- `min_track_duration_frames`
- `exit_margin_pixels`
- plus ROI thresholds and smoothing params that depend on tracker progression.

---

## ROI collection subsystem (`src/classifier/ROICollectorService.py`)

This component is intentionally simple and fast.

### Purpose

During tracking, extract ROI crops that are suitable for classification, but **do not classify them yet**.

### How ROI is collected

`collect_roi(track_id, frame, bbox) -> bool`:

- crops bbox with padding
- rejects empty crops
- computes quality:
  - sharpness (Laplacian variance via `compute_sharpness`)
  - brightness (mean intensity via `compute_brightness`)
  - size thresholds
- stores ROI in per-track collection
- tracks the best ROI by quality score

Cleanup behavior:

- stale track collections are removed after `STALE_TIMEOUT_SECONDS`.
- hard cap on number of active collections (`MAX_TRACKS`).

Design tradeoff:

- quality score is currently primarily sharpness; you can incorporate area, brightness closeness, or motion stability if needed.

---

## Async classification subsystem (`src/classifier/ClassificationWorker.py`)

### Purpose

Classification can be slow (model inference). To keep the main loop stable, classification runs on a background thread.

### How it works

- `submit_job()` pushes a job onto a bounded queue.
- `_worker_loop()` consumes jobs, runs `classifier.classify(roi)`, and calls a callback.

Backpressure:

- if queue is full, job is dropped and a warning is logged.

Thread-safety:

- ROI is copied on submission to avoid main-thread memory reuse issues.

---

## Classification models / platform abstraction (`ClassifierFactory`)

### Interface (`BaseClassifier`)

- `classify(roi) -> ClassificationResult`
- `classify_batch(rois) -> List[ClassificationResult]`

### Factory (`ClassifierFactory`)

- RDK → `BpuClassifier`
- otherwise → `UltralyticsClassifier`

### Ultralytics classifier (`UltralyticsClassifier.py`)

- loads classification `.pt` model via `ultralytics.YOLO`
- reads top-1 class from `result.probs` when available
- maps class_id to configured class list from `AppConfig.classifier_classes`

Note on class config type:

- In `AppConfig` (`src/config/settings.py`), `classifier_classes` is a dict mapping int → str.
- In `ClassifierFactory.create`, `classes` is treated as a `List[str]` by wrappers.

So the intended shapes should be unified (either always list or always dict) to avoid subtle indexing problems.

---

## Bidirectional batch smoothing (`src/tracking/BidirectionalSmoother.py`)

### Purpose

When classification confidence is low, single predictions can produce outliers. The smoother corrects some outliers based on batch context.

### How it works

- Collect classifications into a batch of size `batch_size` or until timeout.
- Determine the **dominant class** by confidence-weighted sum.
- If dominant class exceeds `min_batch_dominance`, then for each low-confidence record:
  - if the record’s class appears only once in the batch, override it to dominant class.

### Inputs from the app

`ConveyorCounterApp` currently uses:

- `vote_ratio=1.0` (because it submits only one ROI per track → no voting yet)
- smoothing thresholds from `TrackingConfig`:
  - `bidirectional_confidence_threshold`
  - `evidence_ratio_threshold`
  - `bidirectional_buffer_size`
  - `bidirectional_inactivity_timeout_ms`

---

## Counting + persistence (SQLite) (`src/logging/Database.py`)

### Database schema

Schema file: `src/logging/schema.sql`

Tables:

- `bag_types`: catalog of classes with metadata (thumb path, weight, etc.)
- `events`: each counted bag event, FK to `bag_types`
- `config`: optional key/value store

### How events are written

`ConveyorCounterApp._record_count()` does:

- `DatabaseManager.add_event(timestamp, bag_type_name, confidence, track_id, metadata=JSON)`
- `DatabaseManager.get_or_create_bag_type(name, ...)` ensures bag type exists

Threading model:

- `DatabaseManager` uses `threading.local()` for per-thread connections.
- This matters because counts are recorded from the classification worker callback thread.

Operational notes:

- SQLite is safe for this pattern if writes are modest; heavy write concurrency may require batching or a single-writer queue.

---

## Observability (logs + structured events)

Logging is centralized in `src/utils/AppLogging.py`.

- Logs to `data/logs/conveyer_counter_<timestamp>.log`
- Console logs are INFO+.

There is also a `StructuredLogger` that can emit JSON events like:

- `track_created`, `track_updated`, `track_completed`
- `classification_result`, `classification_candidate`
- `smoothing_applied`

Not all modules currently emit structured events, but the facility exists.

---

## Spool / segment recording pipeline (RDK-focused) (`src/spool/`)

This is primarily for the RDK ROS2 environment and is conceptually separate from the OpenCV pipeline.

### Segment file format (`segment_io.py`)

- Segments are `.bin` files with a compact binary format.
- Atomic safety: written as `.tmp`, then renamed to `.bin` on finalize.
- Optional `.json` metadata file per segment.

Key classes:

- `FrameRecord`: one encoded frame plus DTS/PTS timing.
- `SegmentWriter`: writes frames with rotation (duration / IDR aligned).
- `SegmentReader`: reads segments back (not fully shown in excerpt but referenced by processor).

### Recorder node (`spool_recorder_node.py`)

- ROS2 subscriber node that listens for `H26XFrame` frames.
- Extracts SPS/PPS and identifies IDR frames.
- Writes frames through `SegmentWriter`.

### Processor node (`spool_processor_node.py`)

- Reads frames from segments with `SegmentReader`.
- Publishes frames as `H26XFrame` to a topic for decode.
- Supports playback modes (realtime/fast/adaptive/catchup).
- Maintains crash-recovery state in `processor_state.json`.
- Integrates with `RetentionPolicy` to delete processed segments.

### Retention (`retention.py`)

- Background thread deletes segments based on:
  - max age
  - storage budget
  - only-delete-processed policy
  - keep N newest segments

---

## Separate RTSP H.264 recorder (`rtsp_h264_recorder.py`)

This is a standalone utility for production recording:

- uses FFmpeg `-c:v copy` to avoid re-encode
- supports file rotation (`--rotate-hours`) and retention (`--retention-hours`)
- is explicitly different from the spool segment design.

When to use it:

- you want long-duration standard video files for archiving
- you don’t need frame-level spool playback/processing semantics

---

## API / analytics endpoint (`src/endpoint/`)

### Overview

This is a FastAPI app that serves:

- `/health` status
- `/analytics` HTML-based report pages
- static mounts:
  - `/known_classes` → `data/classes`
  - `/unknown_classes` → `data/unknown`
  - `/static` → `src/endpoint/static` (if present)

### Module breakdown

- `src/endpoint/server.py`: FastAPI app + lifespan hooks + static mounts.
- `src/endpoint/routes/analytics.py`: HTTP routes.
- `src/endpoint/services/analytics_service.py`: business logic.
- `src/endpoint/repositories/analytics_repository.py`: database queries.
- `src/endpoint/shared.py`: shared DB/Templates initialization (not included in the snippets but referenced).

### Analytics data flow

1. Route parses `start_time` and `end_time`.
2. Service converts them to datetimes and applies shift/timezone offsets.
3. Repository calls:
   - `DatabaseManager.get_aggregated_stats()`
   - `DatabaseManager.get_events_with_bag_types()`
4. Service builds:
   - class summaries
   - event timeline
   - “runs” of consecutive bag types, filtering short runs as noise
5. Service normalizes thumbnail/paths for web static mounts.
6. Templates render HTML.

---

## Configuration reference and tuning guide

There are two primary configuration dataclasses:

### `AppConfig` (`src/config/settings.py`)

- platform-aware model paths:
  - `detection_model`
  - `classification_model`
- DB location: `db_path`
- default class dictionaries:
  - `classifier_classes: Dict[int, str]`
  - `detector_classes: Dict[int, str]`

Environment variables:

- `VIDEO_PATH`
- `DETECTION_MODEL`
- `CLASS_MODEL`
- `DB_PATH`

### `TrackingConfig` (`src/config/tracking_config.py`)

Detection:

- `min_detection_confidence`

Tracking:

- `iou_threshold`
- `max_frames_without_detection`
- `min_track_duration_frames`
- `max_active_tracks`
- `exit_margin_pixels`

ROI quality:

- `min_sharpness`
- `min_mean_brightness`, `max_mean_brightness`
- ROI size/aspect constraints

Classification decision thresholds:

- `min_candidates_for_classification`
- `evidence_ratio_threshold`
- `high_confidence_threshold`
- `reject_labels`

Smoothing:

- `bidirectional_buffer_size`
- `bidirectional_confidence_threshold`
- `bidirectional_inactivity_timeout_ms`

Spool:

- `spool_dir`
- `spool_segment_duration`
- `spool_retention_seconds`

### Practical tuning scenarios

1. **Missed detections**
   - decrease `min_detection_confidence`
   - lower `iou_threshold` slightly if tracks fail to associate
2. **Track fragmentation (one bag becomes multiple tracks)**
   - increase `max_frames_without_detection`
   - improve detector stability or reduce blur
3. **Wrong classifications**
   - ensure ROI quality thresholds aren’t rejecting most good crops
   - increase smoothing batch size to stabilize, but watch for real mix changes
4. **Latency / throughput**
   - keep classification async (default)
   - consider GPU device for Ultralytics

---

## Failure modes, edge cases, and operational notes

### 1) Classification thread safety

Classification callback runs in the worker thread.

- `CounterState.increment_count` is locked.
- `DatabaseManager` uses a per-thread SQLite connection.

But be aware:

- if you later add UI updates or OpenCV calls in the callback, you may need to marshal back to the main thread.

### 2) Queue full → dropped classifications

If `ClassificationWorker` queue fills, tracks are dropped.

Symptoms:

- warning logs: `Queue full! Dropped track ...`

Mitigation:

- increase queue size
- reduce classification cost
- submit fewer jobs (e.g., only classify confirmed/long tracks)

### 3) OpenCV capture issues (RTSP)

OpenCV may fail to open or may stall on RTSP streams.

Mitigation:

- prefer FFmpeg-based ingestion or wrap OpenCV reconnect logic if required.

### 4) Disk usage

Spool segments can grow quickly.

- ensure `RetentionPolicy` is configured appropriately on RDK
- set spool_dir to a volume with sufficient capacity

### 5) DB file path mismatch

`main.py` has a `--database` argument, but the current `ConveyorCounterApp` uses `AppConfig.db_path`.

If you rely on CLI DB path, you should wire `args.database → app_config.db_path`.

---

## Extensibility guide (how to add/change components safely)

### Add a new detector backend

1. Create a new class implementing `BaseDetector`.
2. Add a factory branch in `DetectorFactory.create()` based on platform/env.
3. Ensure output is `List[Detection]` with integer bbox coords.

### Add a new classifier backend

1. Implement `BaseClassifier`.
2. Register in `ClassifierFactory.create()`.
3. Ensure class mapping aligns with `AppConfig.classifier_classes`.

### Change how classification works (multi-ROI evidence)

Currently, only best ROI is classified.

To reintroduce evidence accumulation:

- have `PipelineCore` submit multiple ROIs (`get_all_rois`) to classification worker
- either:
  - classify batch and then vote in worker
  - or submit multiple jobs and accumulate in an `EvidenceAccumulator` keyed by track_id

### Change counting logic

Counting occurs in `_record_count()`.

You can add:

- reject label filtering (`TrackingConfig.reject_label_set`)
- minimum confidence threshold before writing to DB
- additional metadata fields (e.g., exit direction from `TrackEvent`)

---

## Quick “read this first” file list

If you’re new and want the fastest path to understand everything:

1. `README.md`
2. `main.py`
3. `src/app/ConveyorCounterApp.py`
4. `src/app/pipeline_core.py`
5. `src/tracking/ConveyorTracker.py`
6. `src/classifier/ROICollectorService.py`
7. `src/classifier/ClassificationWorker.py`
8. `src/logging/Database.py` and `src/logging/schema.sql`
9. `src/endpoint/server.py` + `src/endpoint/services/analytics_service.py`
10. `src/spool/segment_io.py` (if you’re on RDK)
