# Conveyor Bread Bag Counter System v2

A simplified, production-ready bread bag counting system optimized for conveyor belt environments.

## Overview

This is **v2** of the BreadBagCounterSystem, redesigned for simpler conveyor belt scenarios where:

- Bread bags move linearly across the frame
- No complex worker interactions or occlusions
- Predictable movement patterns (enter → traverse → exit)

### Key Differences from v1

| Feature | v1 (Original) | v2 (Conveyor) |
|---------|---------------|---------------|
| **Tracking** | EventCentricTracker (~3300 lines) | ConveyorTracker (~500 lines) |
| **State Machine** | open/closing/closed states | None (track → classify → count) |
| **Association** | Parallel IoU + centroid | IoU with velocity prediction |
| **Classification** | During track lifetime | After track completes |
| **Movement** | Chaotic (table/worker) | Linear (conveyor belt) |
| **Complexity** | ~2300 lines (BagCounterApp) | ~600 lines (ConveyorCounterApp) |

### Preserved from v1

- ✅ ROS2 pub/sub support (RDK platform)
- ✅ Spool on disk (video recording with retention)
- ✅ Bidirectional batch smoothing for low confidence
- ✅ Evidence accumulation for classification
- ✅ BPU acceleration on RDK hardware
- ✅ SQLite event logging
- ✅ Reject label filtering

## Architecture

```
┌─────────────┐     ┌───────────┐     ┌──────────────────┐
│ Frame Source│────▶│ Detector  │────▶│ ConveyorTracker  │
│ (OpenCV/ROS)│     │ (YOLO)    │     │ (IoU-based)      │
└─────────────┘     └───────────┘     └────────┬─────────┘
                                               │
                    ┌─────────────────────────▼─────────┐
                    │     Track Completed Event         │
                    └─────────────────────────┬─────────┘
                                              │
┌───────────────┐     ┌─────────────────┐    │
│ Bidirectional │◀────│ ClassifierService│◀───┘
│   Smoother    │     │ (Evidence Accum.)│
└───────┬───────┘     └─────────────────┘
        │
        ▼
┌───────────────┐
│ Count & Log   │
│ (Database)    │
└───────────────┘
```

## Installation

### Prerequisites

- Python 3.8+
- OpenCV
- For Windows/Development: Ultralytics YOLO
- For RDK: hobot_dnn, rclpy

### Install Dependencies

```bash
# Clone repository
git clone <your-repo-url>
cd ConvuyerBreadBagCounterSystem

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Model Files

Place your model files in the `data/model/` directory:

```
data/model/
├── detect_yolo_small_v9.pt                    # Ultralytics detection (Windows)
├── classify_yolo_small_v11.pt                 # Ultralytics classifier (Windows)
├── detect_yolo_small_v9_bayese_640x640_nv12.bin   # BPU detection (RDK)
└── classify_yolo_small_v11_bayese_224x224_nv12.bin # BPU classifier (RDK)
```

## Usage

### Basic Usage

```bash
# Run the counter application (config read from database)
python main.py
```

### Configuration

All runtime configuration is stored in the database (`data/db/bag_events.db`) in the `config` table.

#### View Current Configuration

```bash
sqlite3 data/db/bag_events.db "SELECT * FROM config ORDER BY key;"
```

#### Enable Display

```bash
sqlite3 data/db/bag_events.db "UPDATE config SET value='1' WHERE key='enable_display';"
```

#### Enable RTSP Recording

```bash
# Enable recording
sqlite3 data/db/bag_events.db "UPDATE config SET value='1' WHERE key='enable_recording';"

# Configure RTSP camera
sqlite3 data/db/bag_events.db "UPDATE config SET value='192.168.1.100' WHERE key='rtsp_host';"
sqlite3 data/db/bag_events.db "UPDATE config SET value='admin' WHERE key='rtsp_username';"
sqlite3 data/db/bag_events.db "UPDATE config SET value='password' WHERE key='rtsp_password';"

# Run recorder (separate process)
python rtsp_h264_recorder.py
```

#### Configuration Keys

| Key | Default | Description |
|-----|---------|-------------|
| `enable_display` | `0` | Enable/disable OpenCV display window (1=on, 0=off) |
| `enable_recording` | `0` | Enable/disable RTSP recording (1=on, 0=off) |
| `rtsp_host` | `192.168.2.108` | RTSP camera IP address |
| `rtsp_port` | `554` | RTSP port |
| `rtsp_username` | `admin` | RTSP username |
| `rtsp_password` | `a1234567` | RTSP password |
| `show_ui_screen` | `0` | Legacy UI flag |
| `is_development` | `0` | Development mode flag |
| `is_profiler_enabled` | `0` | Enable profiling |

### RTSP Recording

Recording is handled by a separate process (`rtsp_h264_recorder.py`) that:
- Reads configuration from database
- Records H.264 stream directly (no transcoding)
- Supports automatic file rotation
- Supports retention policies

```bash
# Run with database config
python rtsp_h264_recorder.py

# Or override with command line
python rtsp_h264_recorder.py --url rtsp://192.168.1.100:554/stream
```

### Process video file
python main.py --source video.mp4

# Use webcam
python main.py --source 0

# RTSP stream
python main.py --source "rtsp://user:pass@192.168.1.100:554/stream"
```

### Testing Mode

Process all frames without drops (useful for analysis):

```bash
python main.py --source video.mp4 --testing
```

### Headless Mode

Run without display window (for production/server):

```bash
python main.py --source video.mp4 --no-display
```

### Advanced Options

```bash
python main.py \
    --source video.mp4 \
    --detector-model models/my_detector.pt \
    --classifier-model models/my_classifier.pt \
    --detection-conf 0.6 \
    --classification-conf 0.7 \
    --output-dir ./recordings \
    --database ./counts.db
```

## Configuration

### Application Config (`src/config/settings.py`)

```python
@dataclass
class AppConfig:
    # Model paths (platform-specific)
    detection_model_path: str
    classifier_model_path: str
    
    # Classifier classes
    classifier_classes: List[str] = [
        "Semsemya", "Kaak", "Mafrood",
        "Kaizerli", "Finosamit", "Tanoor",
        "Siyahi", "Rejected"
    ]
    
    # Single detection class for conveyor
    detector_classes: List[str] = ["bread-bag"]
```

### Tracking Config (`src/config/tracking_config.py`)

```python
@dataclass
class TrackingConfig:
    # Tracker settings
    iou_threshold: float = 0.3
    max_age: int = 30  # Frames before track is lost
    min_hits: int = 3  # Min detections to confirm track
    
    # Classification
    min_evidence_samples: int = 3
    min_vote_ratio: float = 0.5
    
    # Bidirectional smoothing
    smoothing_batch_size: int = 10
    smoothing_confidence_threshold: float = 0.7
```

## Project Structure

```
ConvuyerBreadBagCounterSystem/
├── main.py                     # Entry point
├── requirements.txt            # Dependencies
├── README.md                   # This file
│
├── src/
│   ├── app/
│   │   └── ConveyorCounterApp.py   # Main orchestrator
│   │
│   ├── config/
│   │   ├── settings.py             # App configuration
│   │   └── tracking_config.py      # Tracking parameters
│   │
│   ├── detection/
│   │   ├── BaseDetection.py        # Detector interface
│   │   ├── BpuDetector.py          # RDK BPU detector
│   │   ├── UltralyticsDetector.py  # Ultralytics detector
│   │   └── DetectorFactory.py      # Platform factory
│   │
│   ├── classifier/
│   │   ├── BaseClassifier.py       # Classifier interface
│   │   ├── BpuClassifier.py        # RDK BPU classifier
│   │   ├── UltralyticsClassifier.py # Ultralytics classifier
│   │   ├── ClassifierService.py    # Evidence accumulation
│   │   └── ClassifierFactory.py    # Platform factory
│   │
│   ├── tracking/
│   │   ├── ConveyorTracker.py      # Simple IoU tracker
│   │   └── BidirectionalSmoother.py # Batch smoothing
│   │
│   ├── frame_source/
│   │   ├── FrameSource.py          # Source interface
│   │   ├── OpenCvFrameSource.py    # OpenCV source
│   │   ├── Ros2FrameServer.py      # ROS2 source
│   │   └── FrameSourceFactory.py   # Source factory
│   │
│   ├── spool/
│   │   └── segment_io.py           # Recording & retention
│   │
│   ├── logging/
│   │   ├── Database.py             # SQLite logging
│   │   └── ConfigWatcher.py        # Config hot-reload
│   │
│   └── utils/
│       ├── platform.py             # Platform detection
│       ├── AppLogging.py           # Structured logging
│       ├── Utils.py                # Utility functions
│       └── PipelineMetrics.py      # Performance metrics
│
└── models/                     # Model files (not in git)
```

## How It Works

### 1. Detection

Single-class YOLO detection for "bread-bag" objects. Unlike v1 which detected open/closing/closed states, v2 simply detects the presence of bread bags.

### 2. Tracking

Simple IoU-based tracker with velocity prediction:
- New detection → match to existing tracks by IoU
- Unmatched detection → create new track
- Track exits frame or exceeds max_age → emit completion event

### 3. Classification

Classification happens when a track completes:
1. During track lifetime: collect ROI crops, filter by quality
2. Each qualifying ROI → classify → add to evidence accumulator
3. Track completes → get final class from accumulated evidence (majority vote)

### 4. Bidirectional Smoothing

For low-confidence classifications, batch-level smoothing corrects outliers:
- Accumulate classifications into batches
- Find dominant class in batch
- Low-confidence outliers matching only once → override to dominant class


## Database Schema

```sql
-- Classification events
CREATE TABLE events (
    id INTEGER PRIMARY KEY,
    track_id INTEGER,
    bag_type TEXT,
    confidence REAL,
    timestamp DATETIME,
    phash TEXT,
    image_path TEXT,
    candidates_count INTEGER,
    metadata TEXT
);

-- Configuration changes
CREATE TABLE config_changes (
    id INTEGER PRIMARY KEY,
    key TEXT,
    old_value TEXT,
    new_value TEXT,
    timestamp DATETIME
);

-- Performance metrics
CREATE TABLE metrics (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME,
    fps REAL,
    detection_time_ms REAL,
    tracking_time_ms REAL,
    classification_time_ms REAL,
    active_tracks INTEGER,
    total_counted INTEGER
);
```

## License

MIT License - See LICENSE file for details.

## Related

- [BreadBagCounterSystem (v1)](https://github.com/MohamadKhaledAbbas/BreadBagCounterSystem) - Original system for chaotic table environments