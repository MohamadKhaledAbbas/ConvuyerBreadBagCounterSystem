# Installation Guide - Conveyor Bread Bag Counter System v2

## 📋 Overview

This project supports two platforms with different hardware acceleration:

| Platform | Acceleration | Models | FPS |
|----------|-------------|--------|-----|
| **Windows** | NVIDIA GPU (CUDA) | .pt (PyTorch) | 15-18 |
| **RDK X5** | Horizon BPU | .bin (BPU) | 17-21 |

---

## 🪟 Windows Installation (with GPU)

### Prerequisites

- Windows 10/11
- Python 3.8+
- NVIDIA GPU (GTX 1060 or better)
- NVIDIA Drivers (latest)
- CUDA Toolkit 11.8 or 12.1

### Step 1: Install NVIDIA Drivers & CUDA

1. **Check if you have NVIDIA GPU:**
   ```bash
   nvidia-smi
   ```
   Should show your GPU model.

2. **Install NVIDIA Drivers:**
   - Download from: https://www.nvidia.com/Download/index.aspx
   - Install latest Game Ready or Studio drivers

3. **CUDA is optional** (PyTorch includes it)
   - PyTorch bundles CUDA runtime
   - No separate CUDA toolkit installation needed

### Step 2: Create Virtual Environment

```bash
# Clone repository
git clone <your-repo-url>
cd ConvuyerBreadBagCounterSystem

# Create virtual environment
python -m venv .venv

# Activate (PowerShell)
.\.venv\Scripts\Activate.ps1

# Or activate (CMD)
.\.venv\Scripts\activate.bat
```

### Step 3: Install Dependencies

```bash
# Install all Windows dependencies
pip install -r requirements-windows.txt
```

**Note:** This will install PyTorch with CUDA support automatically.

### Step 4: Verify GPU Detection

```bash
python check_gpu.py
```

**Expected output:**
```
✓ CUDA available: True
✓ GPU 0: NVIDIA GeForce RTX 3070
GPU speedup: 5-7x
```

**If CUDA shows False:**
```bash
# Reinstall PyTorch with CUDA
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Step 5: Test System

```bash
# Run on video file
python main.py --source test_video.mp4

# Run on webcam
python main.py --source 0

# Check logs for GPU usage:
# [UltralyticsDetector] Model loaded, device: cuda ✓
# [UltralyticsClassifier] Model loaded, device: cuda ✓
```

---

## 🤖 RDK X5 Installation (with BPU)

### Prerequisites

- Horizon RDK X5 board
- Ubuntu 22.04 (pre-installed)
- Python 3.8+ (pre-installed)
- hobot_dnn (pre-installed system package)

### Step 1: System Update

```bash
# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install pip if not present
sudo apt-get install python3-pip -y
```

### Step 2: Clone Repository

```bash
cd ~
git clone <your-repo-url>
cd ConvuyerBreadBagCounterSystem
```

### Step 3: Install Python Dependencies

```bash
# Install Python packages (use --break-system-packages on RDK)
pip3 install -r requirements-rdk.txt --break-system-packages
```

**Note:** `--break-system-packages` is needed because RDK uses system Python.

### Step 4: Verify BPU Availability

```bash
# Check hobot_dnn
python3 -c "from hobot_dnn import pyeasy_dnn as dnn; print('BPU available:', dnn is not None)"

# Check ROS2 (if using ROS2 mode)
python3 -c "import rclpy; print('ROS2 available')"
```

**Expected:** Both should print success messages.

### Step 5: Verify Models

```bash
# Check BPU models exist
ls -lh data/model/*.bin

# Should show:
# detect_yolo_small_v9_bayese_640x640_nv12.bin
# classify_yolo_small_v11_bayese_224x224_nv12.bin
```

### Step 6: Test System

```bash
# Run with camera
python3 main.py --source 0

# Run with ROS2 (if configured)
python3 main.py --source ros2

# Check logs for BPU usage:
# [BpuDetector] Model loaded successfully
# [BpuClassifier] Model loaded, classes: 10
```

---

## 🔄 Migrating from V1

If you have an existing V1 installation with working GPU support:

### Option 1: Copy Virtual Environment

```bash
# From V1 project
cd path/to/v1/project

# Copy .venv to V2
xcopy /E /I .venv C:\path\to\v2\project\.venv

# Activate in V2
cd C:\path\to\v2\project
.\.venv\Scripts\Activate.ps1

# Verify
python check_gpu.py
```

### Option 2: Fresh Install (Recommended)

```bash
# In V2 project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-windows.txt
python check_gpu.py
```

**Option 2 is cleaner** and ensures all dependencies match V2 requirements.

---

## 📦 Package Differences

### Windows (`requirements-windows.txt`)

**Required:**
- `opencv-python` - Computer vision
- `numpy` - Numerical computing
- `ultralytics` - YOLO framework
- `torch` - PyTorch with CUDA
- `torchvision` - PyTorch vision utilities
- `scipy` - Scientific computing (Hungarian algorithm)
- `fastapi` - Web API framework
- `uvicorn` - ASGI server
- `jinja2` - Template engine
- `pydantic` - Data validation
- `python-multipart` - File uploads

**Not Needed:**
- ❌ `hobot_dnn` (RDK only)
- ❌ `rclpy` (ROS2, not needed on Windows)
- ❌ `Pillow` (not used in V2)

### RDK (`requirements-rdk.txt`)

**Required:**
- `opencv-python` - Computer vision
- `numpy` - Numerical computing
- `scipy` - Scientific computing
- `fastapi` - Web API framework
- `uvicorn` - ASGI server
- `jinja2` - Template engine
- `pydantic` - Data validation
- `python-multipart` - File uploads

**Pre-installed (System Packages):**
- `hobot_dnn` - BPU inference engine
- `rclpy` - ROS2 Python client
- `launch`, `launch_ros` - ROS2 launch

**Not Needed:**
- ❌ `torch` (uses BPU, not PyTorch)
- ❌ `torchvision` (not needed)
- ❌ `ultralytics` (uses .bin models, not .pt)
- ❌ `Pillow` (not used)

---

## 🧪 Verification Tests

### After Installation - Windows

```bash
# 1. Check Python version
python --version  # Should be 3.8+

# 2. Check GPU
python check_gpu.py
# Expected: CUDA available: True

# 3. Check packages
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
python -c "from ultralytics import YOLO; print('Ultralytics: OK')"
python -c "import fastapi; print('FastAPI:', fastapi.__version__)"

# 4. Run system
python main.py --source 0 --max-frames 100
# Expected: 15-18 FPS, GPU usage in nvidia-smi
```

### After Installation - RDK

```bash
# 1. Check Python version
python3 --version  # Should be 3.8+

# 2. Check BPU
python3 -c "from hobot_dnn import pyeasy_dnn; print('BPU: OK')"

# 3. Check packages
python3 -c "import cv2; print('OpenCV:', cv2.__version__)"
python3 -c "import numpy; print('NumPy:', numpy.__version__)"
python3 -c "import fastapi; print('FastAPI: OK')"

# 4. Run system
python3 main.py --source 0 --max-frames 100
# Expected: 17-21 FPS, BPU usage visible
```

---

## 🐛 Troubleshooting

### Windows: GPU Not Detected

**Symptom:**
```
[UltralyticsDetector] Model loaded, device: cpu
```

**Fix:**
```bash
# 1. Check NVIDIA driver
nvidia-smi

# 2. Reinstall PyTorch with CUDA
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. Verify
python -c "import torch; print(torch.cuda.is_available())"
```

### Windows: Slow Performance (Still on CPU)

**Check:**
- GPU detected but not used
- Models loading on CPU

**Fix:**
```bash
# Force GPU in code (temporary test)
# Edit src/detection/DetectorFactory.py and src/classifier/ClassifierFactory.py
# Add: device='cuda' parameter

# Or set environment variable
set CUDA_VISIBLE_DEVICES=0
python main.py --source video.mp4
```

### RDK: BPU Not Available

**Symptom:**
```
ImportError: No module named 'hobot_dnn'
```

**Fix:**
```bash
# hobot_dnn is a system package, not pip installable
# Verify it's in system Python
python3 -c "import sys; print(sys.path)"

# Should include: /usr/lib/python3/dist-packages

# If missing, reinstall RDK system image
```

### RDK: Permission Errors

**Symptom:**
```
error: externally-managed-environment
```

**Fix:**
```bash
# Use --break-system-packages flag
pip3 install -r requirements-rdk.txt --break-system-packages

# Or use virtual environment (not recommended on RDK)
```

---

## 📊 Performance Benchmarks

### Windows (RTX 3070, i7-10700K)
- Detection: 35ms
- Classification: 8ms
- Tracking: 10ms
- **Total: ~60ms → 16 FPS** ✅

### RDK X5 (BPU)
- Detection: 32ms (BPU)
- Classification: 8ms (BPU)
- Tracking: 10ms
- **Total: ~50ms → 20 FPS** ✅

---

## 🎯 Next Steps

After successful installation:

1. **Test with sample video:**
   ```bash
   python main.py --source test_video.mp4 --testing
   ```

2. **Start analytics endpoint:**
   ```bash
   python run_endpoint.py
   # Visit: http://localhost:8000/analytics
   ```

3. **Configure for production:**
   - Update `src/config/settings.py`
   - Set correct model paths
   - Configure database path

4. **Deploy:**
   - Windows: Run as service
   - RDK: Set up systemd service

---

## 📚 Additional Resources

- **GPU Setup:** See `docs/CRITICAL_GPU_FIX.md`
- **Performance:** See `docs/ASYNC_CLASSIFICATION_IMPLEMENTATION.md`
- **Architecture:** See `README.md`

Need help? Check the logs in `data/logs/` for detailed error messages.
