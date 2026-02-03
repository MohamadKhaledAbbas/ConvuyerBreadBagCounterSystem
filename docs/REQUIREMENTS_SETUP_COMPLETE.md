# ✅ Requirements Files Created - Platform-Specific Dependencies

## 📁 Files Created

### 1. **`requirements-windows.txt`** - Windows with GPU
Clean, minimal dependencies for Windows development with NVIDIA CUDA support.

**Includes:**
- ✅ `opencv-python` (4.8.0+)
- ✅ `numpy` (1.24.0+)
- ✅ `ultralytics` (8.0.0+) - YOLO framework
- ✅ `torch` (2.0.0+) - **PyTorch with CUDA**
- ✅ `torchvision` (0.15.0+)
- ✅ `scipy` (1.10.0+) - For Hungarian algorithm
- ✅ `fastapi` (0.100.0+) - Web API
- ✅ `uvicorn` (0.23.0+) - ASGI server
- ✅ `jinja2` (3.1.0+) - Templates
- ✅ `pydantic` (2.0.0+) - Data validation
- ✅ `python-multipart` (0.0.6+)

**Excludes (not used):**
- ❌ `Pillow` - Not actually used in codebase
- ❌ `hobot_dnn` - RDK only
- ❌ `rclpy` - ROS2, not needed on Windows

---

### 2. **`requirements-rdk.txt`** - RDK X5 with BPU
Minimal dependencies for Horizon RDK X5 platform with BPU acceleration.

**Includes:**
- ✅ `opencv-python` (4.8.0+)
- ✅ `numpy` (1.24.0+)
- ✅ `scipy` (1.10.0+)
- ✅ `fastapi` (0.100.0+)
- ✅ `uvicorn` (0.23.0+)
- ✅ `jinja2` (3.1.0+)
- ✅ `pydantic` (2.0.0+)
- ✅ `python-multipart` (0.0.6+)

**Excludes (not needed on RDK):**
- ❌ `torch` - Uses BPU, not PyTorch
- ❌ `torchvision` - Not needed
- ❌ `ultralytics` - Uses .bin models, not .pt
- ❌ `Pillow` - Not used

**System Packages (pre-installed on RDK):**
- `hobot_dnn` - BPU inference engine
- `rclpy` - ROS2 Python client
- `launch`, `launch_ros` - ROS2 launch system

---

### 3. **`requirements.txt`** - Updated with Instructions
Redirects to platform-specific files with clear instructions.

---

### 4. **`INSTALLATION.md`** - Complete Installation Guide
Comprehensive setup guide for both platforms with troubleshooting.

---

## 🎯 What Changed from Original

### **Removed Unused Packages:**
- ❌ `Pillow` - Not imported anywhere in the codebase

### **Added Version Constraints:**
- ✅ Upper bounds on all packages (e.g., `<5.0.0`)
- ✅ Prevents breaking changes from major updates
- ✅ More stable for production

### **Organized by Category:**
- Core Dependencies
- Machine Learning
- Web API
- Database
- Platform-specific notes

### **Added Comments:**
- Installation instructions
- GPU setup notes
- Troubleshooting tips
- Performance expectations

---

## 📊 Package Analysis

### Actual Imports Found in Codebase:

```python
# External packages actually used:
cv2                  # opencv-python
numpy                # numpy
torch                # torch (Windows only)
ultralytics          # ultralytics (Windows only)
scipy                # scipy (Hungarian algorithm)
fastapi              # fastapi
uvicorn              # uvicorn
jinja2               # jinja2 (via fastapi.templating)
pydantic             # pydantic
hobot_dnn            # hobot_dnn (RDK only, system package)
rclpy                # rclpy (RDK only, system package)
launch, launch_ros   # ROS2 (RDK only, system packages)
```

### Standard Library (no install needed):
```python
os, sys, time, json, queue, threading, dataclasses, typing,
pathlib, datetime, collections, abc, enum, contextlib, hashlib,
statistics, signal, struct, zlib, argparse, logging, sqlite3
```

---

## 🚀 Installation

### Windows (GPU):
```bash
# Create venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install
pip install -r requirements-windows.txt

# Verify GPU
python check_gpu.py

# Should show:
# ✓ CUDA available: True
# ✓ GPU 0: NVIDIA GeForce RTX...
```

### RDK X5:
```bash
# Install
pip3 install -r requirements-rdk.txt --break-system-packages

# Verify BPU
python3 -c "from hobot_dnn import pyeasy_dnn; print('BPU OK')"

# Should print: BPU OK
```

---

## 🔧 For You (Migration from V1)

You mentioned you have a working V1 with GPU. Here are your options:

### Option 1: Copy .venv from V1 (Quick)
```bash
# From V1 directory
xcopy /E /I .venv C:\path\to\v2\.venv

# In V2
.\.venv\Scripts\Activate.ps1
python check_gpu.py  # Should still work
```

**Pros:** Fast, keeps working GPU setup  
**Cons:** May have extra packages from V1

### Option 2: Fresh Install (Recommended)
```bash
# In V2
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-windows.txt
python check_gpu.py
```

**Pros:** Clean, only V2 dependencies  
**Cons:** Takes 5-10 minutes to install

### Option 3: Update Existing .venv
```bash
# Copy V1 venv
xcopy /E /I ..\v1\.venv .\.venv

# Update with V2 requirements
.\.venv\Scripts\Activate.ps1
pip install -r requirements-windows.txt --upgrade

# Clean unused packages
pip uninstall Pillow -y  # If it was in V1
```

**Pros:** Best of both - keeps GPU, removes unused  
**Cons:** Slight risk of conflicts

---

## ✅ Verification

After installation, run:

```bash
# Check all packages
python -c "
import cv2
import numpy
import torch
import ultralytics
import scipy
import fastapi
import uvicorn

print('OpenCV:', cv2.__version__)
print('NumPy:', numpy.__version__)
print('PyTorch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
print('Ultralytics:', ultralytics.__version__)
print('SciPy:', scipy.__version__)
print('FastAPI:', fastapi.__version__)
print('Uvicorn:', uvicorn.__version__)
print('All packages OK!')
"
```

**Expected output:**
```
OpenCV: 4.8.x
NumPy: 1.24.x
PyTorch: 2.x.x
CUDA: True  ← IMPORTANT!
Ultralytics: 8.x.x
SciPy: 1.10.x
FastAPI: 0.100.x
Uvicorn: 0.23.x
All packages OK!
```

---

## 📝 Summary

✅ **Created 2 platform-specific requirements files**  
✅ **Removed unused packages** (Pillow)  
✅ **Added version constraints** for stability  
✅ **Clear installation instructions** for both platforms  
✅ **Comprehensive troubleshooting guide**  

**Your V2 project now has clean, production-ready requirements!**

You can either:
1. Copy your working V1 venv (fastest)
2. Fresh install with new requirements (cleanest)

Both will give you GPU support for 15+ FPS performance!
