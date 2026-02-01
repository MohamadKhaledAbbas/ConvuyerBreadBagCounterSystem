"""
Platform detection for cross-platform compatibility.

Detects whether running on RDK (Horizon Robotics) or Windows/Linux.
"""

import sys

# Detect RDK platform by checking for BPU libraries
IS_RDK = False
try:
    import hobot_dnn
    IS_RDK = True
except ImportError:
    try:
        import hobot_dnn_rdkx5
        IS_RDK = True
    except ImportError:
        IS_RDK = False

# Detect Windows platform
IS_WINDOWS = sys.platform == 'win32'

# Platform description for logging
if IS_RDK:
    PLATFORM_NAME = "RDK"
elif IS_WINDOWS:
    PLATFORM_NAME = "Windows"
else:
    PLATFORM_NAME = "Linux"
