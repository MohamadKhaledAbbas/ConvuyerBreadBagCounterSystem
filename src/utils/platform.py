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

# Detect Linux platform (includes RDK — use alongside IS_RDK for precise checks)
IS_LINUX = sys.platform.startswith('linux')

# Platform description for logging
if IS_RDK:
    PLATFORM_NAME = "RDK"
elif IS_WINDOWS:
    PLATFORM_NAME = "Windows"
elif IS_LINUX:
    PLATFORM_NAME = "Linux"
else:
    PLATFORM_NAME = sys.platform


def is_rdk_platform() -> bool:
    """Check if running on RDK platform."""
    return IS_RDK


def is_linux() -> bool:
    """Check if running on Linux (including RDK)."""
    return IS_LINUX


def is_windows() -> bool:
    """Check if running on Windows."""
    return IS_WINDOWS

