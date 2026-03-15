"""
ROS2 modules for ConveyorBreadBagCounterSystem.

Provides:
- Ros2PipelineLauncher: Launch description for RDK platform
- CodecHealthMonitor: Auto-recovery for VPU decoder stalls
"""

from src.utils.platform import IS_RDK

if IS_RDK:
    from src.ros2.Ros2PipelineLauncher import generate_launch_description
    from src.ros2.codec_health_monitor import CodecHealthMonitor, MonitorConfig
    __all__ = ['generate_launch_description', 'CodecHealthMonitor', 'MonitorConfig']
else:
    __all__ = []
