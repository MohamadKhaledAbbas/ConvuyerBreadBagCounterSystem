"""
ROS2 modules for ConveyorBreadBagCounterSystem.

Provides:
- Ros2PipelineLauncher: Launch description for RDK platform
- CodecHealthMonitor: Auto-recovery for VPU decoder stalls
- RecoveryStage: Staged recovery escalation levels
- perform_startup_cleanup: Clean stale artifacts on boot
"""

from src.utils.platform import IS_RDK

if IS_RDK:
    from src.ros2.Ros2PipelineLauncher import generate_launch_description
    from src.ros2.codec_health_monitor import (
        CodecHealthMonitor,
        MonitorConfig,
        RecoveryStage,
        perform_startup_cleanup,
    )
    __all__ = [
        'generate_launch_description',
        'CodecHealthMonitor',
        'MonitorConfig',
        'RecoveryStage',
        'perform_startup_cleanup',
    ]
else:
    __all__ = []
