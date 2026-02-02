"""
ROS2 modules for ConveyorBreadBagCounterSystem.

Provides:
- Ros2PipelineLauncher: Launch description for RDK platform
"""

from src.utils.platform import IS_RDK

if IS_RDK:
    from src.ros2.Ros2PipelineLauncher import generate_launch_description
    __all__ = ['generate_launch_description']
else:
    __all__ = []
