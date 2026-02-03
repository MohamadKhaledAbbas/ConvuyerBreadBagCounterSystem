"""
ROS2 Pipeline Launcher for ConveyorBreadBagCounterSystem.

This module provides the ROS2 launch description for RDK platform deployment.
It configures:
- RTSP client for video stream ingestion
- Hardware H.264 decoder (hobot_codec)
- Environment variables for ROS2/FastDDS

Usage:
    ros2 launch src/ros2/Ros2PipelineLauncher.py

Production Notes:
- Uses shared memory transport for zero-copy frame passing
- Configurable RTSP credentials from database
- Support for both main stream and substream
"""

import sys
import os

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.AppLogging import logger
from src.utils.platform import IS_RDK

# Only import ROS2 dependencies on RDK platform
if IS_RDK:
    from launch import LaunchDescription # type: ignore
    from launch.actions import SetEnvironmentVariable # type: ignore
    from launch_ros.actions import Node # type: ignore

    from src.logging.Database import DatabaseManager
    from src import constants


def get_rtsp_config() -> dict:
    """
    Get RTSP configuration from database.

    Returns:
        Dictionary with RTSP connection parameters
    """
    try:
        db_path = os.getenv("DB_PATH", "/home/sunrise/ConveyerCounting/data/db/bag_events.db")
        db = DatabaseManager(db_path)

        config = {
            'username': db.get_config(constants.rtsp_username, 'admin'),
            'password': db.get_config(constants.rtsp_password, ''),
            'host': db.get_config(constants.rtsp_host, '192.168.1.100'),
            'port': db.get_config(constants.rtsp_port, '554'),
        }

        db.close()
        return config

    except Exception as e:
        logger.warning(f"[Ros2PipelineLauncher] Could not read RTSP config from DB: {e}")
        # Return defaults
        return {
            'username': os.getenv('RTSP_USERNAME', 'admin'),
            'password': os.getenv('RTSP_PASSWORD', ''),
            'host': os.getenv('RTSP_HOST', '192.168.1.100'),
            'port': os.getenv('RTSP_PORT', '554'),
        }


def build_rtsp_url(config: dict, subtype: int = 0) -> str:
    """
    Build RTSP URL from configuration.

    Args:
        config: RTSP configuration dictionary
        subtype: Stream subtype (0=main, 1=sub)

    Returns:
        Complete RTSP URL
    """
    return (
        f"rtsp://{config['username']}:{config['password']}@"
        f"{config['host']}:{config['port']}"
        f"/cam/realmonitor?channel=1&subtype={subtype}"
    )


def generate_launch_description():
    """
    Generate ROS2 launch description for the conveyor counting pipeline.

    This launches:
    1. hobot_rtsp_client - Receives RTSP stream, publishes H.264 NAL units
    2. hobot_codec - Decodes H.264 to NV12 for processing

    Returns:
        LaunchDescription with configured nodes
    """
    logger.info("[Ros2PipelineLauncher] Generating launch description...")
    logger.debug(f"[Ros2PipelineLauncher] System paths: {sys.path}")

    # Environment setup for FastDDS shared memory transport
    env_setup = [
        SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp'),
        SetEnvironmentVariable(
            'FASTRTPS_DEFAULT_PROFILES_FILE',
            '/opt/tros/humble/lib/hobot_shm/config/shm_fastdds.xml'
        ),
        SetEnvironmentVariable('RMW_FASTRTPS_USE_QOS_FROM_XML', '1'),
        SetEnvironmentVariable('ROS_DISABLE_LOANED_MESSAGES', '0'),
        SetEnvironmentVariable('HOME', '/home/sunrise'),
    ]

    # Get RTSP configuration
    rtsp_config = get_rtsp_config()

    # Build RTSP URL (main stream by default)
    # Use subtype=0 for main stream (higher quality)
    # Use subtype=1 for sub stream (lower bandwidth)
    use_substream = os.getenv('USE_RTSP_SUBSTREAM', 'false').lower() == 'true'
    rtsp_url = build_rtsp_url(rtsp_config, subtype=1 if use_substream else 0)

    logger.info(f"[Ros2PipelineLauncher] RTSP URL: rtsp://{rtsp_config['username']}:***@{rtsp_config['host']}:{rtsp_config['port']}/...")

    # RTSP Client Node
    # Receives RTSP stream and publishes raw H.264 NAL units
    rtsp_node = Node(
        package='hobot_rtsp_client',
        executable='hobot_rtsp_client',
        name='rtsp_client',
        output='screen',
        parameters=[
            {
                'rtsp_url_num': 1,
                'rtsp_url_0': rtsp_url,
                # Transport configuration
                'rtsp_transport': 'tcp',  # TCP for reliability (avoids UDP packet loss)
                'rtsp_subtype': 1 if use_substream else 0,
                # Buffer configuration (increase for high-bitrate streams)
                'rtp_reassembly_buffer_bytes': 1048576,  # 1MB buffer
            }
        ]
    )

    # Hardware Decoder Node
    # Decodes H.264 to NV12 using RDK hardware decoder
    hw_decode_node = Node(
        package='hobot_codec',
        executable='hobot_codec_republish',
        name='hw_decoder',
        output='screen',
        parameters=[
            {
                'in_format': 'h264',
                'out_mode': 'ros',
                'out_format': 'nv12',
                'sub_topic': '/spool_image_ch_0',  # Input from spool processor
                'pub_topic': '/nv12_images',       # Output for detection
                'dump_output': False,
            }
        ],
        arguments=['--ros-args', '--log-level', 'ERROR']
    )

    logger.info("[Ros2PipelineLauncher] Launch description generated")

    return LaunchDescription(env_setup + [rtsp_node, hw_decode_node])


# For non-RDK platforms, provide a stub
if not IS_RDK:
    def generate_launch_description():
        """Stub for non-RDK platforms."""
        logger.error("[Ros2PipelineLauncher] This module requires RDK platform")
        raise RuntimeError("Ros2PipelineLauncher requires RDK platform with ROS2")
