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
sys.path.append("/home/sunrise/ConvuyerBreadCounting")

from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable
from launch_ros.actions import Node

from src.logging.Database import DatabaseManager
import src.constants as constants
from src.utils.AppLogging import logger


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

    # Get RTSP configuration from database
    db_path = os.getenv("DB_PATH", "/home/sunrise/ConvuyerBreadCounting/data/db/bag_events.db")
    db = DatabaseManager(db_path)

    rtsp_username = db.get_config(constants.rtsp_username, 'admin')
    rtsp_password = db.get_config(constants.rtsp_password, '')
    rtsp_host = db.get_config(constants.rtsp_host, '192.168.2.108')
    rtsp_port = db.get_config(constants.rtsp_port, '554')

    # Build RTSP URL
    # Use subtype=0 for main stream (higher quality)
    # Use subtype=1 for sub stream (lower bandwidth)
    use_substream = os.getenv('USE_RTSP_SUBSTREAM', 'false').lower() == 'true'
    subtype = 1 if use_substream else 0

    rtsp_url = (
        f"rtsp://{rtsp_username}:{rtsp_password}@{rtsp_host}:{rtsp_port}"
        f"/cam/realmonitor?channel=1&subtype={subtype}"
    )

    logger.info(f"[Ros2PipelineLauncher] RTSP URL: rtsp://{rtsp_username}:***@{rtsp_host}:{rtsp_port}/...")

    # RTSP Client Node
    # Receives RTSP stream and publishes raw H.264 NAL units
    rtsp_node = Node(
        package='hobot_rtsp_client',
        executable='hobot_rtsp_client',
        output='screen',
        parameters=[
            {
                'rtsp_url_num': 1,
                'rtsp_url_0': rtsp_url,
                # Transport configuration
                'rtsp_transport': 'tcp',  # TCP for reliability (avoids UDP packet loss)
                'rtsp_subtype': subtype,
                # Buffer configuration (increase for high-bitrate streams)
                'rtp_reassembly_buffer_bytes': 1048576,  # 1MB buffer
            }
        ]
    )

    # Hardware Decoder Node
    # Decodes H.264 to NV12 using RDK hardware decoder
    # NOTE: Subscribe to /spool_image_ch_0 for spool-based architecture
    # The spool processor reads from disk and publishes to this topic
    hw_decode_node = Node(
        package='hobot_codec',
        executable='hobot_codec_republish',
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

