"""
ROS2 Pipeline Launcher for Container Camera (Sale Point / صالة).

This module provides the ROS2 launch description for the container tracking pipeline.
It configures:
- RTSP client for container camera video stream (192.168.2.118)
- Hardware H.264 decoder (hobot_codec) for NV12 output
- Environment variables for ROS2/FastDDS

Topic naming convention (with '_container' suffix):
- /rtsp_image_container: H.264 NAL units from RTSP client
- /nv12_images_container: Decoded NV12 frames for QR processing

Usage:
    ros2 launch src/container/ros2/ContainerRos2Launcher.py

Production Notes:
- Uses shared memory transport for zero-copy frame passing
- Configurable RTSP credentials from database (shared with bread camera)
- Default container camera IP: 192.168.2.118
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


# Container camera topic names
CONTAINER_RTSP_TOPIC = '/rtsp_image_container'
CONTAINER_NV12_TOPIC = '/nv12_images_container'

# Default container camera IP
DEFAULT_CONTAINER_HOST = '192.168.2.118'


def generate_launch_description():
    """
    Generate ROS2 launch description for the container tracking pipeline.

    This launches:
    1. hobot_rtsp_client - Receives RTSP stream from container camera,
                          publishes H.264 NAL units to /rtsp_image_container
    2. hobot_codec - Decodes H.264 to NV12 for QR code processing,
                    subscribes to /rtsp_image_container,
                    publishes to /nv12_images_container

    Returns:
        LaunchDescription with configured nodes
    """
    logger.info("[ContainerRos2Launcher] Generating launch description...")
    logger.debug(f"[ContainerRos2Launcher] System paths: {sys.path}")

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
    from src.config.paths import DB_PATH
    db_path = DB_PATH
    db = DatabaseManager(db_path)

    # Container camera uses its own host but shares credentials by default
    # Fallback to bread camera credentials if container-specific ones not set
    container_host = db.get_config(
        constants.container_rtsp_host,
        DEFAULT_CONTAINER_HOST
    )
    
    # Use container-specific credentials if set, otherwise fallback to shared credentials
    container_username = db.get_config(
        constants.container_rtsp_username,
        db.get_config(constants.rtsp_username, 'admin')
    )
    container_password = db.get_config(
        constants.container_rtsp_password,
        db.get_config(constants.rtsp_password, '')
    )
    container_port = db.get_config(
        constants.container_rtsp_port,
        db.get_config(constants.rtsp_port, '554')
    )

    # Build RTSP URL
    # Use subtype=0 for main stream (higher quality for QR detection)
    use_substream = os.getenv('USE_CONTAINER_RTSP_SUBSTREAM', 'false').lower() == 'true'
    subtype = 1 if use_substream else 0

    rtsp_url = (
        f"rtsp://{container_username}:{container_password}@{container_host}:{container_port}"
        f"/cam/realmonitor?channel=1&subtype={subtype}"
    )

    logger.info(
        f"[ContainerRos2Launcher] Container RTSP URL: "
        f"rtsp://{container_username}:***@{container_host}:{container_port}/..."
    )
    logger.info(
        f"[ContainerRos2Launcher] Topics: {CONTAINER_RTSP_TOPIC} -> {CONTAINER_NV12_TOPIC}"
    )

    # RTSP Client Node for Container Camera
    # Receives RTSP stream and publishes raw H.264 NAL units
    # Note: hobot_rtsp_client uses rtsp_url_0 naming convention
    container_rtsp_node = Node(
        package='hobot_rtsp_client',
        executable='hobot_rtsp_client',
        name='container_rtsp_client',  # Unique node name
        output='screen',
        parameters=[
            {
                'rtsp_url_num': 1,
                'rtsp_url_0': rtsp_url,
                # Transport configuration
                'rtsp_transport': 'tcp',  # TCP for reliability
                'rtsp_subtype': subtype,
                # Buffer configuration
                'rtp_reassembly_buffer_bytes': 4194304,  # 4MB buffer
                # Output topic override (requires hobot_rtsp_client version with this param)
                # If not supported, we may need to remap
            }
        ],
        remappings=[
            # Remap default output topic to container-specific topic
            ('/rtsp_image_ch_0', CONTAINER_RTSP_TOPIC),
        ],
    )

    # Hardware Decoder Node for Container Camera
    # Decodes H.264 to NV12 using RDK hardware decoder
    container_codec_node = Node(
        package='hobot_codec',
        executable='hobot_codec_republish',
        name='container_codec',  # Unique node name
        output='screen',
        respawn=True,
        respawn_delay=2.0,
        parameters=[
            {
                'in_format': 'h264',
                'out_mode': 'ros',
                'out_format': 'nv12',
                'sub_topic': CONTAINER_RTSP_TOPIC,   # From container RTSP client
                'pub_topic': CONTAINER_NV12_TOPIC,   # Output for QR detection
                'dump_output': False,
                'input_message_qos_depth': 200,
            }
        ],
        arguments=['--ros-args', '--log-level', 'ERROR']
    )

    logger.info("[ContainerRos2Launcher] Launch description generated")

    return LaunchDescription(env_setup + [container_rtsp_node, container_codec_node])
