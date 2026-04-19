#!/usr/bin/env python3
"""
Container QR Tracking Application Entry Point.

This is the main entry point for the container tracking pipeline that monitors
containers at the sale point (صالة) using QR codes.

The application:
1. Connects to the container camera (192.168.2.118 by default)
2. Detects QR codes (values 1-5) on containers
3. Tracks container direction (positive = filled leaving, negative = empty returning)
4. Saves snapshots for all container events
5. Records events to the database
6. Publishes state for health monitoring

Usage:
    Edit the configuration variables near the top of this file, then run:
        python3 container_main.py

Environment Variables:
    CONTAINER_VIDEO_PATH: Optional fallback video path when VIDEO_PATH is empty
    USE_CONTAINER_RTSP_SUBSTREAM: 'true' to use substream instead of main stream

Supervisor:
    This process is managed by supervisor as 'breadcount-container-main'
"""

import sys
import os

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# This process is isolated, but many imported modules initialize logging at
# import time. Set container_main-specific logging env before any project
# imports so the first AppLogging setup in this process uses these values.
CONTAINER_LOG_BASENAME = "container_main"
CONTAINER_LOG_MAX_BYTES = 20 * 1024 * 1024
CONTAINER_LOG_BACKUP_COUNT = 10
CONTAINER_ERROR_LOG_MAX_BYTES = 5 * 1024 * 1024
CONTAINER_ERROR_LOG_BACKUP_COUNT = 3
CONTAINER_LOG_RETENTION_DAYS = 7

_CONTAINER_LOG_ENV_DEFAULTS = (
    ("APP_LOG_BASENAME", CONTAINER_LOG_BASENAME),
    ("APP_LOG_MAX_BYTES", str(CONTAINER_LOG_MAX_BYTES)),
    ("APP_LOG_BACKUP_COUNT", str(CONTAINER_LOG_BACKUP_COUNT)),
    ("APP_ERROR_LOG_MAX_BYTES", str(CONTAINER_ERROR_LOG_MAX_BYTES)),
    ("APP_ERROR_LOG_BACKUP_COUNT", str(CONTAINER_ERROR_LOG_BACKUP_COUNT)),
    ("APP_LOG_RETENTION_DAYS", str(CONTAINER_LOG_RETENTION_DAYS)),
)
for env_name, env_value in _CONTAINER_LOG_ENV_DEFAULTS:
    os.environ.setdefault(env_name, env_value)

from src.container.ContainerCounterApp import ContainerCounterApp, ContainerConfig
from src.logging.Database import DatabaseManager
from src.utils.AppLogging import logger
from src.utils.platform import PLATFORM_NAME, IS_RDK
from src.config.paths import DB_PATH
import src.constants as constants

# =============================================================================
# Configuration — edit these variables instead of using command-line args
# =============================================================================
if IS_RDK:
    logger.info("Running on RDK platform - using RTSP stream by default")
    DEVELOPMENT = False             # Force development mode (video file instead of ROS2)
    ENABLE_DISPLAY = False           # Force enable cv2 display window
    VIDEO_PATH = ""                # Path to video file (development only)
else:
    logger.info("Not running on RDK platform - using video file by default")
    DEVELOPMENT = True              # Force development mode (video file instead of ROS2)
    ENABLE_DISPLAY = True            # Force enable cv2 display window
    VIDEO_PATH = os.path.expanduser('~/Downloads/cam_qr.mp4') # Path to video file (development only)
MAX_FRAMES = None               # Max frames to process (None = infinite)
DATABASE_PATH = str(DB_PATH)    # Path to database file


def main():
    """Main entry point for container tracking application."""
    logger.info("=" * 60)
    logger.info("Container QR Tracking Application")
    logger.info(f"Platform: {PLATFORM_NAME}")
    logger.info(f"RDK Mode: {IS_RDK}")
    logger.info("=" * 60)
    
    # Initialize database and check/set development mode
    db = None
    config = None
    try:
        db = DatabaseManager(DATABASE_PATH)
        
        db.set_config(constants.is_development_key, '1' if DEVELOPMENT else '0')
        db.set_config(constants.container_enable_display_key, '1' if ENABLE_DISPLAY else '0')
        logger.info(f"Development mode set from container_main: {DEVELOPMENT}")
        logger.info(f"Display mode set from container_main: {ENABLE_DISPLAY}")
        
        is_development = db.get_config(constants.is_development_key, '0') == '1'
        
        # Log configuration
        logger.info(f"Development Mode: {is_development}")
        
        if is_development:
            video_source = VIDEO_PATH or os.getenv('CONTAINER_VIDEO_PATH', '')
            logger.info(f"Video Source: {video_source or '(none - empty frames)'}")
        else:
            container_host = db.get_config(constants.container_rtsp_host, '192.168.2.118')
            logger.info(f"Container Camera Host (RTSP): {container_host}")
        
        # Build configuration
        config = ContainerConfig.from_database(db)
        config.db_path = DATABASE_PATH
        
        if VIDEO_PATH:
            config.video_source = VIDEO_PATH
        
        # Log tracking config
        logger.info(f"Display Enabled: {config.enable_display}")
        logger.info(f"Exit Zone Ratio: {config.exit_zone_ratio}")
        logger.info(f"Lost Timeout: {config.lost_timeout}s")
        logger.info(f"Snapshot Buffer: {config.pre_event_seconds}s pre + {config.post_event_seconds}s post")
        logger.info(f"Snapshot Directory: {config.snapshot_dir}")
        logger.info("=" * 60)
    finally:
        if db is not None:
            db.close()
    
    # Create and run application
    app = ContainerCounterApp(config)
    
    try:
        app.run(max_frames=MAX_FRAMES)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.exception(f"Application error: {e}")
        return 1
    
    logger.info("Container tracking application exited")
    return 0


if __name__ == '__main__':
    sys.exit(main())
