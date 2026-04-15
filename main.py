#!/usr/bin/env python3
"""
Conveyor Bread Bag Counter System - v2
Main entry point for the application.

Usage:
    python main.py

Configuration is read from database (data/db/bag_events.db).
To configure, update the config table in the database.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config.settings import AppConfig
from src.config.tracking_config import TrackingConfig
from src.app.ConveyorCounterApp import ConveyorCounterApp
from src.utils.AppLogging import logger
from src.utils.platform import PLATFORM_NAME
import src.config.paths as _paths


def main():
    """Main entry point."""
    app_config = AppConfig()
    tracking_config = TrackingConfig()

    W = 64
    logger.info("=" * W)
    logger.info(f"  Conveyor Bread Bag Counter  |  v{app_config.APP_VERSION}")
    logger.info(f"  Platform : {PLATFORM_NAME}")
    logger.info("=" * W)
    logger.info(f"  Paths")
    logger.info(f"    Data dir       : {_paths.DATA_DIR}")
    logger.info(f"    Database       : {_paths.DB_PATH}")
    logger.info(f"    Logs           : {_paths.LOG_DIR}")
    logger.info(f"    Recordings     : {_paths.RECORDING_DIR}")
    logger.info(f"    Pipeline state : {_paths.PIPELINE_STATE_FILE}")
    logger.info(f"    IPC status     : {_paths.TMP_STATUS_DIR}/")
    logger.info("-" * W)
    logger.info(f"  Models")
    logger.info(f"    Detection      : {app_config.detection_model}")
    logger.info(f"    Classification : {app_config.classification_model}")
    logger.info("=" * W)
    logger.info("  Display and other runtime settings: read from database config")

    # Create and run application (config read from DB)
    app = ConveyorCounterApp(
        app_config=app_config,
        tracking_config=tracking_config
    )
    
    try:
        app.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
