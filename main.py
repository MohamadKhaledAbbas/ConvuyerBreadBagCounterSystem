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


def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("Conveyor Bread Bag Counter System v2")
    logger.info("=" * 60)
    
    # Create configuration
    app_config = AppConfig()
    tracking_config = TrackingConfig()
    
    # Log configuration
    logger.info(f"Detection model: {app_config.detection_model}")
    logger.info(f"Classifier model: {app_config.classification_model}")
    logger.info(f"Output dir: {tracking_config.spool_dir}")
    logger.info(f"Database: {app_config.db_path}")
    logger.info("Display and other settings: Read from database config")

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
