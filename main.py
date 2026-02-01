#!/usr/bin/env python3
"""
Conveyor Bread Bag Counter System - v2
Main entry point for the application.

Usage:
    python main.py [options]

Examples:
    # Run with video file
    python main.py --source video.mp4

    # Run with webcam
    python main.py --source 0

    # Run with RTSP stream
    python main.py --source "rtsp://user:pass@ip:port/stream"

    # Testing mode (process all frames, no drops)
    python main.py --source video.mp4 --testing

    # Headless mode (no display)
    python main.py --source video.mp4 --no-display
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config.settings import AppConfig
from src.config.tracking_config import TrackingConfig
from src.app.ConveyorCounterApp import ConveyorCounterApp
from src.utils.AppLogging import logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Conveyor Bread Bag Counter System v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --source video.mp4
  python main.py --source 0 --no-recording
  python main.py --source rtsp://ip:port/stream --headless
        """
    )
    
    # Input source
    parser.add_argument(
        '--source', '-s',
        default='0',
        help='Video source: file path, camera index (0), or RTSP URL'
    )
    
    # Mode options
    parser.add_argument(
        '--testing', '-t',
        action='store_true',
        help='Testing mode: process all frames (no frame drops)'
    )
    
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Disable visualization window (headless mode)'
    )
    
    parser.add_argument(
        '--no-recording',
        action='store_true',
        help='Disable video spool recording'
    )
    
    # Model paths (override config)
    parser.add_argument(
        '--detector-model',
        type=str,
        help='Path to detection model file'
    )
    
    parser.add_argument(
        '--classifier-model',
        type=str,
        help='Path to classification model file'
    )
    
    # Thresholds
    parser.add_argument(
        '--detection-conf',
        type=float,
        default=0.5,
        help='Detection confidence threshold (default: 0.5)'
    )
    
    parser.add_argument(
        '--classification-conf',
        type=float,
        default=0.5,
        help='Classification confidence threshold (default: 0.5)'
    )
    
    # Output
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./output',
        help='Output directory for recordings and logs'
    )
    
    parser.add_argument(
        '--database',
        type=str,
        default='./bread_counter.db',
        help='Path to SQLite database file'
    )
    
    # Limits
    parser.add_argument(
        '--max-frames',
        type=int,
        default=None,
        help='Maximum frames to process (for testing)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("Conveyor Bread Bag Counter System v2")
    logger.info("=" * 60)
    
    # Create configuration
    app_config = AppConfig()
    tracking_config = TrackingConfig()
    
    # Apply command line overrides
    if args.source.isdigit():
        app_config.video_source = int(args.source)
    else:
        app_config.video_source = args.source
    
    if args.detector_model:
        app_config.detection_model_path = args.detector_model
    
    if args.classifier_model:
        app_config.classifier_model_path = args.classifier_model
    
    app_config.database_path = args.database
    
    tracking_config.detection_confidence = args.detection_conf
    tracking_config.classification_confidence = args.classification_conf
    tracking_config.spool_dir = args.output_dir
    
    # Log configuration
    logger.info(f"Source: {app_config.video_source}")
    logger.info(f"Testing mode: {args.testing}")
    logger.info(f"Display: {not args.no_display}")
    logger.info(f"Recording: {not args.no_recording}")
    logger.info(f"Detection model: {app_config.detection_model_path}")
    logger.info(f"Classifier model: {app_config.classifier_model_path}")
    logger.info(f"Output dir: {tracking_config.spool_dir}")
    
    # Create and run application
    app = ConveyorCounterApp(
        app_config=app_config,
        tracking_config=tracking_config,
        enable_display=not args.no_display,
        enable_recording=not args.no_recording,
        testing_mode=args.testing
    )
    
    try:
        app.run(max_frames=args.max_frames)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
