#!/usr/bin/env python3
"""
RTSP H.264 Stream Recorder using FFmpeg for Production RDK.

This is a SEPARATE process from the main counting pipeline.
Records H.264 stream directly from RTSP source using FFmpeg (no re-encoding).

Architecture:
=============
RTSP Source → FFmpeg (stream copy) → MP4/MKV File → Disk

Note: This is different from spool_recorder which:
- Records in 5-second segments using SegmentWriter
- Designed for frame-by-frame processing
- Used by SpoolProcessor for playback

This recorder:
- Records continuous stream to standard video files
- No segmentation (or custom rotation logic)
- Direct stream copy (no transcoding)
- Much simpler and more efficient

Usage:
======
    # Run with database config
    python rtsp_h264_recorder.py

    # Or override with command line
    python rtsp_h264_recorder.py --url rtsp://camera_ip:554/stream --output video.mp4

Configuration:
==============
Reads from database config table:
- enable_recording: '1' to enable, '0' to disable
- rtsp_host: Camera IP address
- rtsp_port: RTSP port (default 554)
- rtsp_username: RTSP username
- rtsp_password: RTSP password

Why FFmpeg:
===========
1. Direct H.264 stream copy (no CPU overhead)
2. No decode/encode cycle
3. Robust RTSP handling
4. Automatic reconnection
5. Hardware acceleration support
"""

import argparse
import signal
import sys
import os
import time
import subprocess
import threading
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.AppLogging import logger
from src.logging.Database import DatabaseManager
from src.constants import enable_recording_key, rtsp_host, rtsp_port, rtsp_username, rtsp_password


class RtspH264Recorder:
    """
    Records H.264 stream from RTSP source using FFmpeg.

    Uses FFmpeg subprocess to copy H.264 stream directly to file
    without re-encoding. Supports file rotation and retention.
    """

    def __init__(
        self,
        rtsp_url: str,
        output_file: str = None,
        output_dir: str = None,
        rotate_hours: float = None,
        retention_hours: float = None,
        container: str = "mp4"
    ):
        """
        Initialize RTSP H.264 recorder.

        Args:
            rtsp_url: RTSP stream URL (e.g., rtsp://camera_ip:554/stream)
            output_file: Single output file (if no rotation)
            output_dir: Directory for rotated files
            rotate_hours: Rotate file every N hours (None = no rotation)
            retention_hours: Delete files older than N hours (None = keep all)
            container: Container format ('mp4', 'mkv', 'avi')
        """
        self.rtsp_url = rtsp_url
        self.output_file = output_file
        self.output_dir = output_dir
        self.rotate_hours = rotate_hours
        self.retention_hours = retention_hours
        self.container = container

        # State
        self.running = False
        self.ffmpeg_process = None
        self.current_file = None
        self.start_time = None
        self.last_rotation = None

        # Cleanup thread
        self.cleanup_thread = None

        # Validate settings
        if output_file and output_dir:
            raise ValueError("Specify either output_file or output_dir, not both")

        if not output_file and not output_dir:
            raise ValueError("Must specify either output_file or output_dir")

        if rotate_hours and not output_dir:
            raise ValueError("Rotation requires output_dir")

        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        logger.info(
            f"[RtspH264Recorder] Initialized: url={rtsp_url}, "
            f"output={output_file or output_dir}, rotate={rotate_hours}h"
        )

    def _get_output_filename(self) -> str:
        """Generate output filename with timestamp."""
        if self.output_file:
            return self.output_file

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp}.{self.container}"
        return str(Path(self.output_dir) / filename)

    def _start_ffmpeg(self, output_file: str):
        """
        Start FFmpeg subprocess to record RTSP stream.

        FFmpeg command:
        - Copy H.264 stream (no re-encoding)
        - Handle RTSP connection
        - Write to file
        """
        cmd = [
            'ffmpeg',
            '-rtsp_transport', 'tcp',  # Use TCP for RTSP (more reliable)
            '-i', self.rtsp_url,        # Input RTSP stream
            '-c:v', 'copy',             # Copy video stream (no re-encode)
            '-c:a', 'copy',             # Copy audio stream (if present)
            '-movflags', '+faststart',  # Optimize for streaming (MP4)
            '-f', self.container,       # Output format
            '-y',                       # Overwrite output file
            output_file
        ]

        logger.info(f"[RtspH264Recorder] Starting FFmpeg: {output_file}")
        logger.debug(f"[RtspH264Recorder] Command: {' '.join(cmd)}")

        try:
            self.ffmpeg_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1
            )
            self.current_file = output_file
            logger.info(f"[RtspH264Recorder] FFmpeg started (PID: {self.ffmpeg_process.pid})")
            return True

        except FileNotFoundError:
            logger.error("[RtspH264Recorder] FFmpeg not found! Install with: apt-get install ffmpeg")
            return False
        except Exception as e:
            logger.error(f"[RtspH264Recorder] Failed to start FFmpeg: {e}")
            return False

    def _stop_ffmpeg(self):
        """Stop FFmpeg subprocess gracefully."""
        if self.ffmpeg_process is None:
            return

        logger.info("[RtspH264Recorder] Stopping FFmpeg...")

        try:
            # Send SIGTERM for graceful shutdown
            self.ffmpeg_process.terminate()

            # Wait up to 10 seconds
            try:
                self.ffmpeg_process.wait(timeout=10)
                logger.info("[RtspH264Recorder] FFmpeg stopped gracefully")
            except subprocess.TimeoutExpired:
                # Force kill if not stopped
                logger.warning("[RtspH264Recorder] FFmpeg not responding, force killing...")
                self.ffmpeg_process.kill()
                self.ffmpeg_process.wait()
                logger.info("[RtspH264Recorder] FFmpeg killed")

        except Exception as e:
            logger.error(f"[RtspH264Recorder] Error stopping FFmpeg: {e}")

        self.ffmpeg_process = None

    def _should_rotate(self) -> bool:
        """Check if file should be rotated."""
        if not self.rotate_hours:
            return False

        if not self.last_rotation:
            return False

        elapsed_hours = (time.time() - self.last_rotation) / 3600
        return elapsed_hours >= self.rotate_hours

    def _rotate_file(self):
        """Rotate to new output file."""
        logger.info("[RtspH264Recorder] Rotating file...")

        # Stop current FFmpeg
        self._stop_ffmpeg()

        # Start new FFmpeg with new filename
        new_file = self._get_output_filename()
        self._start_ffmpeg(new_file)

        self.last_rotation = time.time()

    def _cleanup_old_files(self):
        """Delete files older than retention period."""
        if not self.retention_hours or not self.output_dir:
            return

        try:
            cutoff_time = time.time() - (self.retention_hours * 3600)
            deleted_count = 0

            for file_path in Path(self.output_dir).glob(f"recording_*.{self.container}"):
                if file_path.stat().st_mtime < cutoff_time:
                    logger.info(f"[RtspH264Recorder] Deleting old file: {file_path.name}")
                    file_path.unlink()
                    deleted_count += 1

            if deleted_count > 0:
                logger.info(f"[RtspH264Recorder] Cleaned up {deleted_count} old files")

        except Exception as e:
            logger.error(f"[RtspH264Recorder] Error cleaning up files: {e}")

    def _cleanup_loop(self):
        """Background thread for periodic cleanup."""
        while self.running:
            time.sleep(300)  # Check every 5 minutes
            if self.running:
                self._cleanup_old_files()

    def start(self):
        """Start recording."""
        if self.running:
            logger.warning("[RtspH264Recorder] Already running")
            return False

        logger.info(f"[RtspH264Recorder] Starting recorder...")

        # Get output filename
        output_file = self._get_output_filename()

        # Start FFmpeg
        if not self._start_ffmpeg(output_file):
            return False

        self.running = True
        self.start_time = time.time()
        self.last_rotation = time.time()

        # Start cleanup thread if retention enabled
        if self.retention_hours:
            self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
            self.cleanup_thread.start()
            logger.info(f"[RtspH264Recorder] Retention enabled: {self.retention_hours}h")

        logger.info("[RtspH264Recorder] Recording started")
        return True

    def monitor_loop(self):
        """
        Monitor FFmpeg process and handle rotation.

        Checks:
        - FFmpeg process still running
        - File rotation needed
        - Reconnection if FFmpeg dies
        """
        while self.running:
            try:
                # Check if FFmpeg is still running
                if self.ffmpeg_process and self.ffmpeg_process.poll() is not None:
                    logger.warning(
                        f"[RtspH264Recorder] FFmpeg exited with code {self.ffmpeg_process.returncode}"
                    )

                    # Try to reconnect
                    logger.info("[RtspH264Recorder] Attempting to reconnect...")
                    time.sleep(5)
                    output_file = self._get_output_filename()
                    self._start_ffmpeg(output_file)

                # Check for rotation
                if self._should_rotate():
                    self._rotate_file()

                # Sleep
                time.sleep(10)

            except KeyboardInterrupt:
                logger.info("[RtspH264Recorder] Interrupted by user")
                break
            except Exception as e:
                logger.error(f"[RtspH264Recorder] Error in monitor loop: {e}")
                time.sleep(5)

    def stop(self):
        """Stop recording."""
        logger.info("[RtspH264Recorder] Stopping recorder...")

        self.running = False

        # Stop FFmpeg
        self._stop_ffmpeg()

        # Wait for cleanup thread
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=2)

        # Final cleanup
        if self.retention_hours:
            self._cleanup_old_files()

        # Log statistics
        if self.start_time:
            elapsed = time.time() - self.start_time
            hours = elapsed / 3600
            logger.info("=" * 60)
            logger.info("[RtspH264Recorder] Recording stopped")
            logger.info(f"  Duration: {hours:.2f} hours")
            logger.info(f"  Output: {self.current_file}")
            logger.info("=" * 60)


def parse_args():
    """Parse command line arguments (optional overrides for DB config)."""
    parser = argparse.ArgumentParser(
        description="RTSP H.264 Stream Recorder using FFmpeg (reads config from DB)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use database config
  python rtsp_h264_recorder.py
  
  # Override with command line
  python rtsp_h264_recorder.py --url rtsp://192.168.1.100:554/stream --output video.mp4
        """
    )

    parser.add_argument(
        "--url",
        type=str,
        help="RTSP stream URL (overrides DB config)"
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (for single file recording)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./recordings",
        help="Output directory (default: ./recordings)"
    )

    parser.add_argument(
        "--rotate-hours",
        type=float,
        default=1.0,
        help="Rotate file every N hours (default: 1.0)"
    )

    parser.add_argument(
        "--retention-hours",
        type=float,
        default=48.0,
        help="Delete recordings older than N hours (default: 48.0)"
    )

    parser.add_argument(
        "--container",
        type=str,
        default="mp4",
        choices=["mp4", "mkv", "avi"],
        help="Container format (default: mp4)"
    )

    parser.add_argument(
        "--db-path",
        type=str,
        default="./data/db/bag_events.db",
        help="Path to database (default: ./data/db/bag_events.db)"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("RTSP H.264 Stream Recorder (FFmpeg)")
    logger.info("=" * 60)

    # Initialize database and read config
    db = DatabaseManager(args.db_path)

    # Check if recording is enabled
    recording_enabled = db.get_config(enable_recording_key, default='0')
    if recording_enabled != '1' and not args.url:
        logger.info("Recording is disabled in database config (enable_recording=0)")
        logger.info("To enable, run: ")
        logger.info(f"  sqlite3 {args.db_path} \"UPDATE config SET value='1' WHERE key='{enable_recording_key}';\"")
        db.close()
        return 0

    # Build RTSP URL from database config or command line
    if args.url:
        rtsp_url = args.url
        logger.info("Using RTSP URL from command line")
    else:
        # Read from database
        host = db.get_config(rtsp_host, default='192.168.2.108')
        port = db.get_config(rtsp_port, default='554')
        username = db.get_config(rtsp_username, default='admin')
        password = db.get_config(rtsp_password, default='a1234567')

        rtsp_url = f"rtsp://{username}:{password}@{host}:{port}/stream"
        logger.info("Using RTSP URL from database config")

    db.close()

    # Validate arguments
    if args.output and args.output_dir and not args.output:
        pass  # output_dir is default

    logger.info(f"RTSP URL: {rtsp_url}")
    if args.output:
        logger.info(f"Output file: {args.output}")
    else:
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Rotation: Every {args.rotate_hours} hours")
        logger.info(f"Retention: {args.retention_hours} hours")
    logger.info(f"Container: {args.container}")
    logger.info("=" * 60)

    # Create recorder
    recorder = RtspH264Recorder(
        rtsp_url=rtsp_url,
        output_file=args.output,
        output_dir=args.output_dir if not args.output else None,
        rotate_hours=args.rotate_hours if not args.output else None,
        retention_hours=args.retention_hours if not args.output else None,
        container=args.container
    )

    # Setup signal handlers
    def signal_handler(_sig, _frame):
        logger.info("[Main] Received signal, stopping...")
        recorder.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start recording
    if not recorder.start():
        logger.error("[Main] Failed to start recorder")
        return 1

    # Monitor loop
    try:
        recorder.monitor_loop()
    except Exception as e:
        logger.error(f"[Main] Error: {e}", exc_info=True)
        return 1
    finally:
        recorder.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
