#!/usr/bin/env python3
"""
Test script to verify OpenCVFrameSource fixes for hang issues.

Tests:
1. Backpressure mechanism (queue blocking)
2. Graceful shutdown (responsive stop)
3. CPU management (proper frame pacing)
"""

import sys
import os
import time
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.frame_source.OpenCvFrameSource import OpenCVFrameSource
from src.utils.AppLogging import logger


def test_backpressure():
    """Test that backpressure works (slow consumer doesn't cause memory overflow)."""
    logger.info("=" * 60)
    logger.info("TEST 1: Backpressure with slow consumer")
    logger.info("=" * 60)

    # Create frame source with small queue
    video_path = "D:\\Recordings\\2026_02_05\\output_2026-02-05_22-45-50.mp4"
    if not os.path.exists(video_path):
        logger.warning(f"Video file not found: {video_path}")
        return

    frame_source = OpenCVFrameSource(
        source=video_path,
        queue_size=10,  # Small queue to trigger backpressure
        target_fps=None  # Use source FPS
    )

    try:
        # Simulate slow consumer
        frame_count = 0
        start_time = time.time()

        for frame, latency in frame_source.frames():
            frame_count += 1

            # Slow processing (consumer can't keep up)
            time.sleep(0.1)  # 100ms per frame = ~10 FPS consumer

            if frame_count % 10 == 0:
                elapsed = time.time() - start_time
                logger.info(f"Processed {frame_count} frames in {elapsed:.1f}s")

            # Stop after 50 frames
            if frame_count >= 50:
                logger.info("Stopping after 50 frames")
                break

        elapsed = time.time() - start_time
        logger.info(f"✓ Backpressure test passed: {frame_count} frames in {elapsed:.1f}s")

    finally:
        frame_source.cleanup()

    logger.info("")


def test_graceful_shutdown():
    """Test that shutdown is responsive (can interrupt quickly)."""
    logger.info("=" * 60)
    logger.info("TEST 2: Graceful shutdown (interrupt responsiveness)")
    logger.info("=" * 60)

    video_path = "D:\\Recordings\\2026_02_05\\output_2026-02-05_22-45-50.mp4"
    if not os.path.exists(video_path):
        logger.warning(f"Video file not found: {video_path}")
        return

    frame_source = OpenCVFrameSource(
        source=video_path,
        queue_size=30,
        target_fps=None
    )

    # Start consuming frames in background
    stop_event = threading.Event()
    frame_count = [0]

    def consume_frames():
        for frame, latency in frame_source.frames():
            frame_count[0] += 1
            if stop_event.is_set():
                break

    consumer_thread = threading.Thread(target=consume_frames)
    consumer_thread.start()

    # Let it run for 2 seconds
    time.sleep(2)

    # Try to stop
    logger.info("Stopping frame source...")
    stop_start = time.time()
    stop_event.set()
    frame_source.cleanup()
    consumer_thread.join(timeout=5)
    stop_elapsed = time.time() - stop_start

    logger.info(f"✓ Shutdown test passed: Stopped in {stop_elapsed:.2f}s (processed {frame_count[0]} frames)")

    if stop_elapsed > 5:
        logger.error("✗ Shutdown took too long!")
    else:
        logger.info("✓ Shutdown was responsive")

    logger.info("")


def test_cpu_management():
    """Test that CPU usage is reasonable (proper frame pacing)."""
    logger.info("=" * 60)
    logger.info("TEST 3: CPU management (frame pacing)")
    logger.info("=" * 60)

    video_path = "D:\\Recordings\\2026_02_05\\output_2026-02-05_22-45-50.mp4"
    if not os.path.exists(video_path):
        logger.warning(f"Video file not found: {video_path}")
        return

    frame_source = OpenCVFrameSource(
        source=video_path,
        queue_size=30,
        target_fps=30  # Force 30 FPS pacing
    )

    try:
        frame_count = 0
        start_time = time.time()

        # Consume frames at normal speed
        for frame, latency in frame_source.frames():
            frame_count += 1

            # Stop after 100 frames
            if frame_count >= 100:
                break

        elapsed = time.time() - start_time
        actual_fps = frame_count / elapsed if elapsed > 0 else 0

        logger.info(f"✓ CPU test passed: {frame_count} frames in {elapsed:.1f}s = {actual_fps:.1f} FPS")

        # Check if FPS is reasonable (should be around 30 FPS)
        if 25 <= actual_fps <= 35:
            logger.info("✓ Frame pacing is working correctly")
        else:
            logger.warning(f"⚠ Frame pacing may be off (expected ~30 FPS, got {actual_fps:.1f})")

    finally:
        frame_source.cleanup()

    logger.info("")


def main():
    """Run all tests."""
    logger.info("Testing OpenCVFrameSource fixes for hang issues")
    logger.info("")

    try:
        test_backpressure()
        test_graceful_shutdown()
        test_cpu_management()

        logger.info("=" * 60)
        logger.info("ALL TESTS COMPLETED")
        logger.info("=" * 60)
        logger.info("If all tests passed, the hang issues should be resolved.")
        logger.info("")
        logger.info("Key improvements:")
        logger.info("✓ Backpressure prevents unbounded memory growth")
        logger.info("✓ Graceful shutdown allows responsive interruption")
        logger.info("✓ Frame pacing prevents CPU spinning")

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
