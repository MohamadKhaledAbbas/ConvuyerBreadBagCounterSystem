"""
Performance benchmark for ConveyorTracker optimizations.

Tests the performance impact of the optimizations:
- Removed debug logging
- Cached config values
- Optimized validation checks
"""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.tracking.ConveyorTracker import ConveyorTracker
from src.config.tracking_config import TrackingConfig
from src.detection.BaseDetection import Detection
import numpy as np


def generate_test_detections(num_detections, frame_height=720, frame_width=1280):
    """Generate random detections moving from bottom to top."""
    detections = []
    for i in range(num_detections):
        # Random x position
        x = np.random.randint(100, frame_width - 200)
        # Random y position (spread across frame)
        y = np.random.randint(100, frame_height - 100)

        # Create bbox
        w = np.random.randint(80, 150)
        h = np.random.randint(80, 150)

        det = Detection(
            bbox=(x, y, x + w, y + h),
            confidence=0.7 + np.random.random() * 0.3,
            class_id=0
        )
        detections.append(det)

    return detections


def benchmark_tracker(num_frames=100, detections_per_frame=5):
    """Benchmark tracker performance."""
    print("=" * 70)
    print(f"Performance Benchmark: {num_frames} frames, {detections_per_frame} detections/frame")
    print("=" * 70)

    config = TrackingConfig()
    tracker = ConveyorTracker(config)

    frame_height = 720
    frame_width = 1280

    # Warmup
    print("\nWarming up...")
    for _ in range(10):
        dets = generate_test_detections(detections_per_frame, frame_height, frame_width)
        tracker.update(dets, (frame_height, frame_width))

    # Clear tracks after warmup
    tracker.tracks.clear()
    tracker.completed_tracks.clear()

    # Actual benchmark
    print(f"\nRunning benchmark...")
    start_time = time.time()
    frame_times = []

    for frame_idx in range(num_frames):
        # Generate detections moving upward
        dets = []
        for i in range(detections_per_frame):
            # Simulate upward movement
            y = frame_height - 100 - (frame_idx * 5) - (i * 100)
            y = max(50, y % frame_height)
            x = 200 + (i * 200)

            det = Detection(
                bbox=(x, y, x + 100, y + 100),
                confidence=0.8,
                class_id=0
            )
            dets.append(det)

        frame_start = time.time()
        tracker.update(dets, (frame_height, frame_width))
        frame_end = time.time()

        frame_times.append((frame_end - frame_start) * 1000)  # Convert to ms

    end_time = time.time()
    total_time = end_time - start_time

    # Statistics
    frame_times_np = np.array(frame_times)
    avg_frame_time = np.mean(frame_times_np)
    min_frame_time = np.min(frame_times_np)
    max_frame_time = np.max(frame_times_np)
    p95_frame_time = np.percentile(frame_times_np, 95)
    p99_frame_time = np.percentile(frame_times_np, 99)

    fps = num_frames / total_time

    print(f"\n{'Results':-^70}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average FPS: {fps:.1f}")
    print(f"\nFrame Processing Time:")
    print(f"  Average: {avg_frame_time:.2f}ms")
    print(f"  Min:     {min_frame_time:.2f}ms")
    print(f"  Max:     {max_frame_time:.2f}ms")
    print(f"  P95:     {p95_frame_time:.2f}ms")
    print(f"  P99:     {p99_frame_time:.2f}ms")
    print(f"\nTracks created: {tracker._next_id - 1}")
    print(f"Completed tracks: {len(tracker.completed_tracks)}")

    # Performance assessment
    print(f"\n{'Performance Assessment':-^70}")
    if avg_frame_time < 10:
        print("✓ EXCELLENT: Avg frame time < 10ms")
    elif avg_frame_time < 30:
        print("✓ GOOD: Avg frame time < 30ms")
    elif avg_frame_time < 50:
        print("⚠ ACCEPTABLE: Avg frame time < 50ms")
    else:
        print("✗ POOR: Avg frame time >= 50ms - needs optimization")

    if max_frame_time < 100:
        print("✓ GOOD: Max frame time < 100ms")
    elif max_frame_time < 300:
        print("⚠ ACCEPTABLE: Max frame time < 300ms")
    else:
        print("✗ POOR: Max frame time >= 300ms - check for bottlenecks")

    print("=" * 70)

    return {
        'avg_frame_time_ms': avg_frame_time,
        'max_frame_time_ms': max_frame_time,
        'fps': fps,
        'total_time_s': total_time
    }


def benchmark_exit_zone_scenario():
    """Benchmark the specific scenario: tracks in exit zone (previously slow)."""
    print("\n" + "=" * 70)
    print("Exit Zone Scenario Benchmark (Previously Slow)")
    print("=" * 70)

    config = TrackingConfig()
    tracker = ConveyorTracker(config)

    frame_height = 720
    frame_width = 1280

    # Create tracks that will be in the exit zone
    print("\nSetting up tracks in exit zone...")

    # Create 5 tracks moving towards exit
    initial_y_positions = [200, 150, 100, 80, 50]

    for idx, y in enumerate(initial_y_positions):
        x = 300 + (idx * 200)
        det = Detection(
            bbox=(x, y + 200, x + 100, y + 300),
            confidence=0.85,
            class_id=0
        )
        tracker.update([det], (frame_height, frame_width))

    print(f"Created {len(tracker.tracks)} tracks")

    # Now simulate 100 frames where tracks are in/near exit zone
    print("Running benchmark with tracks near exit zone...")

    frame_times = []
    num_frames = 100

    for frame_idx in range(num_frames):
        # Move tracks upward (towards exit)
        dets = []
        for idx, initial_y in enumerate(initial_y_positions):
            y = initial_y - (frame_idx * 2)  # Move up 2 pixels per frame
            y = max(10, y)  # Keep in frame
            x = 300 + (idx * 200)

            # Some tracks stop being detected (simulate loss)
            if y > 20 or frame_idx < 50:
                det = Detection(
                    bbox=(x, y, x + 100, y + 100),
                    confidence=0.8,
                    class_id=0
                )
                dets.append(det)

        frame_start = time.time()
        tracker.update(dets, (frame_height, frame_width))
        frame_end = time.time()

        frame_times.append((frame_end - frame_start) * 1000)

    # Statistics
    frame_times_np = np.array(frame_times)
    avg_frame_time = np.mean(frame_times_np)
    max_frame_time = np.max(frame_times_np)
    p95_frame_time = np.percentile(frame_times_np, 95)

    print(f"\n{'Results (Exit Zone Scenario)':-^70}")
    print(f"Average frame time: {avg_frame_time:.2f}ms")
    print(f"Max frame time:     {max_frame_time:.2f}ms")
    print(f"P95 frame time:     {p95_frame_time:.2f}ms")

    print(f"\n{'Performance Assessment':-^70}")
    if avg_frame_time < 30:
        print("✓ EXCELLENT: Exit zone handling is optimized (< 30ms avg)")
    elif avg_frame_time < 50:
        print("✓ GOOD: Exit zone handling is acceptable (< 50ms avg)")
    else:
        print("✗ NEEDS WORK: Exit zone handling is slow (>= 50ms avg)")

    if max_frame_time < 100:
        print("✓ EXCELLENT: No frame spikes (< 100ms max)")
    elif max_frame_time < 300:
        print("✓ GOOD: Acceptable frame spikes (< 300ms max)")
    else:
        print("✗ ISSUE: Large frame spikes detected (>= 300ms max)")

    print("=" * 70)

    return {
        'avg_frame_time_ms': avg_frame_time,
        'max_frame_time_ms': max_frame_time
    }


if __name__ == "__main__":
    print("\n🚀 ConveyorTracker Performance Benchmark\n")

    # Test 1: General performance
    results1 = benchmark_tracker(num_frames=200, detections_per_frame=5)

    # Test 2: Exit zone specific (the problematic scenario)
    results2 = benchmark_exit_zone_scenario()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nGeneral Performance:")
    print(f"  Average: {results1['avg_frame_time_ms']:.2f}ms")
    print(f"  FPS:     {results1['fps']:.1f}")

    print(f"\nExit Zone Performance:")
    print(f"  Average: {results2['avg_frame_time_ms']:.2f}ms")
    print(f"  Max:     {results2['max_frame_time_ms']:.2f}ms")

    print(f"\n{'Status':-^70}")
    if results1['avg_frame_time_ms'] < 30 and results2['max_frame_time_ms'] < 100:
        print("✅ PRODUCTION READY: Excellent performance!")
    elif results1['avg_frame_time_ms'] < 50 and results2['max_frame_time_ms'] < 300:
        print("✅ PRODUCTION READY: Good performance")
    else:
        print("⚠️  NEEDS REVIEW: Performance could be improved")

    print("=" * 70)
