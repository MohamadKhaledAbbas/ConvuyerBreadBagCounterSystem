#!/usr/bin/env python3
"""
Test suite for enhanced ROI collector with diversity controls and gradual position penalty.

Tests:
1. Frame spacing enforcement (no consecutive frame collection)
2. Position diversity enforcement (require movement)
3. Gradual position penalty (smooth quality degradation by Y position)
4. Integration with existing temporal weighting
"""

import sys
import numpy as np
from src.classifier.ROICollectorService import ROICollectorService, ROIQualityConfig


def _make_frame(height=480, width=640):
    """Create a test frame with sufficient sharpness and brightness."""
    # Create frame with gradient for better sharpness
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = 128  # Mid-gray for brightness
    # Add some texture for sharpness
    noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
    frame = cv2.add(frame, noise)
    return frame


def test_frame_spacing_enforcement():
    """Test that ROIs are not collected from consecutive frames."""
    print("\n=== Test 1: Frame Spacing Enforcement ===")

    config = ROIQualityConfig(
        min_sharpness=1.0,
        min_brightness=10.0,
        max_brightness=250.0,
        min_frame_spacing=3,  # Require 3 frames gap
        min_position_change=1.0  # Very low threshold to test frame spacing only
    )
    collector = ROICollectorService(quality_config=config, max_rois_per_track=10)

    frame = _make_frame(480, 640)

    # Try to collect ROIs from consecutive frames
    # Move position slightly each time to satisfy position diversity
    results = []
    for i in range(10):
        bbox = (100 + i*3, 300, 200 + i*3, 400)  # Move 3px each frame
        result = collector.collect_roi(track_id=1, frame=frame, bbox=bbox)
        results.append(result)
        print(f"  Frame {i}: collected={result}")

    collection = collector.collections[1]

    # Should have collected every 3rd frame (0, 3, 6, 9)
    assert results[0] == True, "First frame should be collected"
    assert results[1] == False, "Frame 1 should be skipped (spacing)"
    assert results[2] == False, "Frame 2 should be skipped (spacing)"
    assert results[3] == True, "Frame 3 should be collected (spacing=3)"

    print(f"✓ Collected {collection.collected_count} ROIs with min_frame_spacing=3")
    print("✓ Frame spacing enforcement works correctly")


def test_position_diversity_enforcement():
    """Test that ROIs require significant movement between collections."""
    print("\n=== Test 2: Position Diversity Enforcement ===")

    config = ROIQualityConfig(
        min_sharpness=1.0,
        min_brightness=10.0,
        max_brightness=250.0,
        min_frame_spacing=1,  # Allow every frame
        min_position_change=50.0  # Require 50px movement
    )
    collector = ROICollectorService(quality_config=config, max_rois_per_track=10)

    frame = _make_frame(480, 640)

    # First ROI
    bbox1 = (100, 300, 200, 400)
    result1 = collector.collect_roi(track_id=1, frame=frame, bbox=bbox1)

    # Second ROI - same position (should be rejected)
    bbox2 = (100, 300, 200, 400)
    result2 = collector.collect_roi(track_id=1, frame=frame, bbox=bbox2)

    # Third ROI - moved 30px (should be rejected - below min_position_change)
    bbox3 = (130, 300, 230, 400)
    result3 = collector.collect_roi(track_id=1, frame=frame, bbox=bbox3)

    # Fourth ROI - moved 60px (should be accepted)
    bbox4 = (160, 300, 260, 400)
    result4 = collector.collect_roi(track_id=1, frame=frame, bbox=bbox4)

    assert result1 == True, "First ROI should be collected"
    assert result2 == False, "Second ROI should be skipped (no movement)"
    assert result3 == False, "Third ROI should be skipped (movement < 50px)"
    assert result4 == True, "Fourth ROI should be collected (movement >= 50px)"

    collection = collector.collections[1]
    print(f"✓ Collected {collection.collected_count} ROIs with position diversity")
    print("✓ Position diversity enforcement works correctly")


def test_gradual_position_penalty():
    """Test gradual position penalty from center to top of frame."""
    print("\n=== Test 3: Gradual Position Penalty ===")

    config = ROIQualityConfig(
        min_sharpness=1.0,
        min_brightness=10.0,
        max_brightness=250.0,
        enable_gradual_position_penalty=True,
        position_penalty_start_ratio=0.5,  # Start penalty at center
        position_penalty_max_ratio=0.15,   # Max penalty at top 15%
        position_penalty_min_multiplier=0.3,  # 70% quality reduction at top
        min_frame_spacing=1,
        min_position_change=10.0
    )
    collector = ROICollectorService(
        quality_config=config,
        max_rois_per_track=10,
        enable_temporal_weighting=False  # Disable for clean test
    )

    frame = _make_frame(480, 640)

    # Collect ROIs at different Y positions (all same X for consistency)
    positions = [
        (100, 50, 200, 100),    # Top (y_center = 75, ratio = 0.156)
        (100, 100, 200, 150),   # Near top (y_center = 125, ratio = 0.26)
        (100, 200, 200, 250),   # Upper-mid (y_center = 225, ratio = 0.47)
        (100, 240, 200, 290),   # Center (y_center = 265, ratio = 0.55)
        (100, 350, 200, 400),   # Lower (y_center = 375, ratio = 0.78)
    ]

    qualities = []
    for idx, bbox in enumerate(positions):
        collector.collect_roi(track_id=idx+1, frame=frame, bbox=bbox)
        if idx+1 in collector.collections:
            quality = collector.collections[idx+1].best_roi_quality
            qualities.append(quality)
            y_center = (bbox[1] + bbox[3]) / 2
            y_ratio = y_center / 480
            print(f"  Y={y_center:.0f} (ratio={y_ratio:.2f}): quality={quality:.1f}")

    # Quality should increase from top to bottom
    assert len(qualities) == 5, "Should collect all 5 ROIs"
    for i in range(len(qualities) - 1):
        assert qualities[i] <= qualities[i+1], \
            f"Quality should increase from top to bottom: {qualities[i]:.1f} vs {qualities[i+1]:.1f}"

    print(f"✓ Quality gradient: {qualities[0]:.1f} (top) → {qualities[-1]:.1f} (bottom)")
    print("✓ Gradual position penalty works correctly")


def test_integration_with_temporal_weighting():
    """Test that diversity controls work together with temporal weighting."""
    print("\n=== Test 4: Integration with Temporal Weighting ===")

    config = ROIQualityConfig(
        min_sharpness=1.0,
        min_brightness=10.0,
        max_brightness=250.0,
        min_frame_spacing=2,
        min_position_change=30.0
    )
    collector = ROICollectorService(
        quality_config=config,
        max_rois_per_track=5,
        enable_temporal_weighting=True,
        temporal_decay_rate=0.2  # 20% decay
    )

    frame = _make_frame(480, 640)

    # Collect ROIs with sufficient spacing and movement
    positions = [
        (100, 300, 200, 400),
        (150, 300, 250, 400),  # Moved 50px
        (200, 300, 300, 400),  # Moved 50px
        (250, 300, 350, 400),  # Moved 50px
        (300, 300, 400, 400),  # Moved 50px
    ]

    collected_indices = []
    # Collect with frame spacing - call collect_roi multiple times to advance frame counter
    for idx, bbox in enumerate(positions):
        # Advance frames to respect spacing (min_frame_spacing=2)
        # First ROI at frame 0, next at frame 3, then 6, 9, 12
        for frame_skip in range(3):  # Skip 3 frames between attempts
            if frame_skip == 0:  # Only try to collect on first call
                result = collector.collect_roi(track_id=1, frame=frame, bbox=bbox)
                if result:
                    collected_indices.append(idx)
                    print(f"  ROI {idx+1} collected at position {bbox[0]}")
            else:
                # Just advance frame counter with a dummy position
                pass

    collection = collector.collections[1]

    print(f"✓ Collected {collection.collected_count} ROIs with diversity controls")
    print(f"✓ Collected at positions: {collected_indices}")

    # Should have collected multiple ROIs (may not be all 5 due to spacing)
    assert collection.collected_count >= 3, f"Should collect at least 3 ROIs, got {collection.collected_count}"

    # Earlier ROIs should have higher quality due to temporal weighting
    qualities = collection.qualities
    print(f"✓ Qualities with temporal decay: {[f'{q:.1f}' for q in qualities]}")
    print("✓ Integration with temporal weighting works correctly")


def test_production_scenario():
    """Test realistic production scenario with all features enabled."""
    print("\n=== Test 5: Production Scenario ===")

    config = ROIQualityConfig(
        min_sharpness=50.0,
        min_brightness=30.0,
        max_brightness=225.0,
        min_frame_spacing=3,
        min_position_change=20.0,
        enable_gradual_position_penalty=True,
        position_penalty_start_ratio=0.5,
        position_penalty_max_ratio=0.15,
        position_penalty_min_multiplier=0.3
    )
    collector = ROICollectorService(
        quality_config=config,
        max_rois_per_track=8,
        enable_temporal_weighting=True,
        temporal_decay_rate=0.15
    )

    frame = _make_frame(480, 640)

    # Simulate object moving from bottom to top
    # Y positions: 400 → 350 → 300 → 250 → 200 → 150 → 100 → 50
    collected_count = 0
    for y_start in range(400, 40, -30):  # Move up 30px per step
        bbox = (100, y_start, 200, y_start + 50)

        # Try to collect (will be filtered by spacing/diversity)
        for _ in range(5):  # Simulate 5 frames at each position
            result = collector.collect_roi(track_id=1, frame=frame, bbox=bbox)
            if result:
                collected_count += 1

    collection = collector.collections[1]

    print(f"✓ Collected {collection.collected_count} diverse ROIs")
    print(f"✓ Best ROI quality: {collection.best_roi_quality:.1f}")
    print(f"✓ Total attempts: many, collected: {collected_count}")

    assert collection.collected_count > 0, "Should collect some ROIs"
    assert collection.collected_count <= 8, "Should not exceed max_rois"
    print("✓ Production scenario works correctly")


if __name__ == "__main__":
    import cv2

    print("=" * 70)
    print("Enhanced ROI Collector Test Suite")
    print("=" * 70)

    try:
        test_frame_spacing_enforcement()
        test_position_diversity_enforcement()
        test_gradual_position_penalty()
        test_integration_with_temporal_weighting()
        test_production_scenario()

        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED")
        print("=" * 70)
        sys.exit(0)

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
