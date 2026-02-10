#!/usr/bin/env python3
"""
Test script for ROI saving functionality.
"""

import sys
import os
import tempfile
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np

from src.classifier.ROICollectorService import ROICollectorService, ROIQualityConfig
from src.utils.AppLogging import logger


def create_test_roi(quality_level='good'):
    """Create a test ROI image with specified quality."""
    if quality_level == 'good':
        # Sharp, well-lit image
        roi = np.random.randint(80, 180, (100, 150, 3), dtype=np.uint8)
        # Add some edges for sharpness
        roi[30:70, 50:100] = 255
    elif quality_level == 'blurry':
        # Blurry image (smooth)
        roi = np.random.randint(100, 150, (100, 150, 3), dtype=np.uint8)
        roi = cv2.GaussianBlur(roi, (21, 21), 10)
    elif quality_level == 'dark':
        # Too dark
        roi = np.random.randint(0, 30, (100, 150, 3), dtype=np.uint8)
    elif quality_level == 'bright':
        # Too bright
        roi = np.random.randint(230, 255, (100, 150, 3), dtype=np.uint8)
    else:
        roi = np.random.randint(50, 200, (100, 150, 3), dtype=np.uint8)

    return roi


def test_save_roi_candidates():
    """Test saving only accepted ROI candidates."""
    logger.info("=" * 60)
    logger.info("TEST 1: Save ROI candidates (accepted only)")
    logger.info("=" * 60)

    # Create temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create ROI collector with save_roi_candidates enabled
        collector = ROICollectorService(
            quality_config=ROIQualityConfig(),
            max_rois_per_track=5,
            save_roi_candidates=True,
            save_all_rois=False,
            roi_candidates_dir=tmpdir
        )

        # Create test frame
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

        # Collect good ROI
        good_roi = create_test_roi('good')
        frame[100:200, 100:250] = good_roi
        collector.collect_roi(track_id=1, frame=frame, bbox=(100, 100, 250, 200))

        # Collect blurry ROI (should be rejected)
        blurry_roi = create_test_roi('blurry')
        frame[300:400, 100:250] = blurry_roi
        collector.collect_roi(track_id=2, frame=frame, bbox=(100, 300, 250, 400))

        # Check saved files
        saved_files = glob.glob(f"{tmpdir}/*.jpg")
        accepted_files = [f for f in saved_files if 'accepted' in f]
        rejected_files = [f for f in saved_files if 'rejected' in f]

        logger.info(f"Total files saved: {len(saved_files)}")
        logger.info(f"Accepted files: {len(accepted_files)}")
        logger.info(f"Rejected files: {len(rejected_files)}")

        # With save_roi_candidates=True, only accepted should be saved
        if len(accepted_files) > 0 and len(rejected_files) == 0:
            logger.info("✓ TEST 1 PASSED: Only accepted ROIs saved")
            return True
        else:
            logger.error("✗ TEST 1 FAILED")
            return False


def test_save_all_rois():
    """Test saving all ROIs (accepted and rejected)."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST 2: Save all ROIs (accepted + rejected)")
    logger.info("=" * 60)

    # Create temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create ROI collector with save_all_rois enabled
        collector = ROICollectorService(
            quality_config=ROIQualityConfig(),
            max_rois_per_track=5,
            save_roi_candidates=False,
            save_all_rois=True,
            roi_candidates_dir=tmpdir
        )

        # Create test frame
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

        # Collect multiple quality levels
        test_cases = [
            ('good', 1, (100, 100, 250, 200)),
            ('blurry', 2, (100, 300, 250, 400)),
            ('dark', 3, (400, 100, 550, 200)),
            ('bright', 4, (400, 300, 550, 400))
        ]

        for quality_level, track_id, bbox in test_cases:
            roi = create_test_roi(quality_level)
            x1, y1, x2, y2 = bbox
            frame[y1:y2, x1:x2] = roi
            collector.collect_roi(track_id=track_id, frame=frame, bbox=bbox)

        # Check saved files
        saved_files = glob.glob(f"{tmpdir}/*.jpg")
        accepted_files = [f for f in saved_files if 'accepted' in f]
        rejected_files = [f for f in saved_files if 'rejected' in f]

        logger.info(f"Total files saved: {len(saved_files)}")
        logger.info(f"Accepted files: {len(accepted_files)}")
        logger.info(f"Rejected files: {len(rejected_files)}")

        # List files
        for f in saved_files:
            logger.info(f"  - {os.path.basename(f)}")

        # With save_all_rois=True, both accepted and rejected should be saved
        if len(accepted_files) > 0 and len(rejected_files) > 0:
            logger.info("✓ TEST 2 PASSED: Both accepted and rejected ROIs saved")
            return True
        else:
            logger.error("✗ TEST 2 FAILED")
            return False


def test_no_saving():
    """Test that no ROIs are saved when both flags are False."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST 3: No saving (both flags False)")
    logger.info("=" * 60)

    # Create temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create ROI collector with both flags disabled
        collector = ROICollectorService(
            quality_config=ROIQualityConfig(),
            max_rois_per_track=5,
            save_roi_candidates=False,
            save_all_rois=False,
            roi_candidates_dir=tmpdir
        )

        # Create test frame
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

        # Collect ROI
        good_roi = create_test_roi('good')
        frame[100:200, 100:250] = good_roi
        collector.collect_roi(track_id=1, frame=frame, bbox=(100, 100, 250, 200))

        # Check saved files
        saved_files = glob.glob(f"{tmpdir}/*.jpg")

        logger.info(f"Total files saved: {len(saved_files)}")

        # With both flags False, no files should be saved
        if len(saved_files) == 0:
            logger.info("✓ TEST 3 PASSED: No ROIs saved when disabled")
            return True
        else:
            logger.error("✗ TEST 3 FAILED")
            return False


def test_filename_format():
    """Test that saved files have correct naming format."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST 4: Filename format validation")
    logger.info("=" * 60)

    # Create temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create ROI collector
        collector = ROICollectorService(
            quality_config=ROIQualityConfig(),
            max_rois_per_track=5,
            save_roi_candidates=False,
            save_all_rois=True,
            roi_candidates_dir=tmpdir
        )

        # Create test frame
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

        # Collect ROIs
        good_roi = create_test_roi('good')
        frame[100:200, 100:250] = good_roi
        collector.collect_roi(track_id=42, frame=frame, bbox=(100, 100, 250, 200))

        blurry_roi = create_test_roi('blurry')
        frame[300:400, 100:250] = blurry_roi
        collector.collect_roi(track_id=43, frame=frame, bbox=(100, 300, 250, 400))

        # Check saved files
        saved_files = glob.glob(f"{tmpdir}/*.jpg")

        all_valid = True
        for filepath in saved_files:
            filename = os.path.basename(filepath)
            logger.info(f"Checking filename: {filename}")

            # Check format: track_{id}_{timestamp}_{status}_q{quality}[_{reason}].jpg
            parts = filename.replace('.jpg', '').split('_')

            if parts[0] != 'track':
                logger.error(f"  ✗ Missing 'track' prefix")
                all_valid = False
                continue

            if not parts[1].isdigit():
                logger.error(f"  ✗ Invalid track ID: {parts[1]}")
                all_valid = False
                continue

            if not parts[2].isdigit():
                logger.error(f"  ✗ Invalid timestamp: {parts[2]}")
                all_valid = False
                continue

            if parts[3] not in ['accepted', 'rejected']:
                logger.error(f"  ✗ Invalid status: {parts[3]}")
                all_valid = False
                continue

            if not parts[4].startswith('q'):
                logger.error(f"  ✗ Invalid quality format: {parts[4]}")
                all_valid = False
                continue

            logger.info(f"  ✓ Valid format")

        if all_valid and len(saved_files) > 0:
            logger.info("✓ TEST 4 PASSED: All filenames have correct format")
            return True
        else:
            logger.error("✗ TEST 4 FAILED")
            return False


def main():
    """Run all tests."""
    logger.info("Testing ROI Saving Functionality")
    logger.info("")

    results = []

    try:
        results.append(("Save candidates only", test_save_roi_candidates()))
        results.append(("Save all ROIs", test_save_all_rois()))
        results.append(("No saving", test_no_saving()))
        results.append(("Filename format", test_filename_format()))

        logger.info("")
        logger.info("=" * 60)
        logger.info("TEST RESULTS SUMMARY")
        logger.info("=" * 60)

        for name, passed in results:
            status = "✓ PASSED" if passed else "✗ FAILED"
            logger.info(f"{status}: {name}")

        all_passed = all(r[1] for r in results)

        logger.info("")
        if all_passed:
            logger.info("🎉 ALL TESTS PASSED!")
            logger.info("")
            logger.info("ROI saving feature is working correctly.")
            logger.info("See docs/ROI_SAVING_GUIDE.md for usage instructions.")
        else:
            logger.error("❌ SOME TESTS FAILED")

        return 0 if all_passed else 1

    except Exception as e:
        logger.error(f"Test suite failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
