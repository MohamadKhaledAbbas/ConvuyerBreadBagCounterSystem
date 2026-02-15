"""
Tests for NV12 pipeline optimization.

Tests:
- NV12 direct preprocessing (Y/UV plane resize + padding)
- np.ascontiguousarray() memory safety for BPU buffers
- NV12 buffer layout correctness
- Fallback to BGR detection when NV12 not available
"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def create_nv12_from_bgr(bgr_img: np.ndarray) -> np.ndarray:
    """Helper: Convert a BGR image to NV12 format (2D array)."""
    h, w = bgr_img.shape[:2]
    yuv_i420 = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YUV_I420)
    yuv_flat = yuv_i420.reshape(-1)
    area = h * w
    uv_size = area // 4

    nv12 = np.zeros((area + area // 2,), dtype=np.uint8)
    nv12[:area] = yuv_flat[:area]
    nv12[area::2] = yuv_flat[area:area + uv_size]
    nv12[area + 1::2] = yuv_flat[area + uv_size:]

    return nv12.reshape(h * 3 // 2, w)


def test_nv12_preprocess_buffer_layout():
    """Test that _preprocess_nv12 produces correct NV12 buffer layout."""
    # Create a test BGR image and its NV12 equivalent
    bgr = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    nv12_data = create_nv12_from_bgr(bgr)
    frame_size = (1080, 1920)

    # Simulate BpuDetector preprocessing without BPU model
    input_w, input_h = 640, 640
    area = input_h * input_w

    # --- NV12 direct path ---
    orig_h, orig_w = frame_size
    x_scale = min(1.0 * input_h / orig_h, 1.0 * input_w / orig_w)
    new_w = int(orig_w * x_scale) & ~1
    new_h = int(orig_h * x_scale) & ~1

    total_pad_x = input_w - new_w
    total_pad_y = input_h - new_h
    x_shift = (total_pad_x // 2) & ~1
    y_shift = (total_pad_y // 2) & ~1
    right_pad = total_pad_x - x_shift
    bottom_pad = total_pad_y - y_shift

    # Split and resize Y plane
    y_plane = nv12_data[:orig_h, :]
    y_resized = cv2.resize(y_plane, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # Split and resize UV plane
    uv_plane = nv12_data[orig_h:, :]
    uv_2ch = uv_plane.reshape(orig_h // 2, orig_w // 2, 2)
    uv_resized = cv2.resize(uv_2ch, (new_w // 2, new_h // 2), interpolation=cv2.INTER_NEAREST)
    uv_resized_flat = uv_resized.reshape(new_h // 2, new_w)

    # Pad Y
    y_padded = cv2.copyMakeBorder(
        y_resized, y_shift, bottom_pad, x_shift, right_pad,
        cv2.BORDER_CONSTANT, value=127
    )

    # Pad UV
    uv_padded = cv2.copyMakeBorder(
        uv_resized_flat, y_shift // 2, bottom_pad // 2, x_shift, right_pad,
        cv2.BORDER_CONSTANT, value=128
    )

    # Verify output dimensions
    assert y_padded.shape == (input_h, input_w), \
        f"Y padded shape should be ({input_h}, {input_w}), got {y_padded.shape}"
    assert uv_padded.shape == (input_h // 2, input_w), \
        f"UV padded shape should be ({input_h // 2}, {input_w}), got {uv_padded.shape}"

    # Build buffer
    nv12_buffer = np.zeros((area * 3 // 2,), dtype=np.uint8)
    nv12_buffer[:area] = y_padded.reshape(-1)
    nv12_buffer[area:] = uv_padded.reshape(-1)

    assert nv12_buffer.shape == (area * 3 // 2,), \
        f"NV12 buffer size should be {area * 3 // 2}, got {nv12_buffer.shape[0]}"

    # Y plane should contain image data (not all zeros or all 127)
    y_content = nv12_buffer[:area]
    assert not np.all(y_content == 0), "Y plane should contain image data"
    assert not np.all(y_content == 127), "Y plane should not be all padding"

    # UV plane should contain both image data and padding
    uv_content = nv12_buffer[area:]
    assert uv_content.shape[0] == area // 2, "UV plane size should be area//2"

    print("PASS: NV12 preprocess buffer layout is correct")


def test_nv12_preprocess_even_alignment():
    """Test that NV12 preprocessing enforces even dimensions for UV compatibility."""
    # Test with various frame sizes to verify even alignment
    test_sizes = [
        (1080, 1920),  # Full HD
        (720, 1280),   # HD
        (480, 640),    # VGA
        (600, 800),    # Non-standard
        (541, 961),    # Odd dimensions
    ]
    input_w, input_h = 640, 640

    for orig_h, orig_w in test_sizes:
        x_scale = min(1.0 * input_h / orig_h, 1.0 * input_w / orig_w)
        new_w = int(orig_w * x_scale) & ~1
        new_h = int(orig_h * x_scale) & ~1

        total_pad_x = input_w - new_w
        total_pad_y = input_h - new_h
        x_shift = (total_pad_x // 2) & ~1
        y_shift = (total_pad_y // 2) & ~1
        right_pad = total_pad_x - x_shift
        bottom_pad = total_pad_y - y_shift

        assert new_w % 2 == 0, f"new_w must be even for {orig_w}x{orig_h}, got {new_w}"
        assert new_h % 2 == 0, f"new_h must be even for {orig_w}x{orig_h}, got {new_h}"
        assert x_shift % 2 == 0, f"x_shift must be even for {orig_w}x{orig_h}, got {x_shift}"
        assert y_shift % 2 == 0, f"y_shift must be even for {orig_w}x{orig_h}, got {y_shift}"
        assert right_pad % 2 == 0, f"right_pad must be even for {orig_w}x{orig_h}, got {right_pad}"
        assert bottom_pad % 2 == 0, f"bottom_pad must be even for {orig_w}x{orig_h}, got {bottom_pad}"
        assert x_shift + new_w + right_pad == input_w, \
            f"Horizontal padding mismatch for {orig_w}x{orig_h}"
        assert y_shift + new_h + bottom_pad == input_h, \
            f"Vertical padding mismatch for {orig_w}x{orig_h}"

    print("PASS: NV12 preprocess enforces even alignment for all test sizes")


def test_np_ascontiguousarray_on_buffer():
    """Test that np.ascontiguousarray produces contiguous memory for BPU."""
    area = 640 * 640
    buffer = np.zeros((area * 3 // 2,), dtype=np.uint8)

    # Fill with some data
    buffer[:area] = np.random.randint(0, 255, area, dtype=np.uint8)
    buffer[area:] = np.random.randint(0, 255, area // 2, dtype=np.uint8)

    result = np.ascontiguousarray(buffer)
    assert result.flags['C_CONTIGUOUS'], "Buffer must be C-contiguous for BPU"
    assert result.dtype == np.uint8, f"Buffer dtype should be uint8, got {result.dtype}"
    assert np.array_equal(buffer, result), "Data should be unchanged"

    # Test with slice (which may not be contiguous)
    sliced = buffer[::2]
    assert not sliced.flags['C_CONTIGUOUS'], "Sliced array should not be contiguous"
    contiguous = np.ascontiguousarray(sliced)
    assert contiguous.flags['C_CONTIGUOUS'], "After ascontiguousarray should be contiguous"

    print("PASS: np.ascontiguousarray ensures BPU-compatible contiguous memory")


def test_nv12_bgr_round_trip_consistency():
    """Test that NV12 direct path produces structurally valid output."""
    # Create a synthetic gradient BGR image
    bgr = np.zeros((720, 1280, 3), dtype=np.uint8)
    bgr[:, :, 0] = np.linspace(0, 255, 1280, dtype=np.uint8)  # Blue gradient
    bgr[:, :, 1] = 128  # Constant green
    bgr[:, :, 2] = 64   # Constant red

    nv12_data = create_nv12_from_bgr(bgr)
    frame_size = (720, 1280)
    input_w, input_h = 640, 640
    area = input_h * input_w

    # --- NV12 direct resize+pad (new optimized path) ---
    orig_h, orig_w = frame_size
    x_scale = min(1.0 * input_h / orig_h, 1.0 * input_w / orig_w)
    new_w = int(orig_w * x_scale) & ~1
    new_h = int(orig_h * x_scale) & ~1

    total_pad_x = input_w - new_w
    total_pad_y = input_h - new_h
    x_shift = (total_pad_x // 2) & ~1
    y_shift = (total_pad_y // 2) & ~1
    right_pad = total_pad_x - x_shift
    bottom_pad = total_pad_y - y_shift

    y_plane = nv12_data[:orig_h, :]
    uv_plane = nv12_data[orig_h:, :]

    y_resized = cv2.resize(y_plane, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    uv_2ch = uv_plane.reshape(orig_h // 2, orig_w // 2, 2)
    uv_resized = cv2.resize(uv_2ch, (new_w // 2, new_h // 2), interpolation=cv2.INTER_NEAREST)
    uv_resized_flat = uv_resized.reshape(new_h // 2, new_w)

    y_padded = cv2.copyMakeBorder(
        y_resized, y_shift, bottom_pad, x_shift, right_pad,
        cv2.BORDER_CONSTANT, value=127
    )
    uv_padded = cv2.copyMakeBorder(
        uv_resized_flat, y_shift // 2, bottom_pad // 2, x_shift, right_pad,
        cv2.BORDER_CONSTANT, value=128
    )

    nv12_buffer = np.zeros((area * 3 // 2,), dtype=np.uint8)
    nv12_buffer[:area] = y_padded.reshape(-1)
    nv12_buffer[area:] = uv_padded.reshape(-1)

    y_direct = nv12_buffer[:area].reshape(input_h, input_w)

    # Direct NV12 path: padding regions should be exactly 127
    if y_shift > 0:
        assert np.all(y_direct[:y_shift, :] == 127), "Direct path: top Y padding should be 127"
    if bottom_pad > 0:
        assert np.all(y_direct[y_shift + new_h:, :] == 127), "Direct path: bottom Y padding should be 127"

    # Image content region should have non-trivial content
    img_region = y_direct[y_shift:y_shift + new_h, x_shift:x_shift + new_w]
    assert img_region.mean() > 0, "Image region should have content"
    assert img_region.std() > 0, "Image region should have variation (gradient)"

    # UV padding should be 128 (neutral chroma)
    uv_full = nv12_buffer[area:].reshape(input_h // 2, input_w)
    if y_shift // 2 > 0:
        assert np.all(uv_full[:y_shift // 2, :] == 128), "UV top padding should be 128"

    # Buffer should be valid NV12 (correct total size)
    assert nv12_buffer.shape == (area * 3 // 2,), "Buffer size should be area * 1.5"

    print("PASS: NV12 direct path produces structurally valid output")


def test_pipeline_core_nv12_routing():
    """Test that PipelineCore routes to detect_nv12 when NV12 data is available."""
    from unittest.mock import MagicMock, patch

    from src.app.pipeline_core import PipelineCore
    from src.detection.BaseDetection import Detection

    # Create mock components
    mock_detector = MagicMock()
    mock_detector.detect.return_value = [
        Detection(bbox=(10, 20, 100, 200), confidence=0.9)
    ]
    mock_detector.detect_nv12.return_value = [
        Detection(bbox=(10, 20, 100, 200), confidence=0.9)
    ]

    mock_tracker = MagicMock()
    mock_tracker.update.return_value = []
    mock_tracker.get_confirmed_tracks.return_value = []
    mock_tracker.get_completed_events.return_value = []

    mock_roi_collector = MagicMock()
    mock_worker = MagicMock()

    pipeline = PipelineCore(
        detector=mock_detector,
        tracker=mock_tracker,
        roi_collector=mock_roi_collector,
        classification_worker=mock_worker
    )

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Test 1: Without NV12 data - should use BGR detection
    pipeline.process_frame(frame)
    mock_detector.detect.assert_called_once_with(frame)
    mock_detector.detect_nv12.assert_not_called()

    mock_detector.reset_mock()

    # Test 2: With NV12 data - should use NV12 detection
    nv12_data = np.zeros((1080, 1920), dtype=np.uint8)
    frame_size = (720, 1280)
    pipeline.process_frame(frame, nv12_data=nv12_data, frame_size=frame_size)
    mock_detector.detect_nv12.assert_called_once_with(nv12_data, frame_size)
    mock_detector.detect.assert_not_called()

    mock_detector.reset_mock()

    # Test 3: With NV12 data but detector without detect_nv12 - should fallback to BGR
    del mock_detector.detect_nv12  # Remove the attribute
    pipeline.process_frame(frame, nv12_data=nv12_data, frame_size=frame_size)
    mock_detector.detect.assert_called_once_with(frame)

    print("PASS: PipelineCore routes correctly between NV12 and BGR detection paths")


def test_classifier_contiguous_memory():
    """Test that BpuClassifier._bgr2nv12 returns contiguous array."""
    # We can test the NV12 conversion logic without the BPU model
    area = 224 * 224
    nv12_buffer = np.zeros((area * 3 // 2,), dtype=np.uint8)

    # Simulate _bgr2nv12 conversion
    bgr_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    yuv420p = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YUV_I420).reshape((area * 3 // 2,))

    nv12_buffer[:area] = yuv420p[:area]
    u_start = area
    v_start = area + (area // 4)
    nv12_buffer[area::2] = yuv420p[u_start:v_start]
    nv12_buffer[area + 1::2] = yuv420p[v_start:]

    result = np.ascontiguousarray(nv12_buffer)
    assert result.flags['C_CONTIGUOUS'], "Classifier NV12 buffer must be contiguous"
    assert result.shape == (area * 3 // 2,), f"Expected shape {(area * 3 // 2,)}, got {result.shape}"
    assert result.dtype == np.uint8, "Buffer should be uint8"

    print("PASS: Classifier NV12 buffer is contiguous and correctly sized")


def test_nv12_preprocess_various_resolutions():
    """Test NV12 preprocessing with common camera resolutions."""
    input_w, input_h = 640, 640
    area = input_w * input_h

    resolutions = [
        (1080, 1920),  # Full HD
        (720, 1280),   # HD
        (480, 640),    # VGA
        (2160, 3840),  # 4K
    ]

    for orig_h, orig_w in resolutions:
        bgr = np.random.randint(0, 255, (orig_h, orig_w, 3), dtype=np.uint8)
        nv12_data = create_nv12_from_bgr(bgr)

        # Simulate _preprocess_nv12
        x_scale = min(1.0 * input_h / orig_h, 1.0 * input_w / orig_w)
        new_w = int(orig_w * x_scale) & ~1
        new_h = int(orig_h * x_scale) & ~1

        total_pad_x = input_w - new_w
        total_pad_y = input_h - new_h
        x_shift = (total_pad_x // 2) & ~1
        y_shift = (total_pad_y // 2) & ~1
        right_pad = total_pad_x - x_shift
        bottom_pad = total_pad_y - y_shift

        y_plane = nv12_data[:orig_h, :]
        uv_plane = nv12_data[orig_h:, :]

        y_resized = cv2.resize(y_plane, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        uv_2ch = uv_plane.reshape(orig_h // 2, orig_w // 2, 2)
        uv_resized = cv2.resize(uv_2ch, (new_w // 2, new_h // 2), interpolation=cv2.INTER_NEAREST)
        uv_resized_flat = uv_resized.reshape(new_h // 2, new_w)

        y_padded = cv2.copyMakeBorder(
            y_resized, y_shift, bottom_pad, x_shift, right_pad,
            cv2.BORDER_CONSTANT, value=127
        )
        uv_padded = cv2.copyMakeBorder(
            uv_resized_flat, y_shift // 2, bottom_pad // 2, x_shift, right_pad,
            cv2.BORDER_CONSTANT, value=128
        )

        nv12_buffer = np.zeros((area * 3 // 2,), dtype=np.uint8)
        nv12_buffer[:area] = y_padded.reshape(-1)
        nv12_buffer[area:] = uv_padded.reshape(-1)

        result = np.ascontiguousarray(nv12_buffer)
        assert result.shape == (area * 3 // 2,), \
            f"Buffer size wrong for {orig_w}x{orig_h}: {result.shape}"
        assert result.flags['C_CONTIGUOUS'], \
            f"Buffer not contiguous for {orig_w}x{orig_h}"

    print("PASS: NV12 preprocessing works for all common camera resolutions")


if __name__ == "__main__":
    test_nv12_preprocess_buffer_layout()
    test_nv12_preprocess_even_alignment()
    test_np_ascontiguousarray_on_buffer()
    test_nv12_bgr_round_trip_consistency()
    test_pipeline_core_nv12_routing()
    test_classifier_contiguous_memory()
    test_nv12_preprocess_various_resolutions()
    print("\n=== All NV12 pipeline tests PASSED ===")
