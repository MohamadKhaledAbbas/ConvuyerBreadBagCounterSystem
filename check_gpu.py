"""
GPU Diagnostics and Performance Analysis Script.

Checks:
1. CUDA availability
2. Current model device usage
3. Performance bottlenecks
4. Recommendations for optimization
"""

import time
import numpy as np

def check_cuda():
    """Check CUDA availability."""
    print("=" * 60)
    print("CUDA/GPU Diagnostics")
    print("=" * 60)

    try:
        import torch
        print(f"✓ PyTorch installed: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"✓ CUDA version: {torch.version.cuda}")
            print(f"✓ GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"    Memory: {props.total_memory / 1024**3:.1f} GB")

            # Test GPU speed
            print("\nTesting GPU performance...")
            device = torch.device('cuda')
            x = torch.randn(1000, 1000, device=device)
            start = time.perf_counter()
            for _ in range(100):
                _ = torch.matmul(x, x)
            torch.cuda.synchronize()
            gpu_time = (time.perf_counter() - start) * 1000
            print(f"  GPU computation time: {gpu_time:.1f}ms")

            # Test CPU speed
            print("\nTesting CPU performance...")
            x_cpu = torch.randn(1000, 1000)
            start = time.perf_counter()
            for _ in range(100):
                _ = torch.matmul(x_cpu, x_cpu)
            cpu_time = (time.perf_counter() - start) * 1000
            print(f"  CPU computation time: {cpu_time:.1f}ms")
            print(f"  GPU speedup: {cpu_time / gpu_time:.1f}x")

            return True
        else:
            print("✗ CUDA not available")
            print("\nPossible reasons:")
            print("  1. No NVIDIA GPU installed")
            print("  2. GPU drivers not installed")
            print("  3. CUDA toolkit not installed")
            print("  4. PyTorch installed without CUDA support")
            print("\nTo fix:")
            print("  pip uninstall torch")
            print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
            return False

    except ImportError:
        print("✗ PyTorch not installed")
        print("  pip install torch torchvision")
        return False

def check_ultralytics():
    """Check Ultralytics installation."""
    print("\n" + "=" * 60)
    print("Ultralytics YOLO Diagnostics")
    print("=" * 60)

    try:
        from ultralytics import YOLO
        import ultralytics
        print(f"✓ Ultralytics installed: {ultralytics.__version__}")

        # Check if models exist
        import os
        models_to_check = [
            "data/model/detect_yolo_small_v9.pt",
            "data/model/classify_yolo_small_v11.pt"
        ]

        print("\nModel files:")
        for model_path in models_to_check:
            if os.path.exists(model_path):
                size_mb = os.path.getsize(model_path) / (1024 * 1024)
                print(f"  ✓ {model_path} ({size_mb:.1f} MB)")
            else:
                print(f"  ✗ {model_path} (NOT FOUND)")

        return True

    except ImportError:
        print("✗ Ultralytics not installed")
        print("  pip install ultralytics")
        return False

def test_inference_speed():
    """Test actual inference speed."""
    print("\n" + "=" * 60)
    print("Inference Speed Test")
    print("=" * 60)

    try:
        import torch
        from ultralytics import YOLO
        import cv2

        # Create dummy frame
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

        # Test detection model
        detect_model_path = "data/model/detect_yolo_small_v9.pt"
        if os.path.exists(detect_model_path):
            print(f"\nTesting detection model: {detect_model_path}")
            model = YOLO(detect_model_path)

            # Warmup
            for _ in range(3):
                _ = model(frame, device='cuda' if torch.cuda.is_available() else 'cpu', verbose=False)

            # Test CPU
            times_cpu = []
            for _ in range(10):
                start = time.perf_counter()
                _ = model(frame, device='cpu', verbose=False)
                times_cpu.append((time.perf_counter() - start) * 1000)

            print(f"  CPU: {np.mean(times_cpu):.1f}ms ± {np.std(times_cpu):.1f}ms")

            # Test GPU if available
            if torch.cuda.is_available():
                times_gpu = []
                for _ in range(10):
                    start = time.perf_counter()
                    _ = model(frame, device='cuda', verbose=False)
                    torch.cuda.synchronize()
                    times_gpu.append((time.perf_counter() - start) * 1000)

                print(f"  GPU: {np.mean(times_gpu):.1f}ms ± {np.std(times_gpu):.1f}ms")
                print(f"  Speedup: {np.mean(times_cpu) / np.mean(times_gpu):.1f}x")

        # Test classification model
        classify_model_path = "data/model/classify_yolo_small_v11.pt"
        if os.path.exists(classify_model_path):
            print(f"\nTesting classification model: {classify_model_path}")
            model = YOLO(classify_model_path)
            roi = frame[100:300, 100:300]  # Small ROI

            # Warmup
            for _ in range(3):
                _ = model(roi, device='cuda' if torch.cuda.is_available() else 'cpu', verbose=False)

            # Test CPU
            times_cpu = []
            for _ in range(10):
                start = time.perf_counter()
                _ = model(roi, device='cpu', verbose=False)
                times_cpu.append((time.perf_counter() - start) * 1000)

            print(f"  CPU: {np.mean(times_cpu):.1f}ms ± {np.std(times_cpu):.1f}ms")

            # Test GPU if available
            if torch.cuda.is_available():
                times_gpu = []
                for _ in range(10):
                    start = time.perf_counter()
                    _ = model(roi, device='cuda', verbose=False)
                    torch.cuda.synchronize()
                    times_gpu.append((time.perf_counter() - start) * 1000)

                print(f"  GPU: {np.mean(times_gpu):.1f}ms ± {np.std(times_gpu):.1f}ms")
                print(f"  Speedup: {np.mean(times_cpu) / np.mean(times_gpu):.1f}x")

    except Exception as e:
        print(f"✗ Error during inference test: {e}")

if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    has_cuda = check_cuda()
    has_ultralytics = check_ultralytics()

    if has_ultralytics:
        test_inference_speed()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    if has_cuda:
        print("✓ GPU is available and should be used automatically")
    else:
        print("✗ GPU not available - running on CPU (slower)")
        print("  Expected FPS: 2-3 (CPU)")
        print("  With GPU: 12-15 (CUDA)")

    print("\nTo force GPU usage, ensure:")
    print("  1. PyTorch with CUDA support installed")
    print("  2. NVIDIA drivers installed")
    print("  3. Check logs for 'device: cuda' messages")
