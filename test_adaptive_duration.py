"""
Test script to validate adaptive travel duration logic.

Run this to verify the fix for T13/T14 issue.
"""


def calculate_required_duration(entry_y: int, frame_height: int, base_duration: float = 2.0) -> float:
    """
    Calculate required duration based on entry position.

    Args:
        entry_y: Y coordinate where track first appeared
        frame_height: Total frame height
        base_duration: Base minimum duration (default 2.0s)

    Returns:
        Required duration in seconds
    """
    # Calculate where track started as a ratio (0=top, 1=bottom)
    entry_ratio = entry_y / frame_height

    # Scale the minimum duration requirement proportionally
    # Minimum 30% floor
    duration_scale = max(0.3, entry_ratio)

    return base_duration * duration_scale


def test_case(track_id: int, entry_y: int, actual_duration: float,
              frame_height: int = 720, base_duration: float = 2.0):
    """Test a specific case."""
    required_duration = calculate_required_duration(entry_y, frame_height, base_duration)
    entry_ratio = entry_y / frame_height
    duration_scale = max(0.3, entry_ratio)
    would_pass = actual_duration >= required_duration

    status = "✓ PASS" if would_pass else "✗ FAIL"

    print(f"T{track_id:2d} | entry_y={entry_y:3d} | entry_ratio={entry_ratio:.2f} | "
          f"scale={duration_scale:.2f} | required={required_duration:.2f}s | "
          f"actual={actual_duration:.2f}s | {status}")

    return would_pass


if __name__ == "__main__":
    print("=" * 100)
    print("Adaptive Travel Duration Validation Test")
    print("=" * 100)
    print()
    print("Configuration:")
    print("  - Frame Height: 720px")
    print("  - Base Duration: 2.0s")
    print("  - Min Scale: 30%")
    print()
    print("-" * 100)

    # Test the actual cases from the log
    print("\nActual Cases from Log:")
    print("-" * 100)

    t13_pass = test_case(13, entry_y=638, actual_duration=1.93)
    t14_pass = test_case(14, entry_y=469, actual_duration=3.72)

    print()
    print("-" * 100)
    print("\nAdditional Test Cases:")
    print("-" * 100)

    # Various entry positions
    test_case(1, entry_y=700, actual_duration=1.95)  # Bottom entry - should pass
    test_case(2, entry_y=700, actual_duration=1.90)  # Bottom entry - should fail
    test_case(3, entry_y=500, actual_duration=1.40)  # Mid entry - should pass
    test_case(4, entry_y=500, actual_duration=1.30)  # Mid entry - should fail
    test_case(5, entry_y=300, actual_duration=0.85)  # Upper entry - should pass
    test_case(6, entry_y=300, actual_duration=0.80)  # Upper entry - should fail
    test_case(7, entry_y=100, actual_duration=0.60)  # Top entry - should pass (at floor)
    test_case(8, entry_y=100, actual_duration=0.55)  # Top entry - should fail

    print()
    print("=" * 100)
    print("\nKey Results:")
    print("=" * 100)
    print(f"  T13 would now {'PASS ✓' if t13_pass else 'FAIL ✗'} (was rejected before fix)")
    print(f"  T14 would {'PASS ✓' if t14_pass else 'FAIL ✗'} (passed before and after)")
    print()

    if t13_pass:
        print("✓ FIX SUCCESSFUL: T13 would now be counted!")
    else:
        print("✗ FIX INCOMPLETE: T13 would still be rejected")

    print()
    print("=" * 100)
