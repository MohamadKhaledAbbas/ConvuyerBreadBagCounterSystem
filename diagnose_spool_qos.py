#!/usr/bin/env python3
"""
Diagnostic tool to check ROS2 topic QoS and subscriptions.

Usage:
    python diagnose_spool_qos.py

This will:
1. Check if /rtsp_image_ch_0 is publishing
2. Show QoS settings of publishers and subscribers
3. Test subscription with different QoS profiles
"""

import sys
import time

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
    from img_msgs.msg import H26XFrame
except ImportError as e:
    print(f"Error: Missing ROS2 dependencies: {e}")
    print("This script must be run on the RDK board with ROS2 installed.")
    sys.exit(1)


class DiagnosticNode(Node):
    """Node to test different QoS configurations."""

    def __init__(self):
        super().__init__('qos_diagnostic')
        self.frame_count = 0
        self.last_frame_time = None

    def test_subscription(self, qos_name, qos_profile):
        """Test subscription with given QoS profile."""
        print(f"\n{'='*60}")
        print(f"Testing subscription: {qos_name}")
        print(f"  Reliability: {qos_profile.reliability}")
        print(f"  History: {qos_profile.history}")
        print(f"  Depth: {qos_profile.depth}")
        print(f"{'='*60}")

        self.frame_count = 0
        self.last_frame_time = None

        # Create subscription
        subscription = self.create_subscription(
            H26XFrame,
            '/rtsp_image_ch_0',
            self._frame_callback,
            qos_profile
        )

        # Wait for frames
        print("Waiting for frames (10 seconds)...")
        start_time = time.time()
        while time.time() - start_time < 10.0:
            rclpy.spin_once(self, timeout_sec=0.1)

        # Report results
        if self.frame_count > 0:
            elapsed = time.time() - start_time
            fps = self.frame_count / elapsed
            print(f"\n✅ SUCCESS: Received {self.frame_count} frames ({fps:.1f} fps)")
        else:
            print(f"\n❌ FAILED: No frames received")

        # Cleanup
        self.destroy_subscription(subscription)
        return self.frame_count > 0

    def _frame_callback(self, msg):
        """Callback for received frames."""
        self.frame_count += 1
        now = time.time()

        if self.frame_count == 1:
            print(f"First frame received! Size: {len(msg.data)} bytes, {msg.width}x{msg.height}")

        if self.frame_count % 30 == 0:
            if self.last_frame_time:
                interval = (now - self.last_frame_time) / 30
                fps = 1.0 / interval if interval > 0 else 0
                print(f"  {self.frame_count} frames received, current fps: {fps:.1f}")

        self.last_frame_time = now


def main():
    """Run QoS diagnostics."""
    print("ROS2 Spool QoS Diagnostic Tool")
    print("="*60)

    rclpy.init()
    node = DiagnosticNode()

    # Test different QoS profiles
    qos_configs = [
        ("BEST_EFFORT + KEEP_LAST", QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )),
        ("RELIABLE + KEEP_LAST", QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )),
        ("SYSTEM_DEFAULT", QoSProfile(
            reliability=QoSReliabilityPolicy.SYSTEM_DEFAULT,
            history=QoSHistoryPolicy.SYSTEM_DEFAULT,
            depth=10
        )),
    ]

    results = {}
    for name, qos in qos_configs:
        success = node.test_subscription(name, qos)
        results[name] = success

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, success in results.items():
        status = "✅ WORKING" if success else "❌ FAILED"
        print(f"{name:30s} {status}")

    # Recommendations
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS")
    print(f"{'='*60}")

    if results["BEST_EFFORT + KEEP_LAST"]:
        print("✅ Use BEST_EFFORT + KEEP_LAST in spool_recorder_node.py")
    elif results["RELIABLE + KEEP_LAST"]:
        print("✅ Use RELIABLE + KEEP_LAST in spool_recorder_node.py")
    elif results["SYSTEM_DEFAULT"]:
        print("✅ Use SYSTEM_DEFAULT in spool_recorder_node.py")
    else:
        print("❌ No QoS configuration worked!")
        print("   Possible issues:")
        print("   - hobot_rtsp_client is not publishing to /rtsp_image_ch_0")
        print("   - Topic name mismatch")
        print("   - Message type incompatibility")

    # Cleanup
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
