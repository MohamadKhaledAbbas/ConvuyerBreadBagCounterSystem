#!/usr/bin/env python3
"""
Test subscribing to /rtsp_image_ch_0 with different message types.

Usage:
    python test_message_types.py
"""

import sys
import time

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
except ImportError as e:
    print(f"Error: Missing rclpy: {e}")
    sys.exit(1)

# Try different message types
MESSAGE_TYPES = []

try:
    from img_msgs.msg import H26XFrame as ImgH26XFrame
    MESSAGE_TYPES.append(("img_msgs.msg.H26XFrame", ImgH26XFrame))
except ImportError:
    pass

try:
    from hobot_cv_msgs.msg import H26XFrame as HobotH26XFrame
    MESSAGE_TYPES.append(("hobot_cv_msgs.msg.H26XFrame", HobotH26XFrame))
except ImportError:
    pass

try:
    from sensor_msgs.msg import Image
    MESSAGE_TYPES.append(("sensor_msgs.msg.Image", Image))
except ImportError:
    pass

try:
    from sensor_msgs.msg import CompressedImage
    MESSAGE_TYPES.append(("sensor_msgs.msg.CompressedImage", CompressedImage))
except ImportError:
    pass

if not MESSAGE_TYPES:
    print("ERROR: No ROS2 message types available!")
    print("Available message packages:")
    print("  - img_msgs (contains H26XFrame)")
    print("  - hobot_cv_msgs (contains H26XFrame)")
    print("  - sensor_msgs (contains Image, CompressedImage)")
    sys.exit(1)


class MessageTypeTest(Node):
    """Test node for different message types."""

    def __init__(self):
        super().__init__('msg_type_test')
        self.frame_count = 0

    def test_message_type(self, name, msg_type):
        """Test subscription with specific message type."""
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print(f"{'='*60}")

        self.frame_count = 0

        # Use BEST_EFFORT QoS (most permissive)
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        try:
            subscription = self.create_subscription(
                msg_type,
                '/rtsp_image_ch_0',
                self._callback,
                qos
            )

            print("Waiting 5 seconds for frames...")
            start = time.time()
            while time.time() - start < 5.0:
                rclpy.spin_once(self, timeout_sec=0.1)

            if self.frame_count > 0:
                print(f"✅ SUCCESS: {self.frame_count} frames received with {name}")
                self.destroy_subscription(subscription)
                return True
            else:
                print(f"❌ No frames with {name}")

            self.destroy_subscription(subscription)

        except Exception as e:
            print(f"❌ Error with {name}: {e}")

        return False

    def _callback(self, msg):
        """Frame callback."""
        self.frame_count += 1
        if self.frame_count == 1:
            print(f"First frame! Type: {type(msg).__name__}")
            if hasattr(msg, 'width'):
                print(f"  Resolution: {msg.width}x{msg.height}")
            if hasattr(msg, 'data'):
                print(f"  Data size: {len(msg.data)} bytes")


def main():
    """Run message type tests."""
    print("ROS2 Message Type Compatibility Test")
    print("="*60)
    print(f"Found {len(MESSAGE_TYPES)} message type(s) to test:")
    for name, _ in MESSAGE_TYPES:
        print(f"  - {name}")

    rclpy.init()
    node = MessageTypeTest()

    working_types = []
    for name, msg_type in MESSAGE_TYPES:
        if node.test_message_type(name, msg_type):
            working_types.append(name)

    # Summary
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")

    if working_types:
        print(f"✅ Working message type(s):")
        for name in working_types:
            print(f"   - {name}")
        print(f"\nUpdate spool_recorder_node.py to use:")
        print(f"   from {working_types[0].rsplit('.', 1)[0]} import {working_types[0].split('.')[-1]}")
    else:
        print("❌ No message types worked!")
        print("\nTroubleshooting:")
        print("1. Check if /rtsp_image_ch_0 is actually publishing:")
        print("   ros2 topic hz /rtsp_image_ch_0")
        print("\n2. Check the actual message type being published:")
        print("   ros2 topic type /rtsp_image_ch_0")
        print("\n3. Check topic info:")
        print("   ros2 topic info /rtsp_image_ch_0 -v")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
