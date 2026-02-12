#!/usr/bin/env python3
"""
Check what topics are actually being published and their types.

Usage:
    python check_ros2_topics.py
"""

import subprocess
import sys

def run_command(cmd):
    """Run shell command and return output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        return result.stdout.strip()
    except Exception as e:
        return f"Error: {e}"

print("="*60)
print("ROS2 Topic Investigation")
print("="*60)

# List all topics
print("\n1. All active topics:")
print("-" * 60)
topics = run_command("ros2 topic list")
print(topics)

# Check if /rtsp_image_ch_0 exists
print("\n2. Checking /rtsp_image_ch_0:")
print("-" * 60)
if "/rtsp_image_ch_0" in topics:
    print("✅ Topic exists")

    # Get topic info
    print("\n   Topic info:")
    info = run_command("ros2 topic info /rtsp_image_ch_0 -v")
    print(info)

    # Check publishing rate
    print("\n   Publishing rate:")
    hz = run_command("timeout 5 ros2 topic hz /rtsp_image_ch_0 2>&1 || echo 'No data'")
    print(hz)

    # Check message type
    print("\n   Message type:")
    msg_type = run_command("ros2 topic type /rtsp_image_ch_0")
    print(f"   Type: {msg_type}")

else:
    print("❌ Topic /rtsp_image_ch_0 does NOT exist")
    print("\nPossible reasons:")
    print("  - hobot_rtsp_client is not running")
    print("  - hobot_rtsp_client publishes to a different topic name")
    print("  - Launch file configuration issue")

# Check for similar topic names
print("\n3. Looking for similar RTSP-related topics:")
print("-" * 60)
similar = run_command("ros2 topic list | grep -i rtsp")
if similar:
    print(similar)
else:
    print("No RTSP-related topics found")

# Check for H264/H265/H26X topics
print("\n4. Looking for video frame topics:")
print("-" * 60)
video_topics = run_command("ros2 topic list | grep -E '(image|h26|video|frame|rtsp|spool)'")
if video_topics:
    print(video_topics)
else:
    print("No video-related topics found")

# Check running nodes
print("\n5. Active ROS2 nodes:")
print("-" * 60)
nodes = run_command("ros2 node list")
print(nodes)

# Check if hobot_rtsp_client is running
print("\n6. Checking for hobot_rtsp_client:")
print("-" * 60)
if "rtsp" in nodes.lower():
    print("✅ RTSP client node is running")
else:
    print("❌ RTSP client node NOT found")
    print("   Is the pipeline launcher running?")
    print("   Try: ros2 launch src/ros2/Ros2PipelineLauncher.py")

print("\n" + "="*60)
print("Investigation complete")
print("="*60)
