"""
Container QR Tracking Module for Conveyor Bread Bag Counter System.

This module provides independent container tracking functionality via QR codes.
It monitors a dedicated RTSP camera to track containers moving through the sale point (صالة).

Key Components:
- QR Code Detection: Uses OpenCV QRCodeDetector to read container IDs (1-5)
- Direction Tracking: Counts bottom→top as positive (filled containers leaving)
- Snapshot Buffer: 5-second ring buffer for pre/post event snapshots
- Database Integration: Logs all events with direction and QR code value

ROS2 Topics:
- /rtsp_image_container: H.264 NAL units from RTSP client
- /nv12_images_container: Decoded NV12 frames for processing
"""

from src.container.qr.QRCodeDetector import QRCodeDetector
from src.container.tracking.ContainerTracker import ContainerTracker
from src.container.snapshot.RingBufferSnapshotter import RingBufferSnapshotter

__all__ = [
    'QRCodeDetector',
    'ContainerTracker',
    'RingBufferSnapshotter',
]
