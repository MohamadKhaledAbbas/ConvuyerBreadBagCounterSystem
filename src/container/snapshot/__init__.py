"""
Snapshot components for container tracking.

Provides:
- RingBufferSnapshotter: 5-second ring buffer for pre/post event snapshots
"""

from src.container.snapshot.RingBufferSnapshotter import RingBufferSnapshotter

__all__ = ['RingBufferSnapshotter']
