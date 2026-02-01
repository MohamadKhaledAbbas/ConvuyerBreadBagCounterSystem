"""
Spool Module for H.264 Frame Recording and Playback.

This module implements a robust disk-based spooling system for H.264 frames:

Flow Architecture:
==================
RTSP Client → SpoolRecorderNode → Disk Segments → SpoolProcessorNode → hobot-codec → NV12

Components:
===========
- h264_nal: H.264 NAL unit parsing (SPS/PPS extraction, IDR detection)
- segment_io: Binary segment file I/O (SegmentWriter, SegmentReader)
- retention: Automatic cleanup of processed segments
- spool_utils: Utilities (state persistence, pacing, logging)
- spool_recorder_node: ROS2 node for recording RTSP to disk
- spool_processor_node: ROS2 node for reading and publishing segments
"""

from src.spool.h264_nal import (
    NALUnitType,
    find_start_codes,
    parse_nal_units,
    detect_frame_type,
    extract_sps_pps,
    is_idr_frame,
    get_nal_unit_name
)

from src.spool.segment_io import (
    FrameRecord,
    SegmentMetadata,
    SegmentWriter,
    SegmentReader,
    validate_segment_file,
    SEGMENT_MAGIC,
    SEGMENT_VERSION
)

from src.spool.retention import (
    RetentionConfig,
    RetentionPolicy
)

from src.spool.spool_utils import (
    format_structured_log,
    ProcessorState,
    save_processor_state,
    load_processor_state,
    calculate_crc32,
    verify_segment_integrity,
    get_segment_info,
    cleanup_tmp_files,
    RateLimiter,
    AdaptivePacer
)

# ROS2 nodes are imported conditionally to avoid errors on non-RDK platforms
from src.utils.platform import is_rdk_platform

if is_rdk_platform():
    from src.spool.spool_recorder_node import (
        RecorderConfig,
        SpoolRecorderNode,
        create_recorder_node
    )
    from src.spool.spool_processor_node import (
        ProcessorConfig,
        PlaybackMode,
        SpoolProcessorNode,
        create_processor_node
    )

__all__ = [
    # H.264 NAL parsing
    'NALUnitType',
    'find_start_codes',
    'parse_nal_units',
    'detect_frame_type',
    'extract_sps_pps',
    'is_idr_frame',
    'get_nal_unit_name',
    # Segment I/O
    'FrameRecord',
    'SegmentMetadata',
    'SegmentWriter',
    'SegmentReader',
    'validate_segment_file',
    'SEGMENT_MAGIC',
    'SEGMENT_VERSION',
    # Retention
    'RetentionConfig',
    'RetentionPolicy',
    # Utilities
    'format_structured_log',
    'ProcessorState',
    'save_processor_state',
    'load_processor_state',
    'calculate_crc32',
    'verify_segment_integrity',
    'get_segment_info',
    'cleanup_tmp_files',
    'RateLimiter',
    'AdaptivePacer',
    # ROS2 nodes (RDK only)
    'RecorderConfig',
    'SpoolRecorderNode',
    'create_recorder_node',
    'ProcessorConfig',
    'PlaybackMode',
    'SpoolProcessorNode',
    'create_processor_node'
]
