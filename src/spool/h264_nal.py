"""
H.264 NAL Unit Parsing Module.

Provides minimal NAL parsing to detect SPS(7), PPS(8), and IDR(5) frames
for segment boundary alignment and prepending SPS/PPS to new segments.

The H.264 Annex B format uses start codes (0x000001 or 0x00000001) to
delimit NAL units. Each NAL unit has a 1-byte header where:
- bits 7: forbidden_zero_bit (should be 0)
- bits 6-5: nal_ref_idc (importance)
- bits 4-0: nal_unit_type

Key NAL unit types for our purposes:
- 1: Non-IDR slice (P/B frame)
- 5: IDR slice (I-frame, decoder refresh point)
- 6: SEI (supplemental enhancement information)
- 7: SPS (sequence parameter set)
- 8: PPS (picture parameter set)
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import List, Tuple, Optional


class NALUnitType(IntEnum):
    """H.264 NAL unit types relevant for spooling."""
    NON_IDR = 1      # Non-IDR slice (P/B frame)
    SLICE_A = 2      # Slice data partition A
    SLICE_B = 3      # Slice data partition B
    SLICE_C = 4      # Slice data partition C
    IDR = 5          # IDR slice (I-frame)
    SEI = 6          # Supplemental enhancement information
    SPS = 7          # Sequence parameter set
    PPS = 8          # Picture parameter set
    AUD = 9          # Access unit delimiter
    EOS = 10         # End of sequence
    EOB = 11         # End of bitstream
    FILLER = 12      # Filler data


@dataclass
class NALUnit:
    """Represents a parsed NAL unit."""
    nal_type: NALUnitType
    offset: int       # Byte offset in the data (after start code)
    length: int       # Length of NAL unit data (including header byte)
    start_code_len: int  # Length of start code (3 or 4)


def find_start_codes(data: bytes) -> List[Tuple[int, int]]:
    """
    Find all H.264 start codes in the data.
    
    Searches for both 3-byte (0x000001) and 4-byte (0x00000001) start codes.
    
    Args:
        data: H.264 Annex B encoded data
        
    Returns:
        List of (offset, start_code_length) tuples where offset is the 
        position immediately after the start code (NAL header byte).
    """
    positions = []
    i = 0
    data_len = len(data)
    
    while i < data_len - 2:
        # Check for 3-byte start code
        if data[i] == 0 and data[i + 1] == 0 and data[i + 2] == 1:
            # Check for 4-byte start code
            if i > 0 and data[i - 1] == 0:
                # 4-byte start code (0x00000001)
                positions.append((i + 3, 4))
            else:
                # 3-byte start code (0x000001)
                positions.append((i + 3, 3))
            i += 3
        else:
            i += 1
    
    return positions


def parse_nal_units(data: bytes) -> List[NALUnit]:
    """
    Parse NAL units from H.264 Annex B data.
    
    Args:
        data: H.264 Annex B encoded data
        
    Returns:
        List of NALUnit objects with type, offset, and length
    """
    start_codes = find_start_codes(data)
    nal_units = []
    
    for i, (offset, sc_len) in enumerate(start_codes):
        if offset >= len(data):
            continue
            
        # Get NAL unit type from first byte
        nal_header = data[offset]
        nal_type_value = nal_header & 0x1F  # Lower 5 bits
        
        # Try to convert to known type, or use raw value
        try:
            nal_type = NALUnitType(nal_type_value)
        except ValueError:
            # Unknown NAL type, skip it but preserve as NON_IDR for safety
            nal_type = NALUnitType.NON_IDR
        
        # Calculate length (to next start code or end of data)
        if i + 1 < len(start_codes):
            next_offset = start_codes[i + 1][0] - start_codes[i + 1][1]
            length = next_offset - offset
        else:
            length = len(data) - offset
        
        nal_units.append(NALUnit(
            nal_type=nal_type,
            offset=offset,
            length=length,
            start_code_len=sc_len
        ))
    
    return nal_units


def detect_frame_type(data: bytes) -> Tuple[bool, bool, bool]:
    """
    Detect frame type from H.264 data.
    
    Args:
        data: H.264 Annex B encoded data
        
    Returns:
        Tuple of (has_idr, has_sps, has_pps)
    """
    nal_units = parse_nal_units(data)
    
    has_idr = any(n.nal_type == NALUnitType.IDR for n in nal_units)
    has_sps = any(n.nal_type == NALUnitType.SPS for n in nal_units)
    has_pps = any(n.nal_type == NALUnitType.PPS for n in nal_units)
    
    return has_idr, has_sps, has_pps


def extract_sps_pps(data: bytes) -> Tuple[Optional[bytes], Optional[bytes]]:
    """
    Extract SPS and PPS NAL units from H.264 data.
    
    These are needed to prepend to segment files so that the decoder
    can properly initialize without needing data from previous segments.
    
    Args:
        data: H.264 Annex B encoded data
        
    Returns:
        Tuple of (sps_data, pps_data) where each includes the start code
        and complete NAL unit. Returns None for missing units.
    """
    nal_units = parse_nal_units(data)
    sps_data = None
    pps_data = None
    
    for nal in nal_units:
        # Extract complete NAL unit with start code
        start_pos = nal.offset - nal.start_code_len
        end_pos = nal.offset + nal.length
        nal_data = data[start_pos:end_pos]
        
        if nal.nal_type == NALUnitType.SPS and sps_data is None:
            sps_data = nal_data
        elif nal.nal_type == NALUnitType.PPS and pps_data is None:
            pps_data = nal_data
        
        # Found both, can return early
        if sps_data is not None and pps_data is not None:
            break
    
    return sps_data, pps_data


def is_idr_frame(data: bytes) -> bool:
    """
    Check if the H.264 data contains an IDR (keyframe).
    
    IDR frames are ideal segment boundaries because they:
    - Don't reference previous frames
    - Allow decoder to start fresh
    - Are required for random access
    
    Args:
        data: H.264 Annex B encoded data
        
    Returns:
        True if data contains an IDR slice
    """
    has_idr, _, _ = detect_frame_type(data)
    return has_idr
