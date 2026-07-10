"""Binary serialization / deserialization of ArUco markers.

Binary format (matches C++ ArucoSerdes):
  Each marker:
    id      : int32  (4 bytes, signed)
    corners : 4 × (float32 x, float32 y)  = 32 bytes
  Total per marker: 36 bytes

Detection file header:
  num_cams : uint64  (size_t, 8 bytes)
  Then for each frame (until EOF):
    for each camera:
      num_markers : uint64 (8 bytes)
      for each marker: 36-byte marker record
"""

import struct
import numpy as np
from typing import List, Tuple, Optional, BinaryIO

# Struct formats (little-endian, matching x86-64 Linux ABI)
_FMT_SIZE_T = '<Q'   # size_t  – 8 bytes unsigned
_FMT_INT    = '<i'   # int     – 4 bytes signed
_FMT_FLOAT  = '<f'   # float   – 4 bytes
_FMT_DOUBLE = '<d'   # double  – 8 bytes
_FMT_BOOL   = '<?'   # bool    – 1 byte

_SIZE_T_SIZE = struct.calcsize(_FMT_SIZE_T)
_INT_SIZE    = struct.calcsize(_FMT_INT)
_FLOAT_SIZE  = struct.calcsize(_FMT_FLOAT)


def write_size_t(f: BinaryIO, value: int) -> None:
    f.write(struct.pack(_FMT_SIZE_T, value))


def read_size_t(f: BinaryIO) -> Optional[int]:
    data = f.read(_SIZE_T_SIZE)
    if len(data) < _SIZE_T_SIZE:
        return None
    return struct.unpack(_FMT_SIZE_T, data)[0]


def write_int(f: BinaryIO, value: int) -> None:
    f.write(struct.pack(_FMT_INT, value))


def read_int(f: BinaryIO) -> Optional[int]:
    data = f.read(_INT_SIZE)
    if len(data) < _INT_SIZE:
        return None
    return struct.unpack(_FMT_INT, data)[0]


def write_double(f: BinaryIO, value: float) -> None:
    f.write(struct.pack(_FMT_DOUBLE, value))


def read_double(f: BinaryIO) -> Optional[float]:
    data = f.read(8)
    if len(data) < 8:
        return None
    return struct.unpack(_FMT_DOUBLE, data)[0]


def write_bool(f: BinaryIO, value: bool) -> None:
    f.write(struct.pack(_FMT_BOOL, value))


def read_bool(f: BinaryIO) -> Optional[bool]:
    data = f.read(1)
    if not data:
        return None
    return struct.unpack(_FMT_BOOL, data)[0]


# ---------------------------------------------------------------------------
# Marker-level I/O
# ---------------------------------------------------------------------------

def serialize_marker(f: BinaryIO, marker_id: int, corners: np.ndarray) -> None:
    """Write one marker (id + 4 float32 corners) to binary stream."""
    write_int(f, marker_id)
    for x, y in corners:
        f.write(struct.pack('<ff', float(x), float(y)))


def deserialize_marker(f: BinaryIO) -> Optional[Tuple[int, np.ndarray]]:
    """Read one marker from binary stream.

    Returns (marker_id, corners_4x2) or None on EOF.
    """
    raw = f.read(_INT_SIZE)
    if len(raw) < _INT_SIZE:
        return None
    marker_id = struct.unpack(_FMT_INT, raw)[0]
    corners = []
    for _ in range(4):
        raw = f.read(8)
        if len(raw) < 8:
            return None
        x, y = struct.unpack('<ff', raw)
        corners.append([x, y])
    return marker_id, np.array(corners, dtype=np.float32)


# ---------------------------------------------------------------------------
# Detection-file I/O  (aruco.detections)
# ---------------------------------------------------------------------------

DetectionFrame = List[List[Tuple[int, np.ndarray]]]   # [cam_idx][marker] = (id, 4×2)
Detections     = List[DetectionFrame]


def write_detections_file(path: str, detections: Detections) -> None:
    """Write aruco.detections binary file.

    Args:
        path:       output file path
        detections: detections[frame][cam] = [(marker_id, corners_4x2), ...]
    """
    if not detections:
        raise ValueError('detections is empty')
    num_cams = len(detections[0])
    with open(path, 'wb') as f:
        write_size_t(f, num_cams)
        for frame in detections:
            for cam_markers in frame:
                write_size_t(f, len(cam_markers))
                for marker_id, corners in cam_markers:
                    serialize_marker(f, marker_id, corners)


def read_detections_file(path: str,
                         subseqs: Optional[List[int]] = None) -> Detections:
    """Read aruco.detections binary file.

    Args:
        path:    binary detections file path
        subseqs: optional list of [start, end, start, end, …] frame indices;
                 frames outside these ranges have their detections cleared.

    Returns:
        detections[frame][cam] = [(marker_id, corners_4x2), ...]
    """
    with open(path, 'rb') as f:
        num_cams_val = read_size_t(f)
        if num_cams_val is None:
            raise IOError(f'Cannot read num_cams from {path}')
        num_cams = int(num_cams_val)

        all_frames: Detections = []
        while True:
            frame_markers: DetectionFrame = []
            end_of_data = False
            for cam in range(num_cams):
                n_val = read_size_t(f)
                if n_val is None:
                    end_of_data = True
                    break
                n = int(n_val)
                cam_list = []
                for _ in range(n):
                    result = deserialize_marker(f)
                    if result is None:
                        end_of_data = True
                        break
                    cam_list.append(result)
                frame_markers.append(cam_list)
                if end_of_data:
                    break
            if end_of_data:
                break
            all_frames.append(frame_markers)

    if subseqs:
        prev_last = -1
        for i in range(0, len(subseqs) - 1, 2):
            first_f = subseqs[i]
            for f in range(prev_last + 1, first_f):
                if f < len(all_frames):
                    for c in range(num_cams):
                        if c < len(all_frames[f]):
                            all_frames[f][c] = []
            prev_last = subseqs[i + 1]

    return all_frames
