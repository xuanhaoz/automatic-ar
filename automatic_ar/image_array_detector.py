"""Multi-camera ArUco marker detector – mirrors C++ ImageArrayDetector.

Handles both the legacy OpenCV aruco API (< 4.7) and the new ArucoDetector
API (>= 4.7) transparently.

Dictionary name mapping (C++ string → cv2.aruco constant):
  The C++ code uses the embedded ArUco 3.0 library with 'ARUCO_MIP_36h12'.
  We map this to the closest available OpenCV dict, with a fallback chain.
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple

# (marker_id, corners_4×2)
MarkerList = List[Tuple[int, np.ndarray]]

# -------------------------------------------------------------------
# Dictionary resolution
# -------------------------------------------------------------------

_DICT_NAME_MAP = {
    'ARUCO_MIP_36H12':   ('DICT_ARUCO_MIP_36H12', 'DICT_6X6_250'),
    'ARUCO_MIP_36h12':   ('DICT_ARUCO_MIP_36H12', 'DICT_6X6_250'),
    'ARUCO':             ('DICT_ARUCO_ORIGINAL',),
    'ARUCO_ORIGINAL':    ('DICT_ARUCO_ORIGINAL',),
    '4X4_50':            ('DICT_4X4_50',),
    '4X4_100':           ('DICT_4X4_100',),
    '4X4_250':           ('DICT_4X4_250',),
    '5X5_50':            ('DICT_5X5_50',),
    '5X5_100':           ('DICT_5X5_100',),
    '5X5_250':           ('DICT_5X5_250',),
    '6X6_50':            ('DICT_6X6_50',),
    '6X6_100':           ('DICT_6X6_100',),
    '6X6_250':           ('DICT_6X6_250',),
    '6X6_1000':          ('DICT_6X6_1000',),
    '7X7_50':            ('DICT_7X7_50',),
    '7X7_100':           ('DICT_7X7_100',),
    '7X7_250':           ('DICT_7X7_250',),
}


def _resolve_dict_id(name: str) -> int:
    """Resolve a dictionary name string to a cv2.aruco constant."""
    candidates = _DICT_NAME_MAP.get(
        name, _DICT_NAME_MAP.get(name.upper(), ('DICT_6X6_250',)))
    for attr in candidates:
        val = getattr(cv2.aruco, attr, None)
        if val is not None:
            return val
    # Absolute fallback
    return getattr(cv2.aruco, 'DICT_6X6_250', 10)


def get_aruco_dictionary(name: str = 'ARUCO_DICT_6X6_100') -> cv2.aruco_Dictionary:
    """Return an aruco Dictionary object by name string."""
    dict_id = _resolve_dict_id(name)
    try:
        return cv2.aruco.getPredefinedDictionary(dict_id)
    except AttributeError:
        return cv2.aruco.Dictionary_get(dict_id)  # type: ignore[attr-defined]


# -------------------------------------------------------------------
# Single-image detector wrapper
# -------------------------------------------------------------------

class _SingleDetector:
    """Wraps the aruco detection call for one camera."""

    def __init__(self, aruco_dict) -> None:
        self._dict = aruco_dict
        # Try new API first
        try:
            params = cv2.aruco.DetectorParameters()
            self._detector = cv2.aruco.ArucoDetector(aruco_dict, params)
            self._use_new_api = True
        except AttributeError:
            params = cv2.aruco.DetectorParameters_create()  # type: ignore
            self._params = params
            self._use_new_api = False

    def detect(self, img: np.ndarray) -> MarkerList:
        """Detect markers and return list of (id, corners_4x2)."""
        if self._use_new_api:
            corners, ids, _ = self._detector.detectMarkers(img)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(  # type: ignore
                img, self._dict, parameters=self._params
            )
        if ids is None:
            return []
        return [
            (int(ids[i][0]), corners[i].reshape(4, 2))
            for i in range(len(ids))
        ]


# -------------------------------------------------------------------
# Multi-camera detector
# -------------------------------------------------------------------

class ImageArrayDetector:
    """Detect ArUco markers across an array of cameras simultaneously.

    Mirrors C++ ImageArrayDetector with the same min-detections filter:
    a marker is kept only if it appears in at least *min_detections*
    different cameras.
    """

    def __init__(self,
                 num_cams: int,
                 dictionary_name: str = 'ARUCO_DICT_6X6_100') -> None:
        aruco_dict = get_aruco_dictionary(dictionary_name)
        self._detectors = [_SingleDetector(aruco_dict)
                           for _ in range(num_cams)]
        self.num_cams = num_cams

    def detect_markers(self,
                       images: List[Optional[np.ndarray]],
                       min_detections: int = 1) -> List[MarkerList]:
        """Detect markers in all cameras and apply the visibility filter.

        Args:
            images:          one image per camera (None entries are skipped)
            min_detections:  minimum number of cameras that must see a marker

        Returns:
            per-camera lists of (marker_id, corners_4x2) – filtered.
        """
        cam_markers: List[MarkerList] = []
        for cam, img in enumerate(images):
            if img is None:
                cam_markers.append([])
            else:
                cam_markers.append(self._detectors[cam].detect(img))

        # Count how many cameras see each marker ID
        from collections import Counter
        counts: Counter = Counter()
        for markers in cam_markers:
            for mid, _ in markers:
                counts[mid] += 1

        # Filter
        return [
            [(mid, c) for mid, c in markers if counts[mid] >= min_detections]
            for markers in cam_markers
        ]
