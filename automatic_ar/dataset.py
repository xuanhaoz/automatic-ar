"""Dataset loader – mirrors C++ Dataset class.

A dataset folder has one sub-directory per camera (named 0, 1, 2, …).
Each camera directory contains PNG frames named  <frame_number>.png
and optionally a calib.xml / calib.yml calibration file.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Set


class Dataset:
    """Lazy frame loader over a multi-camera image dataset."""

    def __init__(self, folder_path: str) -> None:
        self.folder_path = Path(folder_path)
        self._num_cams: int = self._detect_num_cams()
        # per-camera frame sets and the union of all frames
        self._cam_frame_sets: List[Set[int]] = [set() for _ in range(self._num_cams)]
        self.all_frames: Set[int] = set()
        self._scan_frames()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _detect_num_cams(self) -> int:
        """Return max numeric directory index + 1."""
        indices = [
            int(d.name)
            for d in self.folder_path.iterdir()
            if d.is_dir() and d.name.isdigit()
        ]
        return max(indices) + 1 if indices else 0

    def _scan_frames(self) -> None:
        """Populate per-camera frame sets from PNG filenames."""
        for cam in range(self._num_cams):
            cam_dir = self.folder_path / str(cam)
            if not cam_dir.is_dir():
                continue
            for f in cam_dir.iterdir():
                if f.suffix.lower() == '.png' and f.stem.isdigit():
                    frame_num = int(f.stem)
                    self._cam_frame_sets[cam].add(frame_num)
                    self.all_frames.add(frame_num)

    # ------------------------------------------------------------------
    # Public API  (matches C++ Dataset interface)
    # ------------------------------------------------------------------

    def get_num_cams(self) -> int:
        return self._num_cams

    def get_frame_nums(self) -> List[int]:
        """Return sorted list of all frame numbers present in any camera."""
        return sorted(self.all_frames)

    def get_frame(self, frame_num: int) -> List[Optional[np.ndarray]]:
        """Load one frame from every camera.

        Returns a list of length *num_cams*.  Entries are None where the
        frame is absent for a particular camera.
        """
        frames: List[Optional[np.ndarray]] = []
        for cam in range(self._num_cams):
            if frame_num in self._cam_frame_sets[cam]:
                img_path = self.folder_path / str(cam) / f'{frame_num}.png'
                img = cv2.imread(str(img_path))
                frames.append(img)
            else:
                frames.append(None)
        return frames
