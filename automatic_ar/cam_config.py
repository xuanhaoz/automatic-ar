"""Camera configuration – mirrors C++ CamConfig class.

Reads calibration files written by OpenCV (FileStorage XML/YAML) with keys:
  image_height, image_width, camera_matrix, distortion_coefficients
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple


class CamConfig:
    """Camera intrinsic parameters and image size."""

    def __init__(self,
                 cam_mat: np.ndarray,
                 dist_coeffs: np.ndarray,
                 image_size: Tuple[int, int]) -> None:
        """
        Args:
            cam_mat:     3×3 float64 camera matrix K
            dist_coeffs: (5,) float64 distortion coefficients
            image_size:  (width, height)
        """
        if cam_mat.shape != (3, 3):
            raise ValueError('cam_mat must be 3×3')
        self.cam_mat    = cam_mat.astype(np.float64)
        # Always store exactly 5 coefficients (pad with zeros if needed)
        d = dist_coeffs.flatten().astype(np.float64)
        self.dist_coeffs = np.zeros(5, dtype=np.float64)
        self.dist_coeffs[:min(len(d), 5)] = d[:5]
        self.image_size = tuple(image_size)   # (width, height)

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_file(cls, path: str) -> Optional['CamConfig']:
        """Read calibration from OpenCV FileStorage file.

        Returns None if the file cannot be opened or is missing required keys.
        """
        fs = cv2.FileStorage(str(path), cv2.FileStorage_READ)
        if not fs.isOpened():
            return None

        h_node = fs.getNode('image_height')
        w_node = fs.getNode('image_width')
        K_node = fs.getNode('camera_matrix')
        D_node = fs.getNode('distortion_coefficients')

        if any(n.empty() for n in (h_node, w_node, K_node, D_node)):
            fs.release()
            return None

        h = int(h_node.real())
        w = int(w_node.real())
        K = K_node.mat().astype(np.float64)
        D = D_node.mat().astype(np.float64)
        fs.release()

        return cls(K, D, (w, h))

    @classmethod
    def read_cam_configs(cls, folder_path: str) -> List['CamConfig']:
        """Scan *folder_path* for numbered sub-directories and load calib files.

        Mirrors C++ CamConfig::read_cam_configs – looks for:
            <folder>/<N>/calib.xml  (or .yml / .yaml)

        Returns configs ordered by camera index (0, 1, 2, …).
        """
        folder = Path(folder_path)
        # Collect numeric sub-directories
        cam_dirs = sorted(
            (d for d in folder.iterdir() if d.is_dir() and d.name.isdigit()),
            key=lambda d: int(d.name),
        )
        configs: List[Optional[CamConfig]] = []
        for cam_dir in cam_dirs:
            config = None
            for ext in ('xml', 'yml', 'yaml'):
                cfg = cls.from_file(cam_dir / f'calib.{ext}')
                if cfg is not None:
                    config = cfg
                    break
            if config is not None:
                configs.append(config)
        return configs
