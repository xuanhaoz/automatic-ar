"""Camera configuration – mirrors C++ CamConfig class.

Reads calibration files written by OpenCV (FileStorage XML/YAML) with keys:
  image_height, image_width, camera_matrix, distortion_coefficients
"""

import cv2
import math
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

    def to_file(self, path: str) -> bool:
        """Save camera calibration to OpenCV FileStorage YAML file.
        
        Args:
            path: Output file path (should end with .xml, .yml, or .yaml)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            fs = cv2.FileStorage(str(path), cv2.FileStorage_WRITE)
            if not fs.isOpened():
                return False
            
            fs.write('image_width', int(self.image_size[0]))
            fs.write('image_height', int(self.image_size[1]))
            fs.write('camera_matrix', self.cam_mat)
            fs.write('distortion_coefficients', self.dist_coeffs)
            fs.release()
            return True
        except Exception as e:
            print(f'Error saving calibration to {path}: {e}')
            return False

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

    @classmethod
    def create_default_config(cls,
                             image_size: Tuple[int, int],
                             fov_degrees: float = 60.0) -> 'CamConfig':
        """Create a default camera config with estimated parameters.
        
        Args:
            image_size:    (width, height) in pixels
            fov_degrees:   Field of view in degrees (default 60°)
        
        Returns:
            CamConfig with estimated focal length and zero distortion
        """
        # Convert FOV to focal length
        # FOV = 2 * arctan(width / (2 * f))
        # f = width / (2 * tan(FOV/2))
        fov_rad = math.radians(fov_degrees)
        focal_length = image_size[0] / (2.0 * math.tan(fov_rad / 2.0))
        
        K = np.array([
            [focal_length, 0.0, image_size[0] / 2.0],
            [0.0, focal_length, image_size[1] / 2.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        
        D = np.zeros(5, dtype=np.float64)
        
        return cls(K, D, image_size)

    @classmethod
    def create_default_configs(cls,
                              folder_path: str,
                              num_cameras: int,
                              fov_degrees: float = 60.0) -> List['CamConfig']:
        """Create default camera configs for all cameras in a folder.
        
        Infers image size from first detected frame or uses defaults.
        
        Args:
            folder_path:   Path to dataset folder
            num_cameras:   Number of cameras expected
            fov_degrees:   Field of view in degrees
        
        Returns:
            List of default CamConfig objects
        """
        # Try to infer image size from first image
        image_size = (1280, 720)  # Default fallback
        
        from pathlib import Path
        folder = Path(folder_path)
        
        # Try to find first image in any camera folder
        for cam_idx in range(num_cameras):
            cam_dir = folder / str(cam_idx)
            if cam_dir.exists():
                # Look for any image file (jpg or png)
                img_files = list(cam_dir.glob('*.jpg')) + list(cam_dir.glob('*.png'))
                for img_file in img_files:
                    try:
                        import cv2
                        img = cv2.imread(str(img_file))
                        if img is not None:
                            h, w = img.shape[:2]
                            image_size = (w, h)
                            break
                    except Exception:
                        pass
            if image_size != (1280, 720):
                break
        
        # Create default config for each camera
        configs = []
        for _ in range(num_cameras):
            config = cls.create_default_config(image_size, fov_degrees)
            configs.append(config)
        
        return configs
