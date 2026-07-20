"""MultiCamMapper – mirrors C++ MultiCamMapper.

Responsibilities:
  - Bundle-adjustment style optimisation of camera poses, marker poses, object
    poses and (optionally) camera intrinsics using scipy Levenberg-Marquardt.
  - Single-frame object-pose tracking (6 DOF only, real-time).
  - Binary solution file read / write (compatible with C++ binary format).
  - YAML text solution file write.
  - Overlay (project marker outlines onto camera images).

Coordinate conventions (same as C++):
  transforms_to_root_cam[c]    = T_{root_cam ← cam_c}
  transforms_to_local_cam[c]   = T_{cam_c ← root_cam}   (= inverse of above)
  transforms_to_root_marker[m] = T_{root_marker ← marker_m}
  object_to_global[f]          = T_{root_cam ← root_marker}  (per frame f)

Projection of marker-m corner point p (in marker-m frame) onto camera-c image:
  T_final = transforms_to_local_cam[c] @ object_to_global[f] @ transforms_to_root_marker[m]
  p_cam   = T_final[:3,:] @ [p; 1]
  proj    = K_c @ p_cam   (undistorted pinhole, distortion already removed)
  pixel   = proj[:2] / proj[2]
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import yaml
from scipy.optimize import least_squares

from .aruco_serdes import (
    deserialize_marker, serialize_marker,
    read_size_t, write_size_t,
    read_int, write_int,
    read_double, write_double,
    read_bool, write_bool,
)
from .cam_config import CamConfig
from .initializer import Initializer

T4 = np.ndarray                      # 4×4 float64
Marker = Tuple[int, np.ndarray]          # (marker_id, corners_4×2)


# ---------------------------------------------------------------------------
# Optimisation configuration
# ---------------------------------------------------------------------------

@dataclass
class OptimConfig:
    optimize_cam_poses:     bool = True
    optimize_marker_poses:  bool = True
    optimize_object_poses:  bool = True
    optimize_cam_intrinsics: bool = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rvec_tvec_to_mat(rvec: np.ndarray, tvec: np.ndarray) -> T4:
    R, _ = cv2.Rodrigues(rvec.reshape(3))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T


def _mat_to_rvec_tvec(T: T4) -> Tuple[np.ndarray, np.ndarray]:
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    return rvec.flatten(), T[:3, 3].copy()


def _huber_weight(sq_err: np.ndarray, delta: float) -> np.ndarray:
    """Element-wise Huber weight √(ρ(e²)/e²)."""
    w = np.ones_like(sq_err)
    mask = sq_err > 0
    outlier = mask & (sq_err > delta ** 2)
    w[outlier] = delta / np.sqrt(sq_err[outlier])
    return w


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MultiCamMapper:
    """Multi-camera pose optimiser and real-time object tracker."""

    # ----------------------------------------------------------------
    # Construction
    # ----------------------------------------------------------------

    def __init__(self) -> None:
        self.config = OptimConfig()
        self.root_cam:    int = 0
        self.root_marker: int = 0
        self.num_cameras: int = 0
        self.num_markers: int = 0
        self.num_frames:  int = 0
        self.marker_size: float = 0.0
        self.with_huber:  bool = False
        self.huber_delta: float = 10.0

        # {cam_id → 4×4}, {marker_id → 4×4}, {frame_id → 4×4}
        self.mat_arrays: Dict[str, Dict[int, np.ndarray]] = {
            'transforms_to_root_cam':    {},
            'transforms_to_local_cam':   {},
            'transforms_to_root_marker': {},
            'object_to_global':          {},
            'cam_mats':                  {},
            'dist_coeffs':               {},
        }

        # {frame_id: {cam_id: [(marker_id, corners_4×2)]}}
        self.frame_cam_markers: Dict[int, Dict[int, List[Marker]]] = {}

        self.image_sizes: List[Tuple[int, int]] = []   # (w, h) per cam index
        self.cam_configs: List[CamConfig] = []

        # pre-computed 3-D marker points
        self._marker_points_3d:      Optional[np.ndarray] = None  # (4, 3)
        # (4, 4) homogeneous cols
        self._marker_pts_3d_hom:     Optional[np.ndarray] = None

        self.num_point_xys: int = 0   # total residual count

        # deterministic ordering caches
        self._cam_ids_sorted:    List[int] = []
        self._marker_ids_sorted: List[int] = []
        self._frame_ids_sorted:  List[int] = []

    # ----------------------------------------------------------------
    # Factory from Initializer
    # ----------------------------------------------------------------

    @classmethod
    def from_initializer(cls, initializer: Initializer) -> 'MultiCamMapper':
        mcm = cls()
        mcm.init(
            root_cam=initializer.get_root_cam(),
            transforms_to_root_cam=initializer.get_transforms_to_root_cam(),
            root_marker=initializer.get_root_marker(),
            transforms_to_root_marker=initializer.get_transforms_to_root_marker(),
            object_poses=initializer.get_object_transforms(),
            frame_cam_markers=initializer.get_frame_cam_markers(),
            marker_size=initializer.get_marker_size(),
            cam_configs=initializer.get_cam_configs(),
        )
        return mcm

    # ----------------------------------------------------------------
    # Configuration setters
    # ----------------------------------------------------------------

    def set_optmize_flag_cam_poses(
        self, flag: bool) -> None: self.config.optimize_cam_poses = flag

    def set_optmize_flag_marker_poses(
        self, flag: bool) -> None: self.config.optimize_marker_poses = flag

    def set_optmize_flag_object_poses(
        self, flag: bool) -> None: self.config.optimize_object_poses = flag
    def set_optmize_flag_cam_intrinsics(
        self, flag: bool) -> None: self.config.optimize_cam_intrinsics = flag

    def set_with_huber(self, flag: bool) -> None: self.with_huber = flag

    def get_root_cam(self) -> int: return self.root_cam
    def get_root_marker(self) -> int: return self.root_marker
    def get_marker_size(self) -> float: return self.marker_size

    def get_image_sizes(self) -> List[Tuple[int, int]]:
        return list(self.image_sizes)

    def get_mat_arrays(self) -> Dict[str, Dict[int, np.ndarray]]:
        return {k: dict(v) for k, v in self.mat_arrays.items()}

    # ----------------------------------------------------------------
    # Initialisation
    # ----------------------------------------------------------------

    def init(self,
             root_cam: int,
             transforms_to_root_cam: Dict[int, T4],
             root_marker: int,
             transforms_to_root_marker: Dict[int, T4],
             object_poses: Dict[int, T4],
             frame_cam_markers: Dict[int, Dict[int, List[Marker]]],
             marker_size: float,
             cam_configs: List[CamConfig]) -> None:
        """Full initialisation from pose data."""
        self.root_cam = root_cam
        self.root_marker = root_marker
        self.marker_size = marker_size
        self.cam_configs = cam_configs

        ma = self.mat_arrays
        ma['transforms_to_root_cam'] = {k: v.astype(
            np.float64) for k, v in transforms_to_root_cam.items()}
        ma['transforms_to_root_marker'] = {k: v.astype(
            np.float64) for k, v in transforms_to_root_marker.items()}
        ma['object_to_global'] = {k: v.astype(
            np.float64) for k, v in object_poses.items()}

        # local cam = inverse of root cam transform
        ma['transforms_to_local_cam'] = {
            cid: np.linalg.inv(T) for cid, T in ma['transforms_to_root_cam'].items()
        }

        # camera intrinsics indexed by cam_id
        cam_ids_sorted = sorted(ma['transforms_to_root_cam'].keys())
        for idx, cam_id in enumerate(cam_ids_sorted):
            ma['cam_mats'][cam_id] = cam_configs[cam_id].cam_mat.copy()
            ma['dist_coeffs'][cam_id] = cam_configs[cam_id].dist_coeffs.copy()
            if idx < len(cam_configs):
                h, w = cam_configs[cam_id].image_size[1], cam_configs[cam_id].image_size[0]
                self.image_sizes.append((cam_configs[cam_id].image_size[0],
                                         cam_configs[cam_id].image_size[1]))

        self.frame_cam_markers = {
            fid: {cid: list(markers) for cid, markers in cams.items()}
            for fid, cams in frame_cam_markers.items()
        }

        self.num_cameras = len(ma['transforms_to_root_cam'])
        self.num_markers = len(ma['transforms_to_root_marker'])

        self._init_marker_points_3d()
        self._fill_iteration_arrays()
        self._remove_distortions()

        print(f'Initial reprojection error: {self._total_sq_error():.4f}')

    def init_tracking(self,
                      object_poses: Dict[int, T4],
                      frame_cam_markers: Dict[int, Dict[int, List[Marker]]]) -> None:
        """Light re-init for tracking: update only object poses and observations."""
        self.mat_arrays['object_to_global'] = {k: v.astype(np.float64)
                                               for k, v in object_poses.items()}
        self.frame_cam_markers = {
            fid: {cid: list(markers) for cid, markers in cams.items()}
            for fid, cams in frame_cam_markers.items()
        }
        self._fill_iteration_arrays()
        self._remove_distortions()

    # ----------------------------------------------------------------
    # Internal setup helpers
    # ----------------------------------------------------------------

    def _init_marker_points_3d(self) -> None:
        h = self.marker_size / 2.0
        self._marker_points_3d = np.array([
            [-h,  h, 0],
            [h,  h, 0],
            [h, -h, 0],
            [-h, -h, 0],
        ], dtype=np.float64)
        # Homogeneous columns: (4, 4)
        self._marker_pts_3d_hom = np.vstack([
            self._marker_points_3d.T,
            np.ones((1, 4), dtype=np.float64),
        ])

    def _fill_iteration_arrays(self) -> None:
        """Count residuals and remove observations with unknown cameras/markers."""
        self.num_point_xys = 0
        ma = self.mat_arrays

        for fid in list(self.frame_cam_markers.keys()):
            for cid in list(self.frame_cam_markers[fid].keys()):
                if cid not in ma['transforms_to_root_cam']:
                    del self.frame_cam_markers[fid][cid]
                    continue
                filtered = [
                    (mid, c) for mid, c in self.frame_cam_markers[fid][cid]
                    if mid in ma['transforms_to_root_marker']
                ]
                self.frame_cam_markers[fid][cid] = filtered
                self.num_point_xys += 8 * len(filtered)

            # prune empty cameras
            self.frame_cam_markers[fid] = {
                cid: m for cid, m in self.frame_cam_markers[fid].items() if m
            }

        # prune empty frames
        self.frame_cam_markers = {
            fid: cams for fid, cams in self.frame_cam_markers.items() if cams
        }

        self.num_frames = len(self.mat_arrays['object_to_global'])
        self._cam_ids_sorted = sorted(
            self.mat_arrays['transforms_to_root_cam'].keys())
        self._marker_ids_sorted = sorted(
            self.mat_arrays['transforms_to_root_marker'].keys())
        self._frame_ids_sorted = sorted(
            self.mat_arrays['object_to_global'].keys())

    def _remove_distortions(self) -> None:
        """Undistort all stored marker corners in-place.

        After this, ideal pinhole projection (without distortion) is used
        during optimisation, matching the C++ remove_distortions() approach.
        """
        ma = self.mat_arrays
        for fid, cam_markers in self.frame_cam_markers.items():
            for cid, markers in cam_markers.items():
                if not markers:
                    continue
                K = ma['cam_mats'][cid].astype(np.float64)
                D = ma['dist_coeffs'][cid].astype(np.float64)
                # Stack all corners
                all_corners = np.array([c for _, c in markers],
                                       dtype=np.float32).reshape(-1, 1, 2)
                undist = cv2.undistortPoints(all_corners, K, D, R=None, P=K)
                undist = undist.reshape(-1, 4, 2)
                self.frame_cam_markers[fid][cid] = [
                    (markers[i][0], undist[i].astype(np.float32))
                    for i in range(len(markers))
                ]

    # ----------------------------------------------------------------
    # Projection
    # ----------------------------------------------------------------

    def _project_marker(self,
                        ma: Dict[str, Dict[int, np.ndarray]],
                        frame_id: int,
                        marker_id: int,
                        cam_id: int) -> np.ndarray:
        """Project marker corners into cam_id's image plane.  Returns (4, 2)."""
        T_obj = ma['object_to_global'][frame_id]

        # Transform to camera frame
        T_local = ma['transforms_to_local_cam'].get(cam_id)
        if T_local is None:
            T_local = np.eye(4, dtype=np.float64)
        T = T_local @ T_obj

        # Account for non-root markers
        if marker_id != self.root_marker:
            T = T @ ma['transforms_to_root_marker'][marker_id]

        K = ma['cam_mats'][cam_id]
        # Ideal pinhole projection (no distortion – already removed)
        pts_cam = K @ T[:3, :] @ self._marker_pts_3d_hom   # (3, 4)
        pts_2d = pts_cam[:2, :] / pts_cam[2, :]            # (2, 4)
        return pts_2d.T                                      # (4, 2)

    # ----------------------------------------------------------------
    # Residual evaluation
    # ----------------------------------------------------------------

    def _eval_residuals(self,
                        ma: Dict[str, Dict[int, np.ndarray]]) -> np.ndarray:
        """Compute full reprojection-error vector (length = num_point_xys)."""
        out = np.empty(self.num_point_xys, dtype=np.float64)
        idx = 0
        for fid, cam_markers in self.frame_cam_markers.items():
            for cid, markers in cam_markers.items():
                for mid, observed in markers:
                    projected = self._project_marker(
                        ma, fid, mid, cid)  # (4,2)
                    diff = observed.astype(np.float64) - \
                        projected        # (4,2)
                    if self.with_huber:
                        sq = np.sum(diff ** 2, axis=1)          # (4,)
                        w = _huber_weight(sq, self.huber_delta)  # (4,)
                        diff *= w[:, np.newaxis]
                    out[idx:idx + 8] = diff.flatten()
                    idx += 8
        return out

    def _total_sq_error(self) -> float:
        r = self._eval_residuals(self.mat_arrays)
        return float(r @ r)

    # ----------------------------------------------------------------
    # Parameter vector encoding / decoding
    # ----------------------------------------------------------------

    def _mats_to_vec(self, ma: Dict[str, Dict[int, np.ndarray]],
                     cfg: OptimConfig) -> np.ndarray:
        """Flatten optimisation parameters into a 1-D vector."""
        parts: List[np.ndarray] = []

        if cfg.optimize_cam_poses:
            for cid in self._cam_ids_sorted:
                if cid == self.root_cam:
                    continue
                rv, tv = _mat_to_rvec_tvec(ma['transforms_to_root_cam'][cid])
                parts.append(rv)
                parts.append(tv)

        if cfg.optimize_marker_poses:
            for mid in self._marker_ids_sorted:
                if mid == self.root_marker:
                    continue
                rv, tv = _mat_to_rvec_tvec(
                    ma['transforms_to_root_marker'][mid])
                parts.append(rv)
                parts.append(tv)

        if cfg.optimize_object_poses:
            for fid in self._frame_ids_sorted:
                rv, tv = _mat_to_rvec_tvec(ma['object_to_global'][fid])
                parts.append(rv)
                parts.append(tv)

        if cfg.optimize_cam_intrinsics:
            for cid in self._cam_ids_sorted:
                K = ma['cam_mats'][cid]
                D = ma['dist_coeffs'][cid]
                parts.append(
                    np.array([K[0, 0], K[0, 2], K[1, 1], K[1, 2]], dtype=np.float64))
                parts.append(D.flatten()[:5].astype(np.float64))

        return np.concatenate(parts) if parts else np.empty(0)

    def _vec_to_mats(self, vec: np.ndarray,
                     cfg: OptimConfig) -> Dict[str, Dict[int, np.ndarray]]:
        """Decode parameter vector back to a dict-of-dicts (shallow copy + overwrite)."""
        ma: Dict[str, Dict[int, np.ndarray]] = {
            k: dict(v) for k, v in self.mat_arrays.items()
        }
        idx = 0

        if cfg.optimize_cam_poses:
            for cid in self._cam_ids_sorted:
                if cid == self.root_cam:
                    ma['transforms_to_root_cam'][cid] = np.eye(4)
                    ma['transforms_to_local_cam'][cid] = np.eye(4)
                    continue
                T = _rvec_tvec_to_mat(vec[idx:idx + 3], vec[idx + 3:idx + 6])
                idx += 6
                ma['transforms_to_root_cam'][cid] = T
                ma['transforms_to_local_cam'][cid] = np.linalg.inv(T)

        if cfg.optimize_marker_poses:
            for mid in self._marker_ids_sorted:
                if mid == self.root_marker:
                    ma['transforms_to_root_marker'][mid] = np.eye(4)
                    continue
                T = _rvec_tvec_to_mat(vec[idx:idx + 3], vec[idx + 3:idx + 6])
                idx += 6
                ma['transforms_to_root_marker'][mid] = T

        if cfg.optimize_object_poses:
            for fid in self._frame_ids_sorted:
                T = _rvec_tvec_to_mat(vec[idx:idx + 3], vec[idx + 3:idx + 6])
                idx += 6
                ma['object_to_global'][fid] = T

        if cfg.optimize_cam_intrinsics:
            for cid in self._cam_ids_sorted:
                fx, cx, fy, cy = vec[idx:idx + 4]
                idx += 4
                d = vec[idx:idx + 5]
                idx += 5
                K = np.eye(3, dtype=np.float64)
                K[0, 0] = fx
                K[0, 2] = cx
                K[1, 1] = fy
                K[1, 2] = cy
                ma['cam_mats'][cid] = K
                ma['dist_coeffs'][cid] = d.astype(np.float64)

        return ma

    # ----------------------------------------------------------------
    # Solve / Track
    # ----------------------------------------------------------------

    def solve(self) -> None:
        """Full bundle-adjustment optimisation."""
        self.huber_delta = 10.0
        x0 = self._mats_to_vec(self.mat_arrays, self.config)
        if x0.size == 0:
            return

        def residuals(x: np.ndarray) -> np.ndarray:
            ma = self._vec_to_mats(x, self.config)
            return self._eval_residuals(ma)

        print(
            f'Optimising {x0.size} parameters over {self.num_point_xys} residuals …')
        result = least_squares(
            residuals, x0,
            method='lm',
            ftol=1e-4, xtol=1e-8, gtol=1e-8,
            max_nfev=10000,
            verbose=1,
        )
        # Write optimised values back
        self.mat_arrays = self._vec_to_mats(result.x, self.config)

    def track(self) -> None:
        """Optimise only the current-frame object pose (6 DOF)."""
        cfg = OptimConfig(
            optimize_cam_poses=True,
            optimize_marker_poses=False,
            optimize_object_poses=False,
            optimize_cam_intrinsics=True,
        )
        x0 = self._mats_to_vec(self.mat_arrays, cfg)
        if x0.size == 0:
            return

        def residuals(x: np.ndarray) -> np.ndarray:
            ma = self._vec_to_mats(x, cfg)
            return self._eval_residuals(ma)

        result = least_squares(
            residuals, x0,
            method='lm',
            ftol=1e-6, xtol=1e-8,
            max_nfev=200,
            verbose=0,
        )
        self.mat_arrays = self._vec_to_mats(result.x, cfg)

    def track_independent_markers(self) -> None:
        """Track each marker independently with full 6-DOF freedom.

        Creates a separate optimization per marker to avoid underdetermined systems.
        Returns a dict of {marker_id: optimized_transform}.
        """
        self.marker_poses_independent: Dict[int, np.ndarray] = {}

        for marker_id in self._marker_ids_sorted:
            if marker_id == self.root_marker:
                self.marker_poses_independent[marker_id] = np.eye(
                    4, dtype=np.float64)
                continue

            # Create temporary config: optimize only this marker's pose
            cfg = OptimConfig(
                optimize_cam_poses=False,
                optimize_marker_poses=True,
                optimize_object_poses=False,
                optimize_cam_intrinsics=False,
            )

            # Temporarily swap marker to root position for this optimization
            orig_markers = dict(self.mat_arrays['transforms_to_root_marker'])
            orig_object_poses = dict(self.mat_arrays['object_to_global'])

            # Map this marker to be the "object" being optimized
            T_marker_current = orig_markers[marker_id]
            self.mat_arrays['object_to_global'] = {
                fid: T_marker_current.copy() for fid in self._frame_ids_sorted
            }

            x0 = self._mats_to_vec(self.mat_arrays, cfg)
            if x0.size < 6 or self.num_point_xys < x0.size:
                # Underdetermined or no observations
                self.marker_poses_independent[marker_id] = T_marker_current
                continue

            def residuals(x: np.ndarray) -> np.ndarray:
                ma = self._vec_to_mats(x, cfg)
                return self._eval_residuals(ma)

            try:
                result = least_squares(
                    residuals, x0,
                    method='lm',
                    ftol=1e-6, xtol=1e-8,
                    max_nfev=200,
                    verbose=0,
                )
                ma_optimized = self._vec_to_mats(result.x, cfg)
                # Extract optimized marker pose
                T_optimized = ma_optimized['object_to_global'][self._frame_ids_sorted[0]]
                self.marker_poses_independent[marker_id] = T_optimized
            except Exception as e:
                # If optimization fails, keep original pose
                print(f'Warning: marker {marker_id} optimization failed: {e}')
                self.marker_poses_independent[marker_id] = T_marker_current

            # Restore original state
            self.mat_arrays['transforms_to_root_marker'] = orig_markers
            self.mat_arrays['object_to_global'] = orig_object_poses

    def track_with_fallback(self, mode: str = 'rigid') -> None:
        """Track with adaptive fallback between rigid and independent modes.

        Args:
            mode: 'rigid' (default), 'independent', or 'adaptive'
                - 'rigid': Only optimize object pose (fast, markers constrained)
                - 'independent': Each marker 6 DOF (slow, complete freedom)
                - 'adaptive': Auto-switch based on system determinedness
        """
        if mode == 'rigid':
            self.track()
        elif mode == 'independent':
            self.track_independent_markers()
        elif mode == 'adaptive':
            # Check if system is well-determined for joint marker optimization
            cfg_marker = OptimConfig(
                optimize_cam_poses=False,
                optimize_marker_poses=True,
                optimize_object_poses=False,
                optimize_cam_intrinsics=False,
            )
            x_marker = self._mats_to_vec(self.mat_arrays, cfg_marker)

            if x_marker.size > 0 and self.num_point_xys >= x_marker.size:
                # Well-determined: try marker optimization
                try:
                    def residuals(x: np.ndarray) -> np.ndarray:
                        ma = self._vec_to_mats(x, cfg_marker)
                        return self._eval_residuals(ma)

                    result = least_squares(
                        residuals, x_marker,
                        method='lm',
                        ftol=1e-6, xtol=1e-8,
                        max_nfev=200,
                        verbose=0,
                    )
                    self.mat_arrays = self._vec_to_mats(result.x, cfg_marker)
                except ValueError:
                    # Fallback to independent if fails
                    self.track_independent_markers()
            else:
                # Underdetermined: use independent
                self.track_independent_markers()

    # ----------------------------------------------------------------
    # Overlay
    # ----------------------------------------------------------------

    def overlay_markers(self, img: np.ndarray, frame_id: int,
                        camera_index: int) -> np.ndarray:
        """Project marker outlines (green) onto *img* for *camera_index*.

        Uses the stored (possibly optimised) solution.
        frame_id is the sequential frame index (0-based).
        camera_index is the sequential camera index (0-based).
        """
        camera_id = self._cam_ids_sorted[camera_index]
        ma = self.mat_arrays

        if img is None:
            w, h = self.image_sizes[camera_index]
            img = np.full((h, w, 3), 255, dtype=np.uint8)

        if frame_id not in ma['object_to_global']:
            return img

        T_obj = ma['object_to_global'][frame_id]
        T_local = ma['transforms_to_local_cam'][camera_id]
        K = ma['cam_mats'][camera_id]
        D = ma['dist_coeffs'][camera_id]

        for mid, T_m2root in ma['transforms_to_root_marker'].items():
            T = T_local @ T_obj @ T_m2root
            # Only draw if marker faces the camera (z-column check)
            if T[2, 2] < 0:
                rvec, _ = cv2.Rodrigues(T[:3, :3])
                tvec = T[:3, 3]
                pts_2d, _ = cv2.projectPoints(
                    self._marker_points_3d.astype(np.float32),
                    rvec, tvec, K.astype(np.float32), D.astype(np.float32),
                )
                pts = pts_2d.reshape(4, 2).astype(np.int32)
                for j in range(4):
                    cv2.line(img, tuple(pts[j]), tuple(pts[(j + 1) % 4]),
                             (0, 255, 0), 2)
        return img

    # ----------------------------------------------------------------
    # Text (YAML) solution file
    # ----------------------------------------------------------------

    def write_text_solution_file(self, path: str) -> None:
        """Write a human-readable YAML summary of the current solution."""
        ma = self.mat_arrays

        def mat_to_list(T: np.ndarray) -> list:
            return T.tolist()

        data = {
            'marker_size': float(self.marker_size),
            'transforms_to_root_cam': [
                {'cam_id': cid, 'transform': mat_to_list(
                    ma['transforms_to_root_cam'][cid])}
                for cid in self._cam_ids_sorted
            ],
            'transforms_to_root_marker': [
                {'marker_id': mid, 'transform': mat_to_list(
                    ma['transforms_to_root_marker'][mid])}
                for mid in self._marker_ids_sorted
            ],
            'root_marker_to_root_cam': [
                {'frame_id': fid, 'transform': mat_to_list(
                    ma['object_to_global'][fid])}
                for fid in self._frame_ids_sorted
            ],
        }
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)

    # ----------------------------------------------------------------
    # Binary solution file write / read
    # ----------------------------------------------------------------

    def write_solution_file(self, path: str) -> bool:
        """Write binary .solution file (compatible with C++ format)."""
        try:
            ma = self.mat_arrays
            # Build full io_vec with all parameters (regardless of config)
            full_cfg = OptimConfig()
            io_vec = self._mats_to_vec(ma, full_cfg)

            with open(path, 'wb') as f:
                # -- cameras --
                write_size_t(f, self.num_cameras)
                for cid in self._cam_ids_sorted:
                    write_int(f, cid)
                write_size_t(f, self.root_cam)
                for i in range(self.num_cameras):
                    w, h = self.image_sizes[i] if i < len(
                        self.image_sizes) else (0, 0)
                    f.write(struct.pack('<ii', w, h))

                # -- markers --
                write_size_t(f, self.num_markers)
                for mid in self._marker_ids_sorted:
                    write_int(f, mid)
                write_size_t(f, self.root_marker)

                # -- frames --
                write_double(f, self.marker_size)
                write_size_t(f, self.num_frames)
                for fid in self._frame_ids_sorted:
                    write_int(f, fid)

                # -- parameter vector --
                for v in io_vec:
                    write_double(f, float(v))

                # -- frame_cam_markers --
                self._serialize_frame_cam_markers(f)

                # -- config flags --
                write_bool(f, self.config.optimize_cam_poses)
                write_bool(f, self.config.optimize_marker_poses)
                write_bool(f, self.config.optimize_object_poses)
                write_bool(f, self.config.optimize_cam_intrinsics)
            return True
        except Exception as e:
            print(f'write_solution_file failed: {e}')
            return False

    def read_solution_file(self, path: str) -> bool:
        """Read binary .solution file (compatible with C++ format)."""
        try:
            with open(path, 'rb') as f:
                # -- cameras --
                self.num_cameras = int(read_size_t(f))
                cam_ids = [read_int(f) for _ in range(self.num_cameras)]
                self.root_cam = int(read_size_t(f))

                raw_img_sizes = []
                for _ in range(self.num_cameras):
                    w, h = struct.unpack('<ii', f.read(8))
                    raw_img_sizes.append((w, h))
                self.image_sizes = raw_img_sizes

                # -- markers --
                self.num_markers = int(read_size_t(f))
                marker_ids = [read_int(f) for _ in range(self.num_markers)]
                self.root_marker = int(read_size_t(f))

                # -- frames --
                self.marker_size = read_double(f)
                self._init_marker_points_3d()
                self.num_frames = int(read_size_t(f))
                frame_ids = [read_int(f) for _ in range(self.num_frames)]

                # Update sorted ID caches immediately
                self._cam_ids_sorted = sorted(cam_ids)
                self._marker_ids_sorted = sorted(marker_ids)
                self._frame_ids_sorted = sorted(frame_ids)

                # -- parameter vector (full config) --
                full_cfg = OptimConfig()
                n_params = (
                    (self.num_cameras - 1) * 6
                    + (self.num_markers - 1) * 6
                    + self.num_frames * 6
                    + self.num_cameras * 9
                )
                io_vec = np.array([read_double(f) for _ in range(n_params)])

                # Decode into mat_arrays
                ma = self.mat_arrays
                for cid in self._cam_ids_sorted:
                    ma['transforms_to_root_cam'][cid] = np.eye(4)
                    ma['transforms_to_local_cam'][cid] = np.eye(4)
                for mid in self._marker_ids_sorted:
                    ma['transforms_to_root_marker'][mid] = np.eye(4)
                for fid in self._frame_ids_sorted:
                    ma['object_to_global'][fid] = np.eye(4)
                for cid in self._cam_ids_sorted:
                    ma['cam_mats'][cid] = np.eye(3)
                    ma['dist_coeffs'][cid] = np.zeros(5)

                decoded = self._vec_to_mats(io_vec, full_cfg)
                self.mat_arrays = decoded

                # -- frame_cam_markers --
                self._deserialize_frame_cam_markers(f)

                # -- config flags --
                self.config.optimize_cam_poses = read_bool(f)
                self.config.optimize_marker_poses = read_bool(f)
                self.config.optimize_object_poses = read_bool(f)
                self.config.optimize_cam_intrinsics = read_bool(f)

                self._fill_iteration_arrays()
            return True
        except Exception as e:
            print(f'read_solution_file failed: {e}')
            return False

    # ----------------------------------------------------------------
    # Internal binary serialisation of frame_cam_markers
    # ----------------------------------------------------------------

    def _serialize_frame_cam_markers(self, f) -> None:
        write_size_t(f, len(self.frame_cam_markers))
        for fid in sorted(self.frame_cam_markers.keys()):
            write_int(f, fid)
            cam_markers = self.frame_cam_markers[fid]
            write_size_t(f, len(cam_markers))
            for cid in sorted(cam_markers.keys()):
                write_int(f, cid)
                markers = cam_markers[cid]
                write_size_t(f, len(markers))
                for mid, corners in markers:
                    serialize_marker(f, mid, corners)

    def _deserialize_frame_cam_markers(self, f) -> None:
        self.frame_cam_markers = {}
        n_frames = int(read_size_t(f))
        for fi in range(n_frames):
            fid = read_int(f)
            n_cams = int(read_size_t(f))
            cam_dict: Dict[int, List[Marker]] = {}
            for ci in range(n_cams):
                cid = read_int(f)
                n_markers = int(read_size_t(f))
                markers: List[Marker] = []
                for _ in range(n_markers):
                    result = deserialize_marker(f)
                    if result:
                        markers.append(result)
                cam_dict[cid] = markers
            self.frame_cam_markers[fid] = cam_dict

    # ----------------------------------------------------------------
    # Static utility
    # ----------------------------------------------------------------

    @staticmethod
    def read_subseqs(path: str) -> List[int]:
        with open(path) as f:
            return [int(x) for x in f.read().split()]

    @staticmethod
    def write_detections_file(path: str, detections) -> None:
        from .aruco_serdes import write_detections_file as _wdf
        _wdf(path, detections)
