"""Initializer – mirrors C++ Initializer class.

Responsible for:
1. Reading binary detection files.
2. Solving PnP (IPPE) for every observed marker in every camera.
3. Building relative-transform candidate sets between camera pairs and
   between marker pairs.
4. Selecting the best transform per pair via consensus scoring.
5. Building a Minimum-Spanning-Tree (Prim's) over cameras / markers.
6. Walking the MST to compute transforms-to-root for all cameras and markers.
7. Computing per-frame object poses (root-marker in root-camera frame).
"""

from __future__ import annotations

import math
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from .aruco_serdes import read_detections_file
from .cam_config import CamConfig

# type aliases
T4 = np.ndarray                          # 4×4 float64 transform
Marker = Tuple[int, np.ndarray]              # (marker_id, corners_4×2)
FrameDetections = List[List[Marker]]          # [cam][marker]
Detections = List[FrameDetections]

# Candidate tuple: (T_transform, T1_inv, T2_inv, error_score)
Candidate = Tuple[T4, T4, T4, float]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _rvec_tvec_to_mat(rvec: np.ndarray, tvec: np.ndarray) -> T4:
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T


def _marker_3d_points(marker_size: float) -> np.ndarray:
    """4 corner points of a square marker (marker local frame, z=0)."""
    h = marker_size / 2.0
    return np.array([
        [-h,  h, 0],
        [h,  h, 0],
        [h, -h, 0],
        [-h, -h, 0],
    ], dtype=np.float32)


def _solve_pnp_ippe(marker_size: float,
                    corners: np.ndarray,
                    cam_mat: np.ndarray,
                    dist_coeffs: np.ndarray,
                    threshold: float = 2.0,
                    ) -> List[Tuple[T4, float]]:
    """Solve PnP with IPPE for a square marker.

    Returns up to 2 (T, reprojection_error) tuples, primary solution first.
    Falls back to SOLVEPNP_ITERATIVE if IPPE_SQUARE is unavailable.
    """
    obj_pts = _marker_3d_points(marker_size)
    img_pts = corners.reshape(4, 1, 2).astype(np.float32)

    solutions: List[Tuple[T4, float]] = []

    ippe_flag = getattr(cv2, 'SOLVEPNP_IPPE_SQUARE', None)
    if ippe_flag is not None:
        retval, rvecs, tvecs, errors = cv2.solvePnPGeneric(
            obj_pts, img_pts,
            cam_mat.astype(np.float32),
            dist_coeffs.astype(np.float32),
            flags=ippe_flag,
        )
        if retval >= 1:
            for k in range(retval):
                T = _rvec_tvec_to_mat(rvecs[k], tvecs[k])
                err = float(errors[k]) if errors is not None else 0.0
                solutions.append((T, err))
    else:
        ok, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts,
            cam_mat.astype(np.float32),
            dist_coeffs.astype(np.float32),
        )
        if ok:
            solutions.append((_rvec_tvec_to_mat(rvec, tvec), 0.0))

    if not solutions:
        return []

    # Sort by reprojection error (lowest first)
    solutions.sort(key=lambda x: x[1])

    # Keep secondary only if ambiguity is below threshold
    if len(solutions) >= 2 and solutions[0][1] > 0:
        if solutions[1][1] / solutions[0][1] < threshold:
            return solutions[:2]
        return solutions[:1]
    return solutions[:1]


# ---------------------------------------------------------------------------
# Initializer
# ---------------------------------------------------------------------------

class Initializer:
    """Compute camera and marker poses from multi-camera ArUco detections."""

    def __init__(self,
                 marker_size: float,
                 cam_configs: List[CamConfig],
                 excluded_cams: Optional[Set[int]] = None,
                 detections: Optional[Detections] = None,
                 threshold: float = 2.0,
                 min_detections: int = 2) -> None:
        """
        Args:
            marker_size:    physical marker side length (metres)
            cam_configs:    one CamConfig per camera
            excluded_cams:  camera indices to ignore
            detections:     if provided, run the full initialisation pipeline
            threshold:      IPPE ambiguity ratio threshold for secondary pose
            min_detections: min cameras that must see a marker per frame
        """
        self.marker_size = marker_size
        self.cam_configs = cam_configs
        self.excluded_cams = excluded_cams or set()
        self.threshold = threshold
        self.min_detections = min_detections

        # outputs
        self.cam_ids:    Set[int] = set()
        self.marker_ids: Set[int] = set()
        self.root_cam:    int = 0
        self.root_marker: int = 0
        self.transforms_to_root_cam:    Dict[int, T4] = {}
        self.transforms_to_root_marker: Dict[int, T4] = {}
        self.object_transforms:         Dict[int, T4] = {}

        # frame_cam_markers: {frame_id: {cam_id: [(marker_id, corners_4×2)]}}
        self.frame_cam_markers: Dict[int, Dict[int, List[Marker]]] = {}

        # internal pose tables
        # frame_poses_cam[frame][marker_id][cam_id] = [(T, err)]
        self._frame_poses_cam:    Dict[int, Dict[int,
                                                 Dict[int, List[Tuple[T4, float]]]]] = {}
        # frame_poses_marker[frame][cam_id][marker_id] = [(T, err)]
        self._frame_poses_marker: Dict[int, Dict[int,
                                                 Dict[int, List[Tuple[T4, float]]]]] = {}

        if detections is not None:
            self._detections = detections
            self.obtain_pose_estimations()
            self._init_transforms()

    # ------------------------------------------------------------------
    # Static / class methods
    # ------------------------------------------------------------------

    @staticmethod
    def read_detections_file(path: str,
                             subseqs: Optional[List[int]] = None) -> Detections:
        return read_detections_file(path, subseqs)

    # ------------------------------------------------------------------
    # Setters (used by tracking mode)
    # ------------------------------------------------------------------

    def set_detections(self, detections: Detections) -> None:
        self._detections = detections

    def set_transforms_to_root_cam(self, t: Dict[int, T4]) -> None:
        self.transforms_to_root_cam = t

    def set_transforms_to_root_marker(self, t: Dict[int, T4]) -> None:
        self.transforms_to_root_marker = t

    # ------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------

    def get_marker_ids(self) -> Set[int]: return set(self.marker_ids)
    def get_cam_ids(self) -> Set[int]: return set(self.cam_ids)
    def get_root_cam(self) -> int: return self.root_cam
    def get_root_marker(self) -> int: return self.root_marker
    def get_marker_size(self) -> float: return self.marker_size
    def get_cam_configs(self) -> List[CamConfig]: return self.cam_configs

    def get_transforms_to_root_cam(self) -> Dict[int, T4]:
        return dict(self.transforms_to_root_cam)

    def get_transforms_to_root_marker(self) -> Dict[int, T4]:
        return dict(self.transforms_to_root_marker)

    def get_object_transforms(self) -> Dict[int, T4]:
        return dict(self.object_transforms)

    def get_frame_cam_markers(self) -> Dict[int, Dict[int, List[Marker]]]:
        return self.frame_cam_markers

    # ------------------------------------------------------------------
    # Phase 1 – PnP per observation
    # ------------------------------------------------------------------

    def obtain_pose_estimations(self) -> None:
        """Solve PnP for every marker-camera observation in every frame."""
        self.cam_ids.clear()
        self.marker_ids.clear()
        self.frame_cam_markers.clear()
        self._frame_poses_cam.clear()
        self._frame_poses_marker.clear()

        for frame_num, frame_data in enumerate(self._detections):
            # Count total detections in this frame (excluding excluded cams)
            total = sum(
                len(frame_data[cam])
                for cam in range(len(frame_data))
                if cam not in self.excluded_cams
            )
            if total < self.min_detections:
                continue

            poses_cam:    Dict[int, Dict[int, List[Tuple[T4, float]]]] = {}
            poses_marker: Dict[int, Dict[int, List[Tuple[T4, float]]]] = {}

            for cam, cam_markers in enumerate(frame_data):
                if cam in self.excluded_cams or not cam_markers:
                    continue
                cfg = self.cam_configs[cam]
                K, D = cfg.cam_mat, cfg.dist_coeffs

                self.cam_ids.add(cam)

                for marker_id, corners in cam_markers:
                    self.marker_ids.add(marker_id)

                    # Store raw corners for later use
                    self.frame_cam_markers \
                        .setdefault(frame_num, {}) \
                        .setdefault(cam, []) \
                        .append((marker_id, corners.copy()))

                    solutions = _solve_pnp_ippe(
                        self.marker_size, corners, K, D, self.threshold
                    )
                    if not solutions:
                        continue

                    poses_cam \
                        .setdefault(marker_id, {}) \
                        .setdefault(cam, []) \
                        .extend(solutions)
                    poses_marker \
                        .setdefault(cam, {}) \
                        .setdefault(marker_id, []) \
                        .extend(solutions)

            if poses_cam:
                self._frame_poses_cam[frame_num] = poses_cam
            if poses_marker:
                self._frame_poses_marker[frame_num] = poses_marker

    # ------------------------------------------------------------------
    # Phase 2 – build candidate transform sets
    # ------------------------------------------------------------------

    def _fill_transformation_sets_cam(
        self,
        pose_estimations: Dict[int, Dict[int, List[Tuple[T4, float]]]],
    ) -> Dict[int, Dict[int, List[Candidate]]]:
        """Build relative-transform candidates between camera pairs.

        pose_estimations[marker_id][cam_id] = [(T_{cam←marker}, err)]
        Returns sets[id1][id2] with id1 < id2.
        """
        sets: Dict[int, Dict[int, List[Candidate]]] = {}
        for _marker_id, cam_poses in pose_estimations.items():
            cam_ids_seen = sorted(cam_poses.keys())
            if len(cam_ids_seen) < 2:
                continue
            for i, id1 in enumerate(cam_ids_seen):
                for id2 in cam_ids_seen[i + 1:]:
                    for T1, e1 in cam_poses[id1]:
                        for T2, e2 in cam_poses[id2]:
                            # T_{c2←c1}
                            T_rel = T2 @ np.linalg.inv(T1)
                            # T1_inv = T_{c1←marker}
                            # T2_inv = T_{marker←c2}
                            sets.setdefault(id1, {}) \
                                .setdefault(id2, []) \
                                .append((T_rel, T1, np.linalg.inv(T2), e1 * e2))
        return sets

    def _fill_transformation_sets_marker(
        self,
        pose_estimations: Dict[int, Dict[int, List[Tuple[T4, float]]]],
    ) -> Dict[int, Dict[int, List[Candidate]]]:
        """Build relative-transform candidates between marker pairs.

        pose_estimations[cam_id][marker_id] = [(T_{cam←marker}, err)]
        Returns sets[id1][id2] with id1 < id2.
        """
        sets: Dict[int, Dict[int, List[Candidate]]] = {}
        for _cam_id, marker_poses in pose_estimations.items():
            marker_ids_seen = sorted(marker_poses.keys())
            if len(marker_ids_seen) < 2:
                continue
            for i, id1 in enumerate(marker_ids_seen):
                for id2 in marker_ids_seen[i + 1:]:
                    for T1, e1 in marker_poses[id1]:
                        for T2, e2 in marker_poses[id2]:
                            # T_{m2←m1}
                            T_rel = np.linalg.inv(T2) @ T1
                            # T1_inv = T_{m1←cam}
                            # T2_inv = T_{cam←m2}
                            sets.setdefault(id1, {}) \
                                .setdefault(id2, []) \
                                .append((T_rel, np.linalg.inv(T1), T2, e1 * e2))
        return sets

    # ------------------------------------------------------------------
    # Phase 3 – best candidate selection
    # ------------------------------------------------------------------

    @staticmethod
    def _find_best_transformation(marker_size: float,
                                  solutions: List[Candidate]) -> Tuple[int, float]:
        """Select the self-consistent transform by consensus scoring.

        For each candidate T_i, compute the total reconstruction error
        over all observations j:
            err_j = sum_corners || points - T2_j^{-1} · T_i · T1_j^{-1} · points ||

        Returns (best_index, min_error).
        """
        if not solutions:
            return -1, math.inf

        h = marker_size / 2.0
        # Homogeneous corner points (4, 4): rows = [x, y, z, 1], columns = corners
        pts = np.array([
            [-h,  h,  h, -h],
            [h,  h, -h, -h],
            [0,  0,  0,  0],
            [1,  1,  1,  1],
        ], dtype=np.float64)

        min_err = math.inf
        min_idx = 0

        for i, (T_i, _, _, _) in enumerate(solutions):
            curr_err = 0.0
            for _, T1_inv_j, T2_inv_j, _ in solutions:
                p2 = T2_inv_j @ T_i @ T1_inv_j @ pts   # (4, 4)
                diff = pts[:3] - p2[:3]                    # (3, 4)
                curr_err += np.sqrt(np.sum(diff ** 2, axis=0)).sum()
            if curr_err < min_err:
                min_err = curr_err
                min_idx = i

        return min_idx, min_err

    def _find_best_transformations(
        self,
        transform_sets: Dict[int, Dict[int, List[Candidate]]],
    ) -> Dict[int, Dict[int, Tuple[T4, float]]]:
        """Select best transform for every (id1, id2) pair."""
        best: Dict[int, Dict[int, Tuple[T4, float]]] = {}
        for id1, inner in transform_sets.items():
            for id2, solutions in inner.items():
                idx, err = self._find_best_transformation(
                    self.marker_size, solutions)
                if idx >= 0:
                    best.setdefault(id1, {})[id2] = (solutions[idx][0], err)
        return best

    # ------------------------------------------------------------------
    # Phase 4 – Minimum Spanning Tree (Prim's algorithm)
    # ------------------------------------------------------------------

    @staticmethod
    def _make_mst(
        root: int,
        node_ids: Set[int],
        adjacency: Dict[int, Dict[int, Tuple[T4, float]]],
    ) -> Dict[int, Set[int]]:
        """Prim's MST.  Returns children[parent] = {child, …}."""
        distances: Dict[int, float] = {n: math.inf for n in node_ids}
        distances[root] = 0.0
        parents:   Dict[int, int] = {n: -1 for n in node_ids}
        children:  Dict[int, Set[int]] = {n: set() for n in node_ids}
        outside = set(node_ids)

        while outside:
            u = min(outside, key=lambda n: distances[n])
            outside.discard(u)
            for v in list(outside):
                id1, id2 = min(u, v), max(u, v)
                entry = adjacency.get(id1, {}).get(id2)
                if entry is None:
                    continue
                err = entry[1]
                if err < distances[v]:
                    if parents[v] != -1:
                        children[parents[v]].discard(v)
                    parents[v] = u
                    children[u].add(v)
                    distances[v] = err

        return children

    # ------------------------------------------------------------------
    # Phase 5 – transforms to root via BFS
    # ------------------------------------------------------------------

    @staticmethod
    def _find_transforms_to_root(
        root: int,
        children: Dict[int, Set[int]],
        best_transforms: Dict[int, Dict[int, Tuple[T4, float]]],
    ) -> Dict[int, T4]:
        """BFS walk of MST to accumulate transform-to-root for every node."""
        transforms: Dict[int, T4] = {root: np.eye(4, dtype=np.float64)}
        queue: deque[int] = deque([root])

        while queue:
            parent = queue.popleft()
            for child in children.get(parent, set()):
                id1, id2 = min(parent, child), max(parent, child)
                T = best_transforms[id1][id2][0]
                # best_transforms[id1][id2] stores T_{id2←id1}
                if child < parent:
                    # best_transforms[child][parent] → T_{parent←child}
                    T_child_to_parent = T
                else:
                    # best_transforms[parent][child] → T_{child←parent}; invert
                    T_child_to_parent = np.linalg.inv(T)

                transforms[child] = T_child_to_parent
                if parent != root:
                    transforms[child] = transforms[parent] @ transforms[child]
                queue.append(child)

        return transforms

    # ------------------------------------------------------------------
    # Phase 6 – init camera and marker transforms
    # ------------------------------------------------------------------

    def _init_transforms_cam(self) -> None:
        """Aggregate candidates across all frames, then find transforms-to-root."""
        all_sets_cam: Dict[int, Dict[int, List[Candidate]]] = {}
        for frame_poses in self._frame_poses_cam.values():
            frame_sets = self._fill_transformation_sets_cam(frame_poses)
            for id1, inner in frame_sets.items():
                for id2, cands in inner.items():
                    all_sets_cam.setdefault(id1, {}).setdefault(
                        id2, []).extend(cands)

        best_cam = self._find_best_transformations(all_sets_cam)
        self.root_cam = min(self.cam_ids)
        cam_tree = self._make_mst(self.root_cam, self.cam_ids, best_cam)
        self.transforms_to_root_cam = self._find_transforms_to_root(
            self.root_cam, cam_tree, best_cam
        )

    def _init_transforms_marker(self) -> None:
        """Aggregate candidates across all frames, then find transforms-to-root."""
        all_sets_marker: Dict[int, Dict[int, List[Candidate]]] = {}
        for frame_poses in self._frame_poses_marker.values():
            frame_sets = self._fill_transformation_sets_marker(frame_poses)
            for id1, inner in frame_sets.items():
                for id2, cands in inner.items():
                    all_sets_marker.setdefault(
                        id1, {}).setdefault(id2, []).extend(cands)

        best_marker = self._find_best_transformations(all_sets_marker)
        self.root_marker = min(self.marker_ids)
        marker_tree = self._make_mst(
            self.root_marker, self.marker_ids, best_marker)
        self.transforms_to_root_marker = self._find_transforms_to_root(
            self.root_marker, marker_tree, best_marker
        )

    # ------------------------------------------------------------------
    # Phase 7 – per-frame object transforms
    # ------------------------------------------------------------------

    def init_object_transforms(self) -> None:
        """Estimate per-frame object pose (root-marker in root-camera frame)."""
        self.object_transforms.clear()
        for frame, frame_poses in self._frame_poses_cam.items():
            candidates: List[Candidate] = []
            for marker_id, cam_poses_map in frame_poses.items():
                T_mr = self.transforms_to_root_marker.get(marker_id, np.eye(4))
                T_rm = np.linalg.inv(T_mr)
                for cam_id, pose_list in cam_poses_map.items():
                    T_cr = self.transforms_to_root_cam.get(cam_id, np.eye(4))
                    T_rc = np.linalg.inv(T_cr)
                    for T_mc, err in pose_list:
                        T_cm = np.linalg.inv(T_mc)
                        # T_{root_cam←root_marker} = T_{root_cam←cam} @ T_{cam←marker} @ T_{marker←root_marker}
                        T_obj = T_cr @ T_mc @ T_rm
                        candidates.append((T_obj, T_mr @ T_cm, T_rc, err))

            if candidates:
                idx, _ = self._find_best_transformation(
                    self.marker_size, candidates)
                if idx >= 0:
                    self.object_transforms[frame] = candidates[idx][0]

    # ------------------------------------------------------------------
    # Full init pipeline
    # ------------------------------------------------------------------

    def _init_transforms(self) -> None:
        self._init_transforms_cam()
        self._init_transforms_marker()
        self.init_object_transforms()
