"""track_independent_markers – track markers on independent rigid objects.

Usage:
    python -m apps.track_independent_markers <data_folder_path> <path_to_solution_file>

For each frame:
  1. Detect markers across all cameras.
  2. For each detected marker:
     - Create a single-marker "pseudo-object" (marker pose = object pose)
     - Refine that marker's pose via Levenberg-Marquardt
  3. Track each marker independently with full 6-DOF freedom.
"""

import argparse
import csv
import math
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from automatic_ar.cam_config import CamConfig
from automatic_ar.dataset import Dataset
from automatic_ar.image_array_detector import ImageArrayDetector
from automatic_ar.initializer import Initializer
from automatic_ar.multicam_mapper import MultiCamMapper


def _triangulate_marker_center(
    cam_configs: List[CamConfig],
    transforms_to_root_cam: Dict[int, np.ndarray],
    detections_by_cam: Dict[int, np.ndarray],
) -> Tuple[np.ndarray, float]:
    """Triangulate marker center position from detections across cameras.

    Args:
        cam_configs: Camera calibration info
        transforms_to_root_cam: Camera poses relative to root camera
        detections_by_cam: {cam_id: corners_4x2} marker corner detections

    Returns:
        (position_3d: np.ndarray, reprojection_error: float)
    """
    if not detections_by_cam:
        return np.array([0.0, 0.0, 0.0]), float('inf')

    # Compute marker center position using DLT (Direct Linear Transform)
    # Each 2D detection contributes 2 linear equations
    A_rows = []

    for cam_id, corners_2d in detections_by_cam.items():
        if cam_id >= len(cam_configs):
            continue

        cam_cfg = cam_configs[cam_id]
        K = np.array(cam_cfg.K, dtype=np.float64)
        T_cam_from_root = transforms_to_root_cam[cam_id]
        P = K @ T_cam_from_root[:3, :]  # 3×4 projection matrix

        # Marker center (average of corners in image space)
        center_2d = corners_2d.mean(axis=0)  # (x, y)

        # Add 2 DLT equations for this 2D point
        x, y = center_2d
        A_rows.append([P[0, :] - x * P[2, :]])
        A_rows.append([P[1, :] - y * P[2, :]])

    if len(A_rows) < 4:
        # Not enough observations, can't triangulate reliably
        return np.array([0.0, 0.0, 0.0]), float('inf')

    A = np.vstack(A_rows)
    _, _, VT = np.linalg.svd(A)
    X_hom = VT[-1, :]  # Last row (smallest singular value)
    X = X_hom[:3] / (X_hom[3] + 1e-10)  # Dehomogenize

    return X, 0.0  # Simplified: no reprojection error calculation


def make_mosaic(images: List[Optional[np.ndarray]], mosaic_width: int) -> Optional[np.ndarray]:
    """Tile images into a square mosaic of total width *mosaic_width* pixels."""
    valid = [img for img in images if img is not None]
    if not valid:
        return None
    n = len(valid)
    side = math.ceil(math.sqrt(n))
    h0, w0 = valid[0].shape[:2]
    tile_w = mosaic_width // side
    tile_h = int(tile_w * h0 / w0)
    cols = side
    rows = math.ceil(n / side)
    mosaic = np.zeros((tile_h * rows, tile_w * cols, 3), dtype=np.uint8)
    for i, img in enumerate(valid):
        r, c = divmod(i, cols)
        resized = cv2.resize(img, (tile_w, tile_h))
        mosaic[r * tile_h:(r + 1) * tile_h, c *
               tile_w:(c + 1) * tile_w] = resized
    return mosaic


def _marker_color(marker_id: int) -> Tuple[int, int, int]:
    # Stable pseudo-random-ish color per marker id (BGR for OpenCV drawing)
    return (
        (37 * marker_id + 80) % 256,
        (67 * marker_id + 120) % 256,
        (97 * marker_id + 160) % 256,
    )


def _draw_distance_plot(
    plot_img: np.ndarray,
    distance_history: Dict[int, List[Tuple[int, float]]],
    frame_min: int,
    frame_max: int,
) -> None:
    h, w = plot_img.shape[:2]
    plot_img[:] = 255
    pad_l, pad_r, pad_t, pad_b = 70, 20, 30, 45
    x0, x1 = pad_l, w - pad_r
    y0, y1 = h - pad_b, pad_t
    cv2.rectangle(plot_img, (x0, y1), (x1, y0), (235, 235, 235), 1)
    cv2.putText(plot_img, 'abs distance vs frame', (10, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

    frame_span = max(1, frame_max - frame_min)
    max_dist = 0.0
    for points in distance_history.values():
        for _, d in points:
            if d > max_dist:
                max_dist = d
    if max_dist <= 0.0:
        max_dist = 1.0

    def to_xy(frame_num: int, dist: float) -> Tuple[int, int]:
        x = x0 + int((frame_num - frame_min) / frame_span * (x1 - x0))
        y = y0 - int((dist / max_dist) * (y0 - y1))
        return x, y

    cv2.putText(plot_img, f'{frame_min}', (x0 - 10, y0 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.putText(plot_img, f'{frame_max}', (x1 - 20, y0 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.putText(plot_img, f'{max_dist:.3f} m', (5, y1 + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.putText(plot_img, '0', (35, y0 + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    for marker_id, points in sorted(distance_history.items()):
        if len(points) < 1:
            continue
        color = _marker_color(marker_id)
        if len(points) >= 2:
            poly = np.array([to_xy(fn, d) for fn, d in points], dtype=np.int32)
            cv2.polylines(plot_img, [poly], False, color, 2)
        x_last, y_last = to_xy(points[-1][0], points[-1][1])
        cv2.circle(plot_img, (x_last, y_last), 3, color, -1)
        cv2.putText(plot_img, f'm{marker_id}', (x_last + 4, y_last - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Track independent markers on separate rigid objects.'
    )
    parser.add_argument('folder',   help='Path to dataset folder to track')
    parser.add_argument(
        'solution', help='Path to .solution file (from find_solution)')
    parser.add_argument(
        '--reference-marker-id', type=int, default=None,
        help='Reference marker id for relative distances (default: first tracked marker)'
    )
    parser.add_argument(
        '--relative-output', choices=['stdout', 'csv', 'both', 'none'],
        default='stdout',
        help='Where to output marker poses and relative distances'
    )
    parser.add_argument(
        '--relative-output-path', default=None,
        help='CSV output path when --relative-output is csv or both '
             '(default: <dataset>/marker_poses.csv)'
    )
    parser.add_argument(
        '--verbose', default=True,
    )
    args = parser.parse_args()

    verbose = args.verbose
    folder_path = args.folder
    solution_path = args.solution

    dataset = Dataset(folder_path)
    num_cams = dataset.get_num_cams()
    frame_nums = dataset.get_frame_nums()
    cam_configs = CamConfig.read_cam_configs(folder_path)

    # Load precomputed solution (to get camera calibration)
    mcm_template = MultiCamMapper()
    if not mcm_template.read_solution_file(solution_path):
        print(f'Cannot read solution: {solution_path}', file=sys.stderr)
        return 1

    ma_template = mcm_template.get_mat_arrays()
    transforms_to_root_cam = ma_template['transforms_to_root_cam']
    transforms_to_root_marker = ma_template['transforms_to_root_marker']

    relative_output = args.relative_output
    relative_output_path = args.relative_output_path
    reference_marker_id = args.reference_marker_id
    csv_file = None
    csv_writer = None
    if relative_output in ('csv', 'both'):
        csv_path = Path(relative_output_path) if relative_output_path else (
            Path(folder_path) / 'marker_poses.csv'
        )
        csv_file = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            'frame_num', 'reference_marker_id', 'marker_id',
            'tx', 'ty', 'tz', 'rx_deg', 'ry_deg', 'rz_deg',
            'rel_dx', 'rel_dy', 'rel_dz', 'rel_distance'
        ])
        print(f'Writing marker poses to CSV: {csv_path}')

    # Set up live detector
    iad = ImageArrayDetector(num_cams)

    # Build tracking initializer
    tracker_init = Initializer(
        marker_size=mcm_template.get_marker_size(),
        cam_configs=cam_configs,
    )
    tracker_init.set_transforms_to_root_cam(transforms_to_root_cam)
    tracker_init.set_transforms_to_root_marker(transforms_to_root_marker)

    num_frames = num_wi_frames = 0
    sum_detect = sum_inf = sum_total = 0.0
    frame_min = min(frame_nums) if frame_nums else 0
    frame_max = max(frame_nums) if frame_nums else 1
    distance_history: Dict[int, List[Tuple[int, float]]] = {}
    distance_plot = np.full((420, 800, 3), 255, dtype=np.uint8)

    for frame_num in frame_nums:
        frames = dataset.get_frame(frame_num)
        num_frames += 1

        # --- Detection ---
        t_start = time.perf_counter()
        detected = iad.detect_markers(frames, min_detections=1)
        frame_detections = [detected]
        detected_marker_ids = sorted(
            {
                marker_id
                for cam_markers in detected
                for marker_id, _ in cam_markers
                if marker_id in transforms_to_root_marker
            }
        )

        total_det = sum(len(m) for m in detected)
        sum_detect += time.perf_counter() - t_start
        rows = []

        if total_det > 0:
            num_wi_frames += 1
            tracker_init.set_detections(frame_detections)
            tracker_init.obtain_pose_estimations()
            tracker_init.init_object_transforms()

            # Track each marker independently
            t_inf = time.perf_counter()
            marker_poses_this_frame = {}

            # Convert frame_detections to proper dict format: {frame_id: {cam_id: detections}}
            # detected is [cam_0_detections, cam_1_detections, ...] where each is [(marker_id, corners)]
            frame_cam_markers_dict = {}
            for cam_id, cam_markers in enumerate(detected):
                if cam_markers:
                    frame_cam_markers_dict[cam_id] = cam_markers

            for marker_id in detected_marker_ids:
                # Create a single-marker mapper: marker_pose = object_pose
                # This treats each marker as if it were the sole object
                mcm_single = MultiCamMapper()

                # Filter detections to only this marker
                filtered_fcm = {}
                obs_count = 0  # Track total observations for diagnostics
                for cam_id, markers in frame_cam_markers_dict.items():
                    filtered_markers = [(mid, corners) for mid, corners in markers
                                        if mid == marker_id]
                    if filtered_markers:
                        filtered_fcm[cam_id] = filtered_markers
                        # 4 corner points × 2 coords each
                        obs_count += len(filtered_markers) * 4

                if not filtered_fcm:
                    # No observations for this marker in this frame
                    continue

                # Estimate residual count: 4 corners × 2 coords per camera view
                estimated_residuals = obs_count * 2
                if verbose:
                    print(f'  Marker {marker_id}: {len(filtered_fcm)} cameras, '
                          f'{obs_count} detections (~{estimated_residuals} residuals)',
                          file=sys.stderr)

                # Get object poses and ensure frame ID matches
                object_poses = tracker_init.get_object_transforms()
                # Remap frame IDs to 0 for single-frame optimization
                object_poses_remapped = {0: list(object_poses.values())[
                    0] if object_poses else np.eye(4)}

                # Use template transforms
                try:
                    mcm_single.init(
                        root_cam=0,
                        transforms_to_root_cam=transforms_to_root_cam,
                        root_marker=marker_id,  # This marker is the "root" for this tracker
                        transforms_to_root_marker={
                            marker_id: np.eye(4, dtype=np.float64)},
                        object_poses=object_poses_remapped,  # Use remapped poses with frame_id=0
                        # Frame ID must match
                        frame_cam_markers={0: filtered_fcm},
                        marker_size=mcm_template.get_marker_size(),
                        cam_configs=cam_configs,
                    )
                except Exception as e:
                    print(f'Warning: failed to initialize marker {marker_id}: {e}',
                          file=sys.stderr)
                    continue

                # Check if we have enough observations to solve for 6 DOF
                # Each 2D point gives 2 residuals, so need at least 3 observations (6 residuals)
                num_residuals = mcm_single.num_point_xys
                if num_residuals < 6:
                    num_cameras = len(filtered_fcm)
                    print(f'Warning: marker {marker_id} frame {frame_num}: '
                          f'only {num_residuals} residuals from {num_cameras} camera(s) '
                          f'(need ≥6 for LM solver). Using triangulation fallback.', file=sys.stderr)

                    # Fallback: triangulate marker center from detections
                    detections_by_cam = {}
                    for cam_id, markers in filtered_fcm.items():
                        if markers:  # markers is [(marker_id, corners_4x2), ...]
                            # Get corners array
                            detections_by_cam[cam_id] = markers[0][1]

                    center_3d, _ = _triangulate_marker_center(
                        cam_configs, transforms_to_root_cam, detections_by_cam
                    )

                    T_marker = np.eye(4, dtype=np.float64)
                    T_marker[:3, 3] = center_3d
                    marker_poses_this_frame[marker_id] = T_marker
                    detected_marker_ids.add(marker_id)
                    continue

                # Optimize only this marker's pose
                mcm_single.set_optmize_flag_cam_poses(False)
                mcm_single.set_optmize_flag_marker_poses(False)
                mcm_single.set_optmize_flag_object_poses(True)
                mcm_single.set_optmize_flag_cam_intrinsics(False)

                try:
                    mcm_single.track()
                    mat_arrays = mcm_single.get_mat_arrays()
                    # Get the optimized object pose (= marker pose)
                    frame_keys = sorted(mat_arrays['object_to_global'].keys())
                    if frame_keys:
                        T_marker = mat_arrays['object_to_global'][frame_keys[0]]
                        marker_poses_this_frame[marker_id] = T_marker

                        # Compute distance from origin
                        t = T_marker[:3, 3]
                        distance = float(np.linalg.norm(t))
                        distance_history.setdefault(marker_id, []).append(
                            (frame_num, distance)
                        )

                        # Convert rotation matrix to Euler angles (degrees)
                        R = T_marker[:3, :3]
                        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
                        singular = sy < 1e-6
                        if not singular:
                            rx = np.arctan2(R[2, 1], R[2, 2])
                            ry = np.arctan2(-R[2, 0], sy)
                            rz = np.arctan2(R[1, 0], R[0, 0])
                        else:
                            rx = np.arctan2(-R[1, 2], R[1, 1])
                            ry = np.arctan2(-R[2, 0], sy)
                            rz = 0

                        rows.append((
                            frame_num,
                            reference_marker_id or 0,  # Will be updated later
                            marker_id,
                            float(t[0]),
                            float(t[1]),
                            float(t[2]),
                            float(np.degrees(rx)),
                            float(np.degrees(ry)),
                            float(np.degrees(rz)),
                            0.0,  # rel_dx placeholder
                            0.0,  # rel_dy placeholder
                            0.0,  # rel_dz placeholder
                            0.0,  # rel_distance placeholder
                        ))
                except Exception as e:
                    print(f'Warning: failed to track marker {marker_id}: {e}',
                          file=sys.stderr)

            sum_inf += time.perf_counter() - t_inf

            # Compute relative distances between markers in this frame
            if marker_poses_this_frame:
                # Choose reference marker: first tracked marker if not specified
                ref_marker_id = reference_marker_id
                if ref_marker_id is None or ref_marker_id not in marker_poses_this_frame:
                    ref_marker_id = min(marker_poses_this_frame.keys())

                if ref_marker_id in marker_poses_this_frame:
                    T_ref = marker_poses_this_frame[ref_marker_id]
                    ref_pos = T_ref[:3, 3]

                    for marker_id in detected_marker_ids:
                        if marker_id not in marker_poses_this_frame:
                            continue

                        T_marker = marker_poses_this_frame[marker_id]
                        marker_pos = T_marker[:3, 3]
                        rel_vec = marker_pos - ref_pos
                        rel_distance = float(np.linalg.norm(rel_vec))

                        # Find and update the row for this marker
                        for i, row in enumerate(rows):
                            # row[2] is marker_id (after ref_marker_id)
                            if row[2] == marker_id:
                                # Update reference marker and relative distances
                                rows[i] = row[:1] + (ref_marker_id,) + row[2:9] + (
                                    float(rel_vec[0]),
                                    float(rel_vec[1]),
                                    float(rel_vec[2]),
                                    rel_distance,
                                )
                                break

                        # Update distance history with relative distance
                        if marker_id != ref_marker_id:
                            distance_history.setdefault(marker_id, []).append(
                                (frame_num, rel_distance)
                            )

        sum_total += time.perf_counter() - t_start

        # --- Overlay ---
        for cam in range(num_cams):
            img = frames[cam]
            if img is None:
                continue
            if total_det == 0:
                cv2.putText(img, 'no reliable detections',
                            (100, 100), cv2.FONT_HERSHEY_SIMPLEX,
                            1.5, (0, 0, 255), 3)
            else:
                # Draw marker positions from this frame's poses
                for marker_id, T_marker in marker_poses_this_frame.items():
                    # Project marker corners using the optimized pose
                    T_local_cam = np.linalg.inv(transforms_to_root_cam[cam])
                    T_proj = T_local_cam @ T_marker
                    rvec, _ = cv2.Rodrigues(T_proj[:3, :3])
                    tvec = T_proj[:3, 3]
                    K = cam_configs[cam].cam_mat
                    D = cam_configs[cam].dist_coeffs
                    marker_size = mcm_template.get_marker_size()
                    h = marker_size / 2.0
                    pts_3d = np.array([
                        [-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]
                    ], dtype=np.float32)
                    pts_2d, _ = cv2.projectPoints(pts_3d, rvec, tvec, K, D)
                    pts = pts_2d.reshape(4, 2).astype(np.int32)
                    color = _marker_color(marker_id)
                    for j in range(4):
                        cv2.line(img, tuple(pts[j]), tuple(pts[(j + 1) % 4]),
                                 color, 2)
            frames[cam] = img

        mosaic = make_mosaic(frames, 1536)
        if mosaic is not None:
            header = f'Independent marker tracking | relative distances (m)'
            cv2.putText(mosaic, header, (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 255), 2)
            if rows:
                for idx, row in enumerate(rows[:8]):
                    ref_marker_id = row[1]
                    marker_id = row[2]
                    rel_distance = row[12]  # Last element is rel_distance
                    color = _marker_color(marker_id)
                    cv2.putText(
                        mosaic,
                        f'm{marker_id}: d={rel_distance:.4f}m (ref=m{ref_marker_id})',
                        (20, 60 + idx * 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        color,
                        2,
                    )
            else:
                cv2.putText(mosaic, 'No marker poses', (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow('Independent Marker Tracking', mosaic)
        _draw_distance_plot(distance_plot, distance_history,
                            frame_min, frame_max)
        cv2.imshow('Distance Plot', distance_plot)
        cv2.waitKey(1)

        if total_det > 0 and relative_output != 'none':
            if rows and relative_output in ('stdout', 'both'):
                for row in rows:
                    (out_frame_num, out_ref_marker_id, out_marker_id, tx, ty, tz,
                     rx_deg, ry_deg, rz_deg, rel_dx, rel_dy, rel_dz, rel_distance) = row
                    print(
                        f'frame={out_frame_num} ref={out_ref_marker_id} marker={out_marker_id} '
                        f'tx={tx:.6f} ty={ty:.6f} tz={tz:.6f} '
                        f'rx={rx_deg:.2f}° ry={ry_deg:.2f}° rz={rz_deg:.2f}° '
                        f'rel_dx={rel_dx:.6f} rel_dy={rel_dy:.6f} rel_dz={rel_dz:.6f} '
                        f'rel_distance={rel_distance:.6f}'
                    )
            if rows and csv_writer is not None:
                csv_writer.writerows(rows)

    cv2.destroyAllWindows()
    if csv_file is not None:
        csv_file.close()

    if num_frames > 0:
        print(
            f'Average per-frame time : {sum_total/num_frames:.4f}s over {num_frames} frames')
    if num_wi_frames > 0:
        print(
            f'Average inference time : {sum_inf/num_wi_frames:.4f}s over {num_wi_frames} frames')
        print(f'Average detection time : {sum_detect/num_frames:.4f}s')

    return 0


if __name__ == '__main__':
    sys.exit(main())
