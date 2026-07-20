"""track – real-time object tracking using a precomputed solution.

Usage:
    python -m apps.track <data_folder_path> <path_to_solution_file>

Mirrors C++ track app.  For each frame:
  1. Detect markers across all cameras.
  2. Estimate initial object pose from known camera/marker geometry.
  3. Refine the object pose (6-DOF) via Levenberg-Marquardt.
  4. Overlay tracked marker outlines and show a mosaic window.
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
        description='Real-time object tracking using a precomputed AR solution.'
    )
    parser.add_argument('folder',   help='Path to dataset folder to track')
    parser.add_argument(
        'solution', help='Path to .solution file (from find_solution)')
    parser.add_argument(
        '--reference-marker-id', type=int, default=None,
        help='Reference marker id for relative marker vectors (default: root marker)'
    )
    parser.add_argument(
        '--relative-output', choices=['stdout', 'csv', 'both', 'none'],
        default='stdout',
        help='Where to output relative marker vectors and absolute distances'
    )
    parser.add_argument(
        '--relative-output-path', default=None,
        help='CSV output path when --relative-output is csv or both '
             '(default: <dataset>/relative_vectors.csv)'
    )
    args = parser.parse_args()

    folder_path = args.folder
    solution_path = args.solution

    dataset = Dataset(folder_path)
    num_cams = dataset.get_num_cams()
    frame_nums = dataset.get_frame_nums()
    cam_configs = CamConfig.read_cam_configs(folder_path)

    # Load precomputed solution
    mcm = MultiCamMapper()
    if not mcm.read_solution_file(solution_path):
        print(f'Cannot read solution: {solution_path}', file=sys.stderr)
        return 1

    # Configure: only optimise object pose during tracking
    mcm.set_optmize_flag_cam_poses(False)
    mcm.set_optmize_flag_marker_poses(True)
    mcm.set_optmize_flag_object_poses(False)
    mcm.set_optmize_flag_cam_intrinsics(False)

    ma = mcm.get_mat_arrays()
    transforms_to_root_cam = ma['transforms_to_root_cam']
    transforms_to_root_marker = ma['transforms_to_root_marker']
    reference_marker_id = (
        args.reference_marker_id
        if args.reference_marker_id is not None
        else mcm.get_root_marker()
    )
    if reference_marker_id not in transforms_to_root_marker:
        print(
            f'Reference marker id {reference_marker_id} not found in solution.',
            file=sys.stderr
        )
        return 1

    # Marker-to-reference vectors are static in the solved rigid marker model.
    T_root_from_ref = transforms_to_root_marker[reference_marker_id]
    T_ref_from_root = np.linalg.inv(T_root_from_ref)
    marker_relative_vectors = {}
    for marker_id, T_root_from_marker in transforms_to_root_marker.items():
        T_ref_from_marker = T_ref_from_root @ T_root_from_marker
        marker_relative_vectors[marker_id] = T_ref_from_marker[:3, 3].copy()

    relative_output = args.relative_output
    relative_output_path = args.relative_output_path
    csv_file = None
    csv_writer = None
    if relative_output in ('csv', 'both'):
        csv_path = Path(relative_output_path) if relative_output_path else (
            Path(folder_path) / 'relative_vectors.csv'
        )
        csv_file = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            'frame_num', 'reference_marker_id', 'marker_id',
            'dx', 'dy', 'dz', 'abs_distance'
        ])
        print(f'Writing relative vectors to CSV: {csv_path}')

    # Set up live detector (min_detections=2: marker must be seen by ≥2 cameras)
    iad = ImageArrayDetector(num_cams)

    # Build a tracking Initializer (no detections yet, just holds transforms)
    tracker_init = Initializer(
        marker_size=mcm.get_marker_size(),
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
        # Pack into [frame][cam] list expected by Initializer
        frame_detections = [detected]
        detected_marker_ids = sorted(
            {
                marker_id
                for cam_markers in detected
                for marker_id, _ in cam_markers
                if marker_id in marker_relative_vectors
            }
        )

        # Count useful detections
        total_det = sum(len(m) for m in detected)
        sum_detect += time.perf_counter() - t_start
        rows = []
        if total_det > 0:
            for marker_id in detected_marker_ids:
                if marker_id == reference_marker_id:
                    continue
                vec = marker_relative_vectors[marker_id]
                abs_distance = float(np.linalg.norm(vec))
                rows.append((
                    frame_num,
                    reference_marker_id,
                    marker_id,
                    float(vec[0]),
                    float(vec[1]),
                    float(vec[2]),
                    abs_distance,
                ))
                distance_history.setdefault(marker_id, []).append(
                    (frame_num, abs_distance)
                )

        t_inf = time.perf_counter()
        if total_det > 0:
            num_wi_frames += 1
            tracker_init.set_detections(frame_detections)
            tracker_init.obtain_pose_estimations()
            tracker_init.init_object_transforms()

            mcm.init_tracking(
                tracker_init.get_object_transforms(),
                tracker_init.get_frame_cam_markers(),
            )
            # mcm.track()
            mcm.track_with_fallback(mode='adaptive')
            sum_inf += time.perf_counter() - t_inf

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
                mcm.overlay_markers(img, 0, cam)
            frames[cam] = img

        mosaic = make_mosaic(frames, 1536)
        if mosaic is not None:
            header = f'Ref marker: {reference_marker_id} | distances (m)'
            cv2.putText(mosaic, header, (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 255), 2)
            if rows:
                for idx, row in enumerate(rows[:8]):
                    marker_id = row[2]
                    abs_distance = row[6]
                    color = _marker_color(marker_id)
                    cv2.putText(
                        mosaic,
                        f'm{marker_id}: {abs_distance:.4f} m',
                        (20, 60 + idx * 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2,
                    )
            else:
                cv2.putText(mosaic, 'No marker distances', (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow('Tracking', mosaic)
        _draw_distance_plot(distance_plot, distance_history,
                            frame_min, frame_max)
        cv2.imshow('Distance Plot', distance_plot)
        cv2.waitKey(1)

        if total_det > 0 and relative_output != 'none':
            if rows and relative_output in ('stdout', 'both'):
                for row in rows:
                    (
                        out_frame_num, out_reference_marker_id, out_marker_id,
                        dx, dy, dz, abs_distance
                    ) = row
                    print(
                        f'frame={out_frame_num} ref={out_reference_marker_id} '
                        f'marker={out_marker_id} '
                        f'dx={dx:.6f} dy={dy:.6f} dz={dz:.6f} '
                        f'abs_distance={abs_distance:.6f}'
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
