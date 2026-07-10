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
import math
import sys
import time
from typing import List, Optional

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


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Real-time object tracking using a precomputed AR solution.'
    )
    parser.add_argument('folder',   help='Path to dataset folder to track')
    parser.add_argument(
        'solution', help='Path to .solution file (from find_solution)')
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
    mcm.set_optmize_flag_marker_poses(False)
    mcm.set_optmize_flag_object_poses(True)
    mcm.set_optmize_flag_cam_intrinsics(False)

    ma = mcm.get_mat_arrays()
    transforms_to_root_cam = ma['transforms_to_root_cam']
    transforms_to_root_marker = ma['transforms_to_root_marker']

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

    for frame_num in frame_nums:
        frames = dataset.get_frame(frame_num)
        num_frames += 1

        # --- Detection ---
        t_start = time.perf_counter()
        detected = iad.detect_markers(frames, min_detections=1)
        # Pack into [frame][cam] list expected by Initializer
        frame_detections = [detected]

        # Count useful detections
        total_det = sum(len(m) for m in detected)
        sum_detect += time.perf_counter() - t_start

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
            mcm.track()
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
            cv2.imshow('Tracking', mosaic)
        cv2.waitKey(1)

    cv2.destroyAllWindows()

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
