"""detect_markers – detect ArUco markers in a dataset and write aruco.detections.

Usage:
    python -m apps.detect_markers <path_to_data_folder> [-d <dictionary>]

Mirrors C++ detect_markers app.
"""

import argparse
import sys
import time
from collections import Counter

import cv2

from automatic_ar.aruco_serdes import write_detections_file
from automatic_ar.dataset import Dataset
from automatic_ar.image_array_detector import _SingleDetector, get_aruco_dictionary


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Detect ArUco markers in a multi-camera dataset.'
    )
    parser.add_argument('folder', help='Path to dataset folder')
    parser.add_argument('-d', '--dictionary', default='ARUCO_DICT_6X6_100',
                        help='ArUco dictionary name (default: ARUCO_DICT_6X6_100)')
    parser.add_argument('-e', '--exclude_marker_ids', default=[],
                        help='Comma-separated list of marker IDs to exclude')
    parser.add_argument('-p', '--visualise', default=False, action='store_true',
                        help='Visualise the detected markers')
    args = parser.parse_args()

    folder_path = args.folder
    exclude_marker_ids = eval(args.exclude_marker_ids)
    visualise = args.visualise
    dict_name = args.dictionary
    output_path = folder_path + '/aruco.detections'
    # matches C++ detect_markers (keeps all markers seen by ≥1 cam)
    min_detections = 1

    dataset = Dataset(folder_path)
    num_cams = dataset.get_num_cams()

    aruco_dict = get_aruco_dictionary(dict_name)
    detectors = [_SingleDetector(aruco_dict) for _ in range(num_cams)]

    frame_nums = dataset.get_frame_nums()
    all_detections = []
    all_marker_ids = set()

    for frame_num in frame_nums:
        print(f'frame num: {frame_num}')
        t0 = time.perf_counter()
        frames = dataset.get_frame(frame_num)

        # Detect per camera
        cam_markers = []
        for cam in range(num_cams):
            img = frames[cam]
            if img is None:
                cam_markers.append([])
            else:
                cam_markers.append(detectors[cam].detect(img))

        elapsed = time.perf_counter() - t0

        # Count camera occurrences per marker
        counts: Counter = Counter()
        for markers in cam_markers:
            for mid, _ in markers:
                counts[mid] += 1

        # Filter by min_detections
        filtered = []
        for cam in range(num_cams):
            kept = [(mid, c) for mid, c in cam_markers[cam]
                    if counts[mid] >= min_detections and (exclude_marker_ids is None or mid not in exclude_marker_ids)]
            filtered.append(kept)
            for mid, _ in kept:
                all_marker_ids.add(mid)

        # Visualise (non-blocking)
        if visualise:
            for cam in range(num_cams):
                img = frames[cam]
                if img is not None:
                    for mid, corners in filtered[cam]:
                        pts = corners.reshape(4, 2).astype(int)
                        for j in range(4):
                            cv2.line(img, tuple(pts[j]), tuple(
                                pts[(j + 1) % 4]), (0, 255, 0), 2)
                    cv2.imshow(f'cam_{cam}', img)
            cv2.waitKey(1)

        all_detections.append(filtered)

    cv2.destroyAllWindows()

    print('Detected marker ids:', sorted(all_marker_ids))
    write_detections_file(output_path, all_detections)
    print(f'Detections written to: {output_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
